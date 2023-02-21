import torch
import time
import pygame
import numpy as np
import sys

from cooperative_transport.gym_table.envs.custom_rewards import custom_reward_function
from cooperative_transport.gym_table.envs.utils import (
    init_joystick,
    debug_print,
    FPS,
    CONST_DT,
    MAX_FRAMESKIP,
    get_keys_to_action,
    set_action_keyboard,
)
from libs.planner.planner_utils import (
    pid_single_step,
    update_queue,
    tf2model,
    tf2sim,
)

def compute_reward(
    states, goal, obs, interaction_forces=False, vectorized=True
) -> float:
    """
    Compute reward for the given state and goal.

    Args:
        states (np.ndarray): shape (N, obs_dim). Evaluating the reward for each state 
                in the batch of size N.
        goal (np.ndarray): shape (2, ). Goal position
        obs (np.ndarray): shape (num_obs, 2). Obstacles positions for each obstacle
        interaction_forces (bool): Whether to use interaction forces in rewards
        vectorized (bool): Whether to vectorize the reward computation. 
                In inference, this should be True since we want to sample from the model.
    """
    reward = custom_reward_function(
        states, goal, obs, interaction_forces=interaction_forces, vectorized=True
    )
    return reward

def play_hil_planner(
    env,
    exp_run_mode="hil",
    human="data",
    robot="planner",
    model=None,
    mcfg=None,
    SEQ_LEN=120,
    H=30,
    playback_trajectory=None,
    n_steps=1000,
    fps=FPS,
    display_pred=False,
    display_gt=False,
    display_past_states=False,
    device="cpu",
    include_interaction_forces_in_rewards=False,
):
    """
    Play a with a trained agent on the gym environment.

        Robot is Player 1 (blue), and human is player 2 (orange).

        Two options for human player:

        1) Human (you, the human) plays with the trained planner (robot). If keyboard mode, use "WASD" to move (NOT arrow keys).
        2) Human data is played back with the trained planner (robot)
        3) Human trained BC policy is played with the trained planner (robot) TODO: add this feature

        Each run begins with a reset of the environment, provided a configuration from a previous
        rollout / demo / ground truth (GT) trajectory. The first H steps from the GT trajectory
        are played out in the enviroment, and can be used for comparison. It is also the trajectory
        that the human plays if running this function in option 2).


        Args:
            env: gym environment
            exp_run_mode: "hil" if human in the loop, "replay_traj" if replaying data
            human: "data" if option 2), "real" if option 1), "policy" if option 3)
            model: trained planner model
            mcfg: model config
            SEQ_LEN: length of sequence for the model to use for planning. Prediction period is SEQ_LEN - H
            H: observation period. THe model uses the past H observations to predict the next SEQ_LEN - H actions
            playback_trajectory: if not None, then this is the trajectory to play back for option 2)
            n_steps: max number of steps to run the episode for
            fps: frames per second
            display_pred: if True, then display the predicted trajectory on the pygame window
            display_gt: if True, then display the ground truth trajectory on the pygame window.
    """
    # -------------------------------------------- SETUP SAVED DATA -------------------------------------------- #

    # Initialize trajectory dictionary for storing states, actions, rewards, etc.
    trajectory = {}
    trajectory["states"] = []
    trajectory["plan"] = []
    trajectory["actions"] = []
    trajectory["rewards"] = []
    trajectory["fluency"] = []

    # -------------------------------------------- CHECK EXPERIMENT ARGS -------------------------------------------- #

    # Initialize human input controller and check for valid experimental setup args passed
    assert human in [
        "data",
        "real",
        "policy",
    ], "human arg must be one of 'data', 'policy', or 'real'"
    if human == "real":
        if env.control_type == "joystick":
            joysticks = init_joystick()
            p2_id = 0
        elif env.control_type == "keyboard":
            keys_to_action = get_keys_to_action()
            relevant_keys = set(sum(map(list, keys_to_action.keys()), []))
            pressed_keys = []
        else:
            raise ValueError("control_type must be 'joystick' or 'keyboard'")
    elif human == "policy":
        raise NotImplementedError("BC policy not implemented yet")
    else:
        assert playback_trajectory is not None, "Must provide playback trajectory"
        assert (
            human == "data"
        ), "human arg must be from 'data' if not 'real' or 'policy'"
    # Set n_steps to data limit if using human or robot data as control inputs
    if human == "data" or robot == "data":
        n_steps = len(playback_trajectory["actions"]) - 1
    
    # Check for valid robot arg
    assert robot in ["planner", "data"], "robot arg must be one of 'planner' or 'data'"

    # ----------------------------------------------- SETUP EXPERIMENT VIS -------------------------------------------- #

    # Initialize list of past states visited in the simulator, for visualization purposes
    if display_past_states:
        past_states = []
        past_states.append(obs.tolist())

    # Initialize list of ground truth waypoints, if displaying ground truth
    if display_gt:
        waypoints_true = playback_trajectory["states"].tolist()

    # ----------------------------------------------------- SIMULTAOR SETUP -----------------------------------------------#

    # reset environment
    obs = env.reset()
    obs = torch.from_numpy(obs).float()
    info = None
    done = False
    n_iter = 0
    running = True
    next_game_tick = time.time()
    clock = pygame.time.Clock()
    success = False
    # Track time used for planning
    delta_plan_sum = 0

    # ------------------------------------------ SETUP DATA STRUCTS FOR MODEL USE ------------------------------------------ #

    # Initialize running list of past H steps of observations for model inputs (need tf2model for model input conversion)
    s_queue = torch.zeros(
        (mcfg.H // mcfg.skip + 1, obs.shape[0]), dtype=torch.float32
    ).to(device)
    s_queue = update_queue(s_queue, obs.unsqueeze(0))
    u_queue = torch.zeros((mcfg.H // mcfg.skip, mcfg.ASIZE), dtype=torch.float32).to(
        device
    )
    # Initialize hidden state
    h = None

    ### ---------------------------------------------------- GAME LOOP ---------------------------------------------------- ###

    start = time.time()
    while running:

        loops = 0

        if env.done:
            time.sleep(1)
            pygame.quit()
            print("Episode finished after {} timesteps".format(n_iter + 1))
            break

        else:

            while time.time() > next_game_tick and loops < MAX_FRAMESKIP and not done:

                if display_gt:
                    env.draw_gt(waypoints_true)

                if h is None:
                    h = torch.zeros(
                        mcfg.n_layers, mcfg.BSIZE, mcfg.RSIZE, device=device
                    )

                # -------------------------------------------- GET HUMAN INPUT -------------------------------------------- #
                if human == "real":

                    if env.control_type == "joystick":
                        u_h = np.array(
                            [
                                joysticks[p2_id].get_axis(0),
                                joysticks[p2_id].get_axis(1),
                            ]
                        )
                        u_h = torch.from_numpy(np.clip(u_h, -1.0, 1.0)).unsqueeze(0)

                    else:
                        u_h = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
                        u_h = set_action_keyboard(u_h)
                        u_h = torch.from_numpy(u_h[1, :]).unsqueeze(0)
                        

                elif human == "policy":

                    pass  # TODO: implement this

                else:
                    # If using human data, then get the actions from the playback trajectory
                    if exp_run_mode == "replay_traj" or human == "data" or robot == "data":
                        n_iter = min(n_iter, playback_trajectory["actions"].shape[0] - 1) # Needed to account finish the playback
                    # else:
                    #     assert n_iter < actions.shape[0], "Ran out of human actions from data."
                    u_h = playback_trajectory["actions"][n_iter, 2:]
                    u_h = torch.from_numpy(u_h).unsqueeze(0)

                
                # --------------------------------------------- GET ROBOT INPUTS -------------------------------------------- #

                # -------------------------------------------- OBSERVATION PERIOD -------------------------------------------- #

                # If we are in the observation period, then we just need to update the state history queue and get the next
                # observation from the simulator by feeding the human input and robot input (which is from human data) to the simulator

                if (n_iter <= mcfg.H + mcfg.skip) or (exp_run_mode == "replay_traj"):

                    if (n_iter % mcfg.skip != 0) and exp_run_mode != "replay_traj":
                        n_iter += 1
                        continue

                    # Feed first H steps of state history into simulator
                    u_r = torch.from_numpy(playback_trajectory["actions"][n_iter, :2]).unsqueeze(0)
                    u_all = torch.cat(
                        (u_r, u_h), dim=-1
                    )  # player 1 is blue, and in this sim human is player 2 (orange)

                    # Update action history queue
                    u_queue = update_queue(u_queue, u_all)

                    # Update env with actions
                    if exp_run_mode == "replay_traj":
                        u_all = playback_trajectory["actions"][n_iter, :]
                    obs, reward, done, info = env.step(list(u_all.squeeze()))

                    # Checks
                    if env.done:
                        if info["success"]:
                            success = True
                        else:
                            success = False
                        env.render(mode="human")
                        running = False
                        trajectory["states"].append(obs)
                        if robot == "planner":
                            trajectory["plan"].append(path)
                        trajectory["actions"].append(u_all)
                        trajectory["rewards"].append(reward)
                        trajectory["fluency"].append(env)
                        break

                    if display_past_states:
                        past_states.append(obs.tolist())
                        env.draw_past_states(past_states)

                    # Update obseravations for model
                    obs = torch.from_numpy(obs).float()
                    s_queue = update_queue(s_queue, obs.unsqueeze(0))
                    n_iter += 1
                    continue

                # -------------------------------------------- PLANNING PERIOD -------------------------------------------- #

                # If we are in the planning period, then we need to continue updating the state history queue, get the next observation
                # from the simulator by feeding the human input and robot input from PID, which controls to waypoints planned by the model.

                else:
                    
                    if robot == "planner":
                        # -------------------------------------------- GET WAYPOINTS -------------------------------------------- #

                        with torch.no_grad():

                            s_tf = tf2model(s_queue).repeat(mcfg.BSIZE, 1, 1)
                            u = u_queue.repeat(mcfg.BSIZE, 1, 1).float()
                            start_plan = time.time()
                            sample = model.sample(
                                s_tf, u, seq_len=mcfg.SEQ_LEN
                            )

                        waypoints = tf2sim(
                            sample[:, :, :4],
                            s_queue,
                            (mcfg.H // mcfg.skip),
                        )

                        # Evaluate the rewards the batch of sampled trajectories using custom reward function
                        eval = np.sum(
                            np.array(
                                [
                                    compute_reward(waypoints[i, :, :3], env.goal, env.obstacles, interaction_forces=include_interaction_forces_in_rewards)
                                    for i in range(waypoints.shape[0])
                                ]
                            ),
                            -1,
                        )

                        # Select the best trajectory
                        best_traj = np.argmax(eval)
                        path = waypoints[best_traj, :, :]

                        end_plan = time.time()
                        delta_plan = end_plan - start_plan
                        delta_plan_sum += delta_plan

                        if display_pred:
                            env.update_prediction(path.tolist())

                        # -------------------------------------------- GET ROBOT CONTROL -------------------------------------------- #

                        pid_actions = pid_single_step(
                            env,
                            path[mcfg.skip, :4],
                            kp=0.15,
                            ki=0.0,
                            kd=0.0,
                            max_iter=40,
                            dt=CONST_DT,
                            eps=1e-2,
                            u_h=u_h.squeeze().numpy(),
                        )
                        pid_actions /= np.linalg.norm(pid_actions)

                        u_r = torch.from_numpy(np.clip(pid_actions, -1.0, 1.0)).unsqueeze(0)

                    else:
                        u_r = torch.from_numpy(playback_trajectory["actions"][n_iter, :2]).unsqueeze(0)

                u_all = torch.cat((u_r, u_h), dim=-1)
                u_queue = update_queue(u_queue, u_all)

                # -------------------------------------------- UPDATE ENVIRONMENT -------------------------------------------- #

                obs, reward, done, info = env.step(list(u_all.squeeze()))
                if display_past_states:
                    past_states.append(obs.tolist())
                    env.draw_past_states(past_states)

                # Update obseravations for model
                obs = torch.from_numpy(obs).float()
                s_queue = update_queue(s_queue, obs.unsqueeze(0))
                
                trajectory["states"].append(obs)
                if robot == "planner":
                    trajectory["plan"].append(torch.tensor(path))
                trajectory["actions"].append(u_all)
                trajectory["rewards"].append(reward)
                trajectory["fluency"].append(env.fluency)
                if env.done:
                    env.reset()
                    if info["success"]:
                        success = True
                    else:
                        success = False
                    env.render(mode="human")
                    running = False    
                    break

                next_game_tick += CONST_DT
                loops += 1

            if loops == 0:
                continue
            else:
                delta_plan_sum = delta_plan_sum / (loops)
            n_iter += 1

            # Update display
            if not env.done:
                env.redraw()
                clock.tick(FPS)

                # process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key in relevant_keys and human == "real" and env.control_type == "keyboard":
                            debug_print("REGISTERED KEY PRESS")
                            pressed_keys.append(event.key)
                        elif event.key == 27:
                            running = False
                    elif event.type == pygame.KEYUP:
                        if event.key in relevant_keys:
                            pressed_keys.remove(event.key)
                    elif event.type == pygame.QUIT:
                        running = False

    stop = time.time()
    duration = stop - start
    print("Average planning time per planning loop: ", delta_plan_sum / n_iter)
    print("Duration of run: ", duration)
    pygame.quit()

    if exp_run_mode != "replay_traj":
        # Save trajectory
        trajectory["states"] = torch.stack(trajectory["states"], dim=0).numpy()
        if robot == "planner":
            trajectory["plan"] = torch.stack(trajectory["plan"], dim=0).numpy()
        trajectory["actions"] = torch.stack(trajectory["actions"], dim=0).numpy()
        trajectory["rewards"] = torch.stack(trajectory["rewards"], dim=0).numpy()
        assert info is not None, "Error: info is None"
        trajectory["fluency"] = info["fluency"]
        trajectory["success"] = info["success"]
        trajectory["done"] = done
        trajectory["n_iter"] = n_iter
        trajectory["duration"] = duration

    return trajectory, success, n_iter, duration
    
