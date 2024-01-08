import copy
import math
import sys
import time
from os.path import dirname, join

import numpy as np
import pygame
import rospy
import tf
import tf2_ros
import torch
import wandb
from geometry_msgs.msg import PoseStamped, Twist

# from algo.planners.cooperative_planner.models import VRNN
from cooperative_transport.gym_table.envs.custom_rewards import \
    custom_reward_function
from cooperative_transport.gym_table.envs.utils import (CONST_DT, FPS,
                                                        MAX_FRAMESKIP,
                                                        WINDOW_H, WINDOW_W, I,
                                                        L, b, d, debug_print,
                                                        get_keys_to_action,
                                                        init_joystick, m,
                                                        set_action_keyboard)
from libs.planner.planner_utils import (is_safe, pid_single_step, tf2model,
                                        tf2sim, update_queue)
from libs.real_robot_utils import (MAX_X, MAX_Y, MIN_X, MIN_Y, compute_reward,
                                   euler_from_quaternion, move, reset_odom,
                                   share_control)

FPS = 30 # ticks per second
new_fps = 10
CONST_DT = 1/FPS #1/FPS # skip ticks
MAX_FRAMESKIP = 3 # max frames to skip
SKIP_FRAME = FPS // new_fps





def mocap_cb(msg):
    global mocap_table_pose
    global prev_mocap_table_pose
    global world_inertial_frame
    global table_center_frame_name
    mocap_table_pose = msg.pose

    #For Real world:
    world_inertial_frame = rospy.get_param('world_frame_name', "nice_origin") #this is origin in sim
    table_center_frame_name= rospy.get_param('table_center_frame_name', "table_carried")

# Initialize node
rospy.init_node("locobot" + "_coop_carry")
rate = rospy.Rate(FPS)
reset_odom()

# while not rospy.is_shutdown():
#     rospy.loginfo("DT check")
#     rate.sleep()

# Create Publisher/Subscribers
# SUBSCRIBERS -- table pose
global sub_table_pose, pub_p_des_vel
sub_table_pose = rospy.Subscriber("/vrpn_client_node/table_carried/pose", PoseStamped, mocap_cb)
# PUBLISHERS -- table velocity and human command
pub_p_des_vel = rospy.Publisher("/table_velocity", Twist, queue_size=1)

# Create tf buffer and listener to nice_origin (RH rule)
global tfBuffer, listener
tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer)



def compute_reward(
    states,
    goal,
    obs,
    env=None,
    vectorized=False,
    interaction_forces=None,
    u_r=None,
    u_h=None,
    collision=None,
    collision_checking_env=None,
    success=None,
) -> float:
    """
    Compute reward for the given state and goal.

    Args:
        states (np.ndarray): shape (N, obs_dim). Evaluating the reward for each state
                in the batch of size N.
        goal (np.ndarray): shape (2, ). Goal position
        obs (np.ndarray): shape (num_obs, 2). Obstacles positions for each obstacle
        include_interaction_forces_in_reward (bool): If True, include the interaction forces in the reward
        interaction_forces (float): If provided, use the interaction forces computed as a part of the reward fn
        vectorized (bool): Whether to vectorize the reward computation.
                In inference, this should be True since we want to sample from the model.
    """
    if env.include_interaction_forces_in_rewards:
        reward = custom_reward_function(
            states,
            goal,
            obs,
            interaction_forces=interaction_forces,
            vectorized=True,
            collision_checking_env=collision_checking_env,
            env=env,
            u_h=u_h,
        )
    else:
        reward = custom_reward_function(
            states, goal, obs, vectorized=True, env=env, collision_checking_env=collision_checking_env, u_h=u_h
        )
    return reward


def play_hil_planner(
    env,
    exp_run_mode="hil",
    human="data",
    robot="planner",
    planner_type="vrnn",
    artifact_path=None,
    mcfg=None,
    SEQ_LEN=120,
    H=30,
    skip=5,
    num_candidates=64,
    playback_trajectory=None,
    n_steps=1000,
    fps=FPS,
    collision_checking_env=None,
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

    def transform_lookup(frame_parent=None, frame_child=None, time=rospy.Time(), past_time=None):
        """ Return transform from frame_parent to frame_child
        frame_parent, frame_child: string, name of frames
        Return: transform from frame_parent to frame_child
        """
        trans = None
        try:
            if past_time is None:
                trans = tfBuffer.lookup_transform(frame_parent, frame_child, time)
            else:
                trans = tfBuffer.lookup_transform_full(frame_child, past_time, frame_child, rospy.Time(), frame_parent)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("TF exception for frames ", frame_parent, " and ", frame_child)
        if type(trans) == type(None): rospy.logwarn("Transform wasn't set!!")

        return trans

    
    # -------------------------------------------- INIT GLOBAL REAL WORLD FUNCS -------------------------------------------- #
    def mocap_pose_to_obs(obs, map_info, grid, past_rod_center_vec=None, past_rod_center_pose=None):
        """ Convert mocap pose to observation in sim
        obs: np.array (7 + 5 + 6 = 18, )
        table x, y, cth, sth, xspeed, yspeed, ang_speed, map_info (5), grid (6)
        Return: np.array (7 + 5 + 6 = 18, )
        """
        start = time.time()
        
        rod_center_vect = transform_lookup(frame_parent=world_inertial_frame, frame_child=table_center_frame_name, time=rospy.Time())
        past_rod_center_vec = rod_center_vect
        tf_y = rod_center_vect.transform.translation.y
        tf_y = WINDOW_H - ((tf_y - MIN_Y) / (MAX_Y - MIN_Y) * WINDOW_H)
        position = [rod_center_vect.transform.translation.x, tf_y]
        theta = tf.transformations.euler_from_quaternion([rod_center_vect.transform.rotation.x, 
                                    rod_center_vect.transform.rotation.y, 
                                    rod_center_vect.transform.rotation.z, 
                                    rod_center_vect.transform.rotation.w])[2] #mocap theta
        ang_tf = np.pi / 2 # mocap says this starts at pi/2, so i need to add pi/2 to make it reflect sim start at pi
        sim_theta = (theta + ang_tf) % (2 * np.pi)
        # calc velocity ## TODO: add velocity pose message
        # past_step_time = rod_center_vect.header.stamp.secs - CONST_DT
        # past_rod_center_vect = transform_lookup(frame_parent=world_inertial_frame, frame_child=table_center_frame_name, time=rospy.Time(), past_time=rospy.Time(past_step_time))
        # TODO: tune params to scale things, polulate ob
        # scale position to match sim scale
        sim_position = [(position[0] - MIN_X) / (MAX_X - MIN_X) * WINDOW_W,
                        tf_y]
        if past_rod_center_pose is None:
            sim_velocity = np.zeros(3)
        else:
            q1_inv = np.zeros(4)
            q1_inv[0] = past_rod_center_vec.transform.rotation.x
            q1_inv[1] = past_rod_center_vec.transform.rotation.y
            q1_inv[2] = past_rod_center_vec.transform.rotation.z
            q1_inv[3] = -past_rod_center_vec.transform.rotation.w # Negate for inverse
            q_curr = np.zeros(4)
            q_curr[0] = rod_center_vect.transform.rotation.x
            q_curr[1] = rod_center_vect.transform.rotation.y
            q_curr[2] = rod_center_vect.transform.rotation.z
            q_curr[3] = rod_center_vect.transform.rotation.w
            q_rot = tf.transformations.quaternion_multiply(q1_inv, q_curr)
            d_q_rot = euler_from_quaternion(q_rot[0], q_rot[1], q_rot[2], q_rot[3])[2]
            d_q_rot = d_q_rot % (2 * np.pi)
            sim_velocity = [(sim_position[0] - past_rod_center_pose[0]) / CONST_DT, (sim_position[1] - past_rod_center_pose[1]) / CONST_DT, d_q_rot]
        past_rod_center_pose = np.array([sim_position[0], sim_position[1], sim_theta])
        new_obs = np.zeros(18, ) # [x, y, cos, sin, goal_x, goal_y, obs_x, obs_y, table.angle]
        
        new_obs[0:2] = sim_position
        new_obs[2:4] = [np.cos(sim_theta), np.sin(sim_theta)]
        new_obs[4:7] = sim_velocity
        new_obs[7:12] = map_info
        new_obs[12:18] = grid
        print('obs tf', time.time() - start, 'mocap th', str(theta), 'transf mocap2sim th', str(sim_theta))
        if planner_type == "cogail":
            obs_hist = env.obs_hist
            obs_hist[: -env.state_dim] = obs_hist[env.state_dim :]
            obs_hist[-env.state_dim :] = new_obs
            new_obs = np.concatenate((obs_hist, env.map_info, env.grid))

        return new_obs, past_rod_center_vec, past_rod_center_pose
    
    
    # Create variables
    global past_rod_center_vec
    past_rod_center_vec = None
    past_rod_center_pose = None
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
        "planner",
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
    elif human == "planner":
        assert (
            robot == "planner" and exp_run_mode == "coplanning"
        ), "Must be in co-planning mode if human is planner. Change robot mode to 'planner', and exp_run_mode to 'coplanning'."
    else:
        assert playback_trajectory is not None, "Must provide playback trajectory"
        if len(playback_trajectory["actions"].shape) == 3:
            print(playback_trajectory["actions"].shape)
            playback_trajectory["actions"] = playback_trajectory["actions"].squeeze() 
        assert (
            human == "data"
        ), "human arg must be from 'data' if not 'real' or 'policy'"
    # Set n_steps to data limit if using human or robot data as control inputs
    if human == "data" or robot == "data":
        n_steps = len(playback_trajectory["actions"]) - 1
    coplanning = True if (human == "planner" and robot == "planner") else False

    # Check for valid robot arg
    assert robot in [
        "planner",
        "data",
        "real",
    ], "robot arg must be one of 'planner' or 'data' or 'real' (for turing test)"
    if robot == "real":
        print("Robot is real")
        p1_id = 1

    # -------------------------------------------- SETUP PLANNER -------------------------------------------- #
    if robot == "planner":
        def _is_safe(state):
            # Check if state is in collision
            return is_safe(state, collision_checking_env)

        path = None
        if planner_type == "vrnn":
            sys.path.append(join(dirname(__file__), "..", "algo", "planners", "cooperative_planner"))
            from algo.planners.cooperative_planner.models import VRNN
            artifact_path = join("/home/eleyng/table-carrying-ai/trained_models/vrnn/model.ckpt")
            model = VRNN.load_from_checkpoint(artifact_path)
            model.eval()
            model.batch_size = num_candidates
            model.skip = 3
            
            a_horizon_ct = 0
            a_horizon = 1
            
        elif planner_type == "rrt":
            sys.path.insert(0, join(dirname(__file__), "../libs/planner/ompl-1.6.0", "py-bindings"))
            sys.path.insert(0, join(dirname(__file__), "../libs/planner", "RobotMP"))
            import libs.planner.RobotMP.robotmp as rmp

            lower_limits = [0, 0, 0]
            upper_limits = [WINDOW_W, WINDOW_H, 2 * np.pi]
            planner = rmp.OMPLPlanner(
                state_space_bounds=[lower_limits, upper_limits],
                state_validity_checker=_is_safe,
                planner="rrt",
            )
            goal_state = [env.goal[0], env.goal[1], np.pi]
        elif planner_type == "diffusion_policy":
            sys.path.insert(0, join(dirname(__file__), "../libs/policy","diffusion_policy"))
            import copy

            import dill
            import hydra
            from omegaconf import OmegaConf

            sys.path.append('/home/eleyng/diffusion_policy')
            import diffusion_policy.common.pytorch_util as ptu
            import diffusion_policy.policy.diffusion_transformer_lowdim_policy as dp_policy
            from diffusion_policy.env_runner.table_lowdim_runner import \
                TableLowdimRunner
            from diffusion_policy.gym_util.multistep_wrapper import \
                MultiStepWrapper
            from diffusion_policy.gym_util.video_recording_wrapper import (
                VideoRecorder, VideoRecordingWrapper)
            from gym.wrappers import FlattenObservation

            # load checkpoint
            if mcfg.human_act_as_cond:
                ckpt_path = join("/home/eleyng/table-carrying-ai/trained_models/diffusion/model_human_act_as_cond_10Hz.ckpt")
                print('Using human actions!')
            else:
                ckpt_path = join("/home/eleyng/table-carrying-ai/trained_models/diffusion/model_10Hz.ckpt")
                print('Not using human actions!')
            
            payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
            cfg = payload['cfg']
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            # configure env and create policy
            a_horizon = 3
            a_horizon_ct = 0

            policy : dp_policy.DiffusionTransformerLowdimPolicy
            policy = workspace.model
            
            policy.eval().to(device)
            policy.num_inference_steps = 16
            policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1


            steps_per_render = max(10 // FPS, 1)
            def env_fn():
                return MultiStepWrapper(
                    VideoRecordingWrapper(
                        FlattenObservation(
                            env
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec="h264",
                            input_pix_fmt="rgb24",
                            crf=22,
                            thread_type="FRAME",
                            thread_count=1,
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render,
                    ),
                    n_obs_steps=cfg.n_obs_steps,
                    n_action_steps=cfg.n_action_steps,
                    max_episode_steps=2000,
                )
            env = env_fn()
        
        elif  planner_type == "cogail":
            # import models  from cogail
            sys.path.append('/home/eleyng/cogail-table')
            import os

            from a2c_ppo_acktr.model import Policy

            # policy (action output)
            recode_dim = (10 * 7 + 2 + 3*2 + 3) + 2

            actor_critic = Policy(
                env.full_observation.shape,
                env.action_space,
                recode_dim,  # changes input states & actions to embedding of size recode_dim
                base_kwargs={
                    "recurrent": False,
                    "code_size": 2,
                    "base_net_small": True,
                },
            )
            actor_critic.eval().to(device)

            # Load model
            model_path = "/home/eleyng/table-carrying-ai/trained_models/cogail/model_10Hz.pt"
            ckpt = torch.load(model_path)
            actor_critic.load_state_dict(ckpt)

        elif planner_type == "bc_lstm_gmm":

            sys.path.append('/home/eleyng/robomimic')
            import robomimic.utils.file_utils as FileUtils
            import robomimic.utils.obs_utils as ObsUtils
            import robomimic.utils.tensor_utils as TensorUtils
            import robomimic.utils.torch_utils as TorchUtils
            from robomimic.algo import RolloutPolicy
            from robomimic.envs.env_base import EnvBase

            if mcfg.human_act_as_cond:
                model_path = "/home/eleyng/table-carrying-ai/trained_models/bc_lstm_gmm/model_human_act_as_cond_10Hz.pth"
            else:
                model_path = "/home/eleyng/table-carrying-ai/trained_models/bc_lstm_gmm/model_10Hz.pth"
            policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=model_path, device=device, verbose=True)
            config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
            rollout_horizon = config.experiment.rollout.horizon

            # configre rollout policy
            policy.start_episode()

        else:
            raise ValueError("Invalid planner type")


    # ----------------------------------------------------- SIMULTAOR SETUP -----------------------------------------------#

    # reset environment
    obs = env.reset()
    obs_model = np.tile(copy.deepcopy(obs), (SKIP_FRAME, 1))
    past_action = np.zeros(obs_model.shape)[..., :4]

    # Warm up policy
    print("Warming up policy inference")
    # obs = env.get_obs()
    with torch.no_grad():
        policy.reset()
        # obs_dict_np = get_real_obs_dict(
        #     env_obs=obs, shape_meta=cfg.task.shape_meta)
        obs_dict_np = {"obs" : obs.astype(np.float32)}
        if mcfg.human_act_as_cond:
            obs_dict_np["past_action"] = past_action.astype(np.float32)
        obs_dict = ptu.dict_apply(obs_dict_np, 
            lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        result = policy.predict_action(obs_dict)
        action = result['action'][0] #.detach().numpy()
        assert action.shape[-1] == 4
        del result
    print('Ready!')
    policy.reset()

    # Convert obs to torch tensor if not already
    if not isinstance(obs, torch.Tensor):
        obs_t = torch.from_numpy(obs).float()
    info = None
    done = False
    n_iter = 0
    running = True
    clock = pygame.time.Clock()
    success = False
    # Track time used for planning
    delta_plan_sum = 0
    plan_cter = 0

    # ----------------------------------------------- SETUP EXPERIMENT VIS -------------------------------------------- #

    # Initialize list of past states visited in the simulator, for visualization purposes
    if display_past_states:
        past_states = []
        past_states.append(obs.tolist())

    # Initialize list of ground truth waypoints, if displaying ground truth
    if display_gt:
        waypoints_true = playback_trajectory["states"].tolist()


    ### ---------------------------------------------------- GAME LOOP ---------------------------------------------------- ###
    

    action_plan = None
    loop_timer_begin = time.time()
    start = time.time()
    next_game_tick = time.time()

    while running:

        loops = 0
        # print('reset loop')

        if done:
            # time.sleep(1)
            pygame.quit()
            print("Episode finished after {} timesteps".format(n_iter + 1))
            break

        else:

            # while time.time() > next_game_tick and loops < MAX_FRAMESKIP:
            # current time > next game tick and we haven't skipped too many frames yet
            start_plan = time.time()

            if display_gt:
                env.draw_gt(waypoints_true)
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
            elif human == "planner":
                pass
            else:
                # If using human data, then get the actions from the playback trajectory
                assert (
                    human == "data"
                ), "human arg must be from 'data' if not 'real', 'planner', or 'policy'"
                n_iter = min(
                    n_iter, playback_trajectory["actions"].shape[0] - 1
                )  # Needed to account finish the playback
                # else:
                #     assert n_iter < actions.shape[0], "Ran out of human actions from data."
                u_h = playback_trajectory["actions"][n_iter, 2:]
                u_h = torch.from_numpy(u_h).unsqueeze(0)

            # interpolation = 0 
            # print("interpolation: ", interpolation)

            if (a_horizon_ct < a_horizon) and n_iter != 0 and planner_type == "diffusion_policy":
                # Fetch the next action from the previously planned action plan
                u_r = torch.from_numpy(
                    action_plan[a_horizon_ct, :2]
                ).unsqueeze(0)
                if coplanning:
                    u_h = torch.from_numpy(
                        action_plan[a_horizon_ct, 2:]
                    ).unsqueeze(0)
                u_all = torch.cat((u_r, u_h), dim=-1)
                # Update past action for HUMAN input
                past_action[:-1, :] = past_action[1:, :]
                past_action[-1, :2] = action_plan[a_horizon_ct, :2].squeeze()
                past_action[-1, 2:] = u_h.flatten() if not coplanning else action_plan[a_horizon_ct, 2:]
                a_horizon_ct += 1 if zoh_ct % SKIP_FRAME == 0 else 0
                zoh_ct += 1
                # print("no planning, ")

            else:
                #### --------------------------------------------- GET ROBOT INPUTS -------------------------------------------- ####
                if robot == "real" and exp_run_mode == "hil": # Turing Test mode
                    u_r = np.array(
                        [
                            joysticks[p1_id].get_axis(0),
                            joysticks[p1_id].get_axis(1),
                        ]
                    )
                    u_r = torch.from_numpy(np.clip(u_r, -1.0, 1.0)).unsqueeze(0)
                    u_all = torch.cat((u_r, u_h), dim=-1)
                    
                else:
                    # -------------------------------------------- PLANNING PERIOD -------------------------------------------- #

                    # If we are in the planning period, then we need to continue updating the state history queue, get the next observation
                    # from the simulator by feeding the human input and robot input from PID, which controls to waypoints planned by the model.

                    a_horizon_ct = 0
                    zoh_ct = 1
                    obs_dict_np = {"obs" : obs_model[::SKIP_FRAME, ...].astype(np.float32)}
                    if mcfg.human_act_as_cond:
                        obs_dict_np["past_action"] = past_action[::SKIP_FRAME, ...].astype(np.float32)
                        print("past_action: ", past_action[::SKIP_FRAME, ...].astype(np.float32))
                    obs_dict = ptu.dict_apply(obs_dict_np,
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                    # lambda x: x.to(device=device))
                    
                    # run policy
                    with torch.no_grad():
                        result = policy.predict_action(obs_dict)
                        plan_cter += 1

                        # np_action_dict = ptu.dict_apply(action_dict,
                        #     lambda x: x.detach().to('cpu').numpy())
                        action_plan = result['action'][0].detach().to('cpu').numpy()

                        # action_plan = np_action_dict['action']
                        u_r = torch.from_numpy(
                                action_plan[a_horizon_ct, :2]
                            ).unsqueeze(0)
                        if coplanning:
                            u_h = torch.from_numpy(
                                action_plan[a_horizon_ct, 2:]
                            ).unsqueeze(0)
                        # u_r[:, :] = 0
                        u_all = torch.cat((u_r, u_h), dim=-1)

                    # update past action
                    past_action[:-1, :] = past_action[1:, :]
                    past_action[-1, :2] = action_plan[a_horizon_ct, :2].squeeze()
                    past_action[-1, 2:] = u_h.flatten() if not coplanning else action_plan[a_horizon_ct, 2:]

                    a_horizon_ct += 1 if zoh_ct % SKIP_FRAME == 0 else 0
                    zoh_ct += 1


                    delta_plan = time.time() - start_plan
                    print("Planning time:", delta_plan)
                    delta_plan_sum += delta_plan

            # -------------------------------------------- UPDATE ENVIRONMENT -------------------------------------------- #
            # -------------------------------------------- UPDATE ENVIRONMENT -------------------------------------------- #

            # Sum forces, then multiply by scaling factor to get desired table velocity
            duration = CONST_DT
            p_des_vel = share_control(u_r.detach().numpy(), u_h.detach().numpy())
            # print("u_h", u_h, "u_r", u_r)
            p_des_vel_x = p_des_vel[0]
            p_des_vel_y = p_des_vel[1]
            p_des_ang_vel_z = p_des_vel[2]

            # Publish desired velocity
            move_timer = time.time()
            # p_des_vel_x = n_iter / 10000
            # p_des_vel_y = n_iter /10000
            # p_des_ang_vel_z = 0
            move(v_x=p_des_vel_x, v_y=p_des_vel_y, yaw=p_des_ang_vel_z, duration=CONST_DT, curr_pub=pub_p_des_vel)
            # print("move pub timer ", time.time() - move_timer)

            end_pub = time.time()
            # print("p_des_vel", p_des_vel)
            # print("sim pose: ", obs, env.table.angle)
            print("Uall", u_all)
            if planner_type != "cogail":
                obs, reward, done, info = env.step(u_all.unsqueeze(0).detach().to("cpu").numpy())
            else:
                obs, reward, succ, done, info, random_seed = env.step(u_all.detach().numpy())

            obs, past_rod_center_vec, past_rod_center_pose = mocap_pose_to_obs(obs, env.map_info, env.grid, past_rod_center_vec, past_rod_center_pose)
            env.table.x = past_rod_center_pose[0]
            env.table.y = past_rod_center_pose[1]
            env.table.angle = past_rod_center_pose[2]
            env.redraw()

            print("sim vel:", env.table.x_speed, env.table.y_speed, env.table.angle_speed)
            # print("mocap pose: ", obs)
            loop_time = time.time() - loop_timer_begin
            # print("loop time", loop_time)
            loop_timer_begin = time.time()

            # update obs_model
            obs_model[:-1, ...] = obs_model[1:, ...]
            obs_model[-1, ...] = obs[-1, ...]

            n_iter += 1
            if display_past_states:
                past_states.append(obs.tolist())
                env.draw_past_states(past_states)

            obs_t = torch.from_numpy(obs).float()


            trajectory["states"].append(obs_t[-1, ...]) #should be torch
            if robot == "planner":
                if planner_type == "vrnn":
                    trajectory["plan"].append(torch.tensor(path))
                elif planner_type == "rrt":
                    trajectory["plan"].append(path.tolist())
                elif planner_type == "diffusion_policy":
                    trajectory["plan"].append(action_plan[a_horizon_ct].tolist())
            trajectory["actions"].append(u_all)
            trajectory["rewards"].append(torch.tensor(reward))
            trajectory["fluency"].append(env.fluency)
            
            if done:
                if info["success"][-1]:
                    success = True
                else:
                    success = False
                env.render(mode="human")
                running = False
                break

            next_game_tick += CONST_DT
            loops += 1
            # print("loops: ", loops, "next_game_tick: ", next_game_tick)

            # print("Skip env update")
            # Update display
            if not done:
                env.redraw()
                clock.tick(FPS)

                # process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if (
                            event.key in relevant_keys
                            and human == "real"
                            and env.control_type == "keyboard"
                        ):
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
    print("Average planning time per planning loop: ", delta_plan_sum / plan_cter)
    print("Duration of run: ", duration)
    pygame.quit()
    print("Fluency: ", np.max(env.fluency["inter_f"]), np.min(env.fluency["inter_f"]))

    if not (human == "data" and robot == "data"):
        # Save trajectory
        trajectory["states"] = torch.stack(trajectory["states"], dim=0).numpy()
        if robot == "planner":
            if planner_type == "vrnn":
                trajectory["plan"] = torch.stack(trajectory["plan"], dim=0).numpy()
        trajectory["actions"] = torch.stack(trajectory["actions"], dim=0).numpy()
        trajectory["rewards"] = torch.stack(trajectory["rewards"], dim=0).numpy()
        assert info is not None, "Error: info is None"
        trajectory["fluency"] = info["fluency"]
        trajectory["success"] = info["success"]
        trajectory["done"] = torch.FloatTensor(
            np.array(
                [
                    [float(done)],
                ]
            )
        )
        trajectory["n_iter"] = n_iter
        trajectory["duration"] = duration

    return trajectory, success, n_iter, duration, delta_plan_sum / plan_cter