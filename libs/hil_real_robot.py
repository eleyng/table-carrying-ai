import copy
import math
import sys
import time
from os.path import dirname, join

import numpy as np
import pygame
import rospy
from geometry_msgs.msg import PoseStamped, Twist
import tf2_ros
import torch
import wandb

from cooperative_transport.gym_table.envs.utils import (CONST_DT, FPS,
                                                        MAX_FRAMESKIP,
                                                        WINDOW_H, WINDOW_W, I,
                                                        L, b, d, debug_print,
                                                        get_keys_to_action,
                                                        init_joystick, m,
                                                        set_action_keyboard)
from libs.planner.planner_utils import (is_safe, pid_single_step, tf2model,
                                        tf2sim, update_queue)

from libs.real_robot_utils import (reset_odom, euler_from_quaternion,
                                   MIN_X, MIN_Y, MAX_X, MAX_Y,
                                   share_control, move, compute_reward)



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
# PUBLISHERS -- table velocity
pub_p_des_vel = rospy.Publisher("/table_velocity", Twist, queue_size=5)

# Create tf buffer and listener to nice_origin (RH rule)
global tfBuffer, listener
tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer)

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
    def mocap_pose_to_obs(obs, map_info, grid, past_rod_center_vec=None):
        """ Convert mocap pose to observation in sim
        obs: np.array (7 + 5 + 6 = 18, )
        table x, y, cth, sth, xspeed, yspeed, ang_speed, map_info (5), grid (6)
        Return: np.array (7 + 5 + 6 = 18, )
        """
        start = time.time()
        
        rod_center_vect = transform_lookup(frame_parent=world_inertial_frame, frame_child=table_center_frame_name, time=rospy.Time())
        position = [rod_center_vect.transform.translation.x, rod_center_vect.transform.translation.y]
        theta = euler_from_quaternion(rod_center_vect.transform.rotation.x, 
                                    rod_center_vect.transform.rotation.y, 
                                    rod_center_vect.transform.rotation.z, 
                                    rod_center_vect.transform.rotation.w)[2]
        ang_tf = np.pi / 2
        new_theta = (theta + ang_tf) % (2 * np.pi)
        # calc velocity ## TODO: add velocity pose message
        # past_step_time = rod_center_vect.header.stamp.secs - CONST_DT
        # past_rod_center_vect = transform_lookup(frame_parent=world_inertial_frame, frame_child=table_center_frame_name, time=rospy.Time(), past_time=rospy.Time(past_step_time))
        if past_rod_center_vec is None:
            velocity = np.zeros(3)
        else:
            # prrv = past_rod_center_vec.transform
            # delta_rot = euler_from_quaternion(prrv.rotation.x, prrv.rotation.y, prrv.rotation.z, prrv.rotation.w)[2]
            velocity = [(position[0] - past_rod_center_vec[0]) / CONST_DT, (position[1] - past_rod_center_vec[1]) / CONST_DT, (new_theta - (past_rod_center_vec[2] % (2 * np.pi)))/ CONST_DT]
        past_rod_center_vec = np.array([position[0], position[1], new_theta])
        new_obs = np.zeros(18, ) # [x, y, cos, sin, goal_x, goal_y, obs_x, obs_y, table.angle]
        # TODO: tune params to scale things, polulate ob
        # scale position to match sim scale
        sim_position = [(position[0] - MIN_X) / (MAX_X - MIN_X) * WINDOW_W,
                        (position[1] - MIN_Y) / (MAX_Y - MIN_Y) * WINDOW_H]

        new_obs[0:2] = sim_position
        new_obs[2:4] = [np.cos(new_theta), np.sin(new_theta)]
        new_obs[4:7] = velocity
        new_obs[7:12] = map_info
        new_obs[12:18] = grid
        print('obs tf', time.time() - start)

        return new_obs, past_rod_center_vec
    
    
    # Create variables
    global past_rod_center_vec
    past_rod_center_vec = None

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
            artifact_path = join("/home/collab1/table-carrying-ai/trained_models/vrnn/model.ckpt")
            model = VRNN.load_from_checkpoint(artifact_path, map_location=torch.device('cpu'))
            model.eval()
            model.batch_size = num_candidates
            model.skip = skip
            
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
            import copy

            import dill
            import hydra
            from omegaconf import OmegaConf

            sys.path.append('/home/collab1/diffusion_policy')
            import diffusion_policy.common.pytorch_util as ptu
            import diffusion_policy.policy.diffusion_transformer_lowdim_policy as dp_policy
            from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
            from diffusion_policy.env_runner.base_lowdim_runner import \
                BaseLowdimRunner
            from diffusion_policy.gym_util.multistep_wrapper import \
                MultiStepWrapper

            # load checkpoint
            if mcfg.human_act_as_cond:
                ckpt_path = join("/home/collab1/table-carrying-ai/trained_models/diffusion/model_human_act_as_cond_10Hz.ckpt")
            else:
                ckpt_path = join("/home/collab1/table-carrying-ai/trained_models/diffusion/model_10Hz.ckpt")
            payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)

            # load hydra config
            cfg = payload['cfg']
            cls = hydra.utils.get_class(cfg._target_)
            # workspace = cls(cfg)
            # workspace: BaseWorkspace
            # workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            # configure env and create policy
            num_inference_steps_ddpm = 3 #4
            a_horizon = 1
            a_horizon_ct = 0
            # update config
            OmegaConf.update(cfg, "policy.num_inference_steps", num_inference_steps_ddpm, merge=False)
            OmegaConf.update(cfg, "policy.n_action_steps", a_horizon, merge=False)
            # OmegaConf.update(cfg, "horizon", 2, merge=False)
            # OmegaConf.update(cfg, "n_obs_steps", 1, merge=False)


            steps_per_render = max(10 // FPS, 1)
            def env_fn():
                return MultiStepWrapper(
                    env,
                    n_obs_steps=cfg.n_obs_steps,
                    n_action_steps=cfg.n_action_steps,
                    max_episode_steps=3000,
                )

            env = env_fn()

            dataset_path = "/home/collab1/diffusion_policy/data/table/table.zarr"
            OmegaConf.update(cfg, "task.dataset.zarr_path", dataset_path, merge=False)
            dataset: BaseLowdimDataset
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            normalizer = dataset.get_normalizer()
            policy : dp_policy.DiffusionTransformerLowdimPolicy
            policy = hydra.utils.instantiate(cfg.policy)
            # ema_model = copy.deepcopy(policy)
            # ema_model.set_normalizer(normalizer)
            # ema = hydra.utils.instantiate(cfg.ema, model=ema_model)
            # policy = ema_model
            policy.set_normalizer(normalizer)
            policy.to(device).eval()

            OmegaConf.update(cfg, "task.env_runner.n_train", 0, merge=False)
            OmegaConf.update(cfg, "task.env_runner.n_train_vis", 0, merge=False)
            OmegaConf.update(cfg, "task.env_runner.n_test", 1, merge=False)
            OmegaConf.update(cfg, "task.env_runner.n_test_vis", 1, merge=False)
            # OmegaConf.update(cfg, "task.env_runner.n_obs_steps", 2, merge=False)
            # env_runner: BaseLowdimRunner
            # env_runner = hydra.utils.instantiate(
            #     cfg.task.env_runner, output_dir="/home/collab1/diffusion_policy/data/table/"
            # )
            # runner_log = env_runner.run(policy)
            # step_log = dict()
            # step_log.update(runner_log)
            # wandb_run = wandb.init(
            #     dir="/home/collab1/diffusion_policy/data/table/test_runner",
            #     config=OmegaConf.to_container(cfg, resolve=True),
            #     **cfg.logging,
            # )
            # wandb.config.update(
            #     {
            #         "output_dir": "/home/collab1/diffusion_policy/data/table/test_runner",
            #     }
            # )
            # wandb_run.log(step_log, step=0)
            

            # configure dataset statistics
            
            
            # policy.set_normalizer(normalizer)
        
        elif  planner_type == "cogail":
            # import models  from cogail
            sys.path.append('/home/collab1/cogail-table')
            import os

            from a2c_ppo_acktr.model import Policy

            # policy (action output)
            recode_dim = (30 * 7 + 2 + 3*2 + 3) + 2

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
            model_path = "/home/collab1/table-carrying-ai/trained_models/cogail/model.pt"
            ckpt = torch.load(model_path)
            actor_critic.load_state_dict(ckpt)

        elif planner_type == "bc_lstm_gmm":

            sys.path.append('/home/collab1/robomimic')
            import robomimic.utils.file_utils as FileUtils
            import robomimic.utils.obs_utils as ObsUtils
            import robomimic.utils.tensor_utils as TensorUtils
            import robomimic.utils.torch_utils as TorchUtils
            from robomimic.algo import RolloutPolicy
            from robomimic.envs.env_base import EnvBase

            if mcfg.human_act_as_cond:
                model_path = "/home/collab1/table-carrying-ai/trained_models/bc_lstm_gmm/model_human_act_as_cond.pth"
            else:
                model_path = "/home/collab1/table-carrying-ai/trained_models/bc_lstm_gmm/model.pth"
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
    if isinstance(obs, tuple): #change table_env reset(): must output both obs and random_seed
        random_seed = obs[1]
        obs = obs[0]
    if planner_type in ["diffusion_policy", "bc_lstm_gmm"]:
        obs_q = np.tile(obs, (30, 1))
        past_action = np.zeros((a_horizon, 4)) if planner_type == "diffusion_policy" else np.zeros_like(obs.squeeze())[..., :4]
    elif planner_type in ["vrnn"]:
        obs = torch.from_numpy(obs).float()
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
    else:
        # Convert obs to torch tensor if not already
        if not isinstance(obs, torch.Tensor):
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

    # ----------------------------------------------- SETUP EXPERIMENT VIS -------------------------------------------- #

    # Initialize list of past states visited in the simulator, for visualization purposes
    if display_past_states:
        past_states = []
        past_states.append(obs.tolist())

    # Initialize list of ground truth waypoints, if displaying ground truth
    if display_gt:
        waypoints_true = playback_trajectory["states"].tolist()

    # ------------------------------------------ SETUP DATA STRUCTS FOR MODEL USE ------------------------------------------ #

    


    ### ---------------------------------------------------- GAME LOOP ---------------------------------------------------- ###
    plan_cter = 0
    loop_timer_begin = time.time()
    rate = rospy.Rate(FPS)
    start = time.time()
    while running and not rospy.is_shutdown():
        rospy.loginfo("OBS RATE")
        

        loops = 0

        if done:
            # time.sleep(1)
            pygame.quit()
            print("Episode finished after {} timesteps".format(n_iter + 1))
            break

        else:

            while time.time() > next_game_tick and loops < MAX_FRAMESKIP and not done:

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
                    if planner_type == "vrnn":
                        if n_iter <= mcfg.H + mcfg.skip:
                            u_h = playback_trajectory["actions"][n_iter, 2:]
                            u_h = torch.from_numpy(u_h).unsqueeze(0)
                    else:
                        pass
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
                    u_h_npy = u_h
                    u_h = torch.from_numpy(u_h).unsqueeze(0)

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
                    start_plan = time.time()


                    # -------------------------------------------- PLANNING PERIOD -------------------------------------------- #

                    # If we are in the planning period, then we need to continue updating the state history queue, get the next observation
                    # from the simulator by feeding the human input and robot input from PID, which controls to waypoints planned by the model.
                    if planner_type == "vrnn":
                        # -------------------------------------------- IF USING VRNN: GET WAYPOINTS -------------------------------------------- #
                

                        if (n_iter <= mcfg.H + mcfg.skip) or (
                            human == "data" and robot == "data"
                        ):
                            # # Feed first H steps of state history into simulator
                            u_r = torch.from_numpy(
                                playback_trajectory["actions"][n_iter, :2]
                            ).unsqueeze(0)
                            # if coplanning:
                            #     u_all = torch.from_numpy(playback_trajectory["actions"][n_iter, :]).unsqueeze(0)
                            # else:
                            #     u_all = torch.cat(
                            #         (u_r, u_h), dim=-1
                            #     )  # player 1 is blue, and in this sim human is player 2 (orange)
                            # # Update action history queue
                            # print('n_iter less than {}, horizon {}.'.format(mcfg.H + mcfg.skip, mcfg.SEQ_LEN))
                            if coplanning:
                                u_all = torch.zeros((1, 4)) 
                            else:
                                # u_r = torch.zeros((1, 2))
                                u_all = torch.cat(
                                    (u_r, u_h), dim=-1
                                )  # player 1 is blue, and in this sim human is player 2 (orange)
                            u_queue = update_queue(u_queue, u_all)

                        else:

                            # if (n_iter % mcfg.skip != 0) and not (
                            #     human == "data" and robot == "data"
                            # ):
                            if a_horizon_ct < a_horizon and path is not None:
                                n_iter += 1
                                a_horizon_ct += 1
                                
                            else:
                                with torch.no_grad():

                                    s_tf = tf2model(s_queue, env.obstacles, zero_padding=False).repeat(mcfg.BSIZE, 1, 1)
                                    u = u_queue.repeat(mcfg.BSIZE, 1, 1).float()
                                    sample = model.sample(s_tf, u, seq_len=mcfg.SEQ_LEN)

                                waypoints = tf2sim(
                                    sample[:, :, :4],
                                    s_queue,
                                    (mcfg.H // mcfg.skip),
                                )

                                # Evaluate the rewards the batch of sampled trajectories using custom reward function
                                eval = np.sum(
                                    np.array(
                                        [
                                            compute_reward(
                                                waypoints[i, :, :4],
                                                env.goal,
                                                env.obstacles,
                                                interaction_forces=include_interaction_forces_in_rewards,
                                                env=env,
                                                u_h=None,
                                                collision_checking_env=collision_checking_env,
                                            )
                                            for i in range(waypoints.shape[0])
                                        ]
                                    ),
                                    -1,
                                )

                                # Select the best trajectory
                                best_traj = np.argmax(eval)
                                path = waypoints[best_traj, :, :]
                                a_horizon_ct = 0

                            idx = a_horizon_ct + mcfg.lookahead // mcfg.skip
                            if coplanning:
                                pid_actions = pid_single_step(
                                    env,
                                    path[idx, :4],
                                    kp=0.15, #0.00001,
                                    ki=0.0,
                                    kd=0.0,
                                    max_iter=40,
                                    dt=CONST_DT,
                                    eps=1e-2,
                                    u_h=None,
                                    joint=coplanning,
                                )
                            else:
                                pid_actions = pid_single_step(
                                    env,
                                    path[idx, :4],
                                    kp=0.15, #0.00001,
                                    ki=0.0,
                                    kd=0.0,
                                    max_iter=40,
                                    dt=CONST_DT,
                                    eps=1e-2,
                                    u_h=u_h.squeeze().numpy(),
                                    joint=coplanning,
                                )
                            pid_actions /= np.linalg.norm(pid_actions)
                            # pid_actions[0] *= -1.0 ## TODO: CHECK THIS
                            plan_cter += 1

                            if not coplanning:
                                u_r = torch.from_numpy(
                                    np.clip(pid_actions, -1.0, 1.0)
                                ).unsqueeze(0)

                                u_r = torch.from_numpy(
                                    playback_trajectory["actions"][n_iter, :2]
                                ).unsqueeze(0)
                                print("N_ITER", n_iter)
                                u_all = torch.cat((u_r, u_h), dim=-1)

                            else:
                                u_all = torch.from_numpy(
                                    np.clip(pid_actions, -1.0, 1.0)
                                ).unsqueeze(0)
                            u_queue = update_queue(u_queue, u_all)
                            # print('u_all: ', u_all)
                    elif planner_type in ["diffusion_policy"] and n_iter != 0 and (a_horizon_ct < a_horizon):

                        #elif planner_type in ["diffusion_policy"] and n_iter != 0 and (a_horizon_ct < a_horizon):
                        # Fetch the next action from the previously planned action plan
                        u_r = torch.from_numpy(
                            action_plan[a_horizon_ct, :2]
                        ).unsqueeze(0)
                        print(coplanning)
                        if coplanning:
                            u_h = torch.from_numpy(
                                action_plan[a_horizon_ct, 2:]
                            ).unsqueeze(0)

                        u_all = torch.cat((u_r, u_h), dim=-1) #.unsqueeze(0)

                        # Update past action for HUMAN input
                        past_action[-(a_horizon - a_horizon_ct), 2:] = u_h.flatten()
                        a_horizon_ct += 1
                        # print("no planning, ")

                    else:

                        if planner_type == "diffusion_policy":

                            a_horizon_ct = 0
                            if mcfg.human_act_as_cond:
                                obs_dict = {
                                    'obs': obs.astype(np.float32),
                                    'past_action': past_action.astype(np.float32)
                                }
                            else:
                                obs_dict = {
                                    'obs': obs.astype(np.float32),
                                }
                            obs_dict = ptu.dict_apply(obs_dict,
                                            lambda x: torch.from_numpy(x).to(
                                                device=device))
                        
                            # run policy
                            with torch.no_grad():
                                action_dict = policy.predict_action(obs_dict)
                                # print("pred r : ", action_dict['action'][0, :, :2])
                                plan_cter += 1

                            np_action_dict = ptu.dict_apply(action_dict,
                                lambda x: x.detach().to('cpu').numpy())

                            action_plan = np_action_dict['action'].squeeze(0).astype(np.float32)
                            u_r = torch.from_numpy(
                                    action_plan[a_horizon_ct, :2]
                                ).unsqueeze(0)
                            
                            if coplanning:
                                u_h = torch.from_numpy(
                                    action_plan[a_horizon_ct, 2:]
                                ).unsqueeze(0)
                            u_all = torch.cat((u_r, u_h), dim=-1) #.unsqueeze(0) if 
                            
                            # # Update past action for ROBOT
                            past_action[:-a_horizon, :] = past_action[a_horizon:, :]
                            past_action[-a_horizon:, :2] = action_plan[:a_horizon, :2]
                            # Update past action for HUMAN input
                            past_action[-(a_horizon - a_horizon_ct), 2:] = u_h if not coplanning else action_plan[a_horizon_ct, 2:]
                            # past_action = u_h # action_plan

                            a_horizon_ct += 1

                        elif planner_type == "bc_lstm_gmm":
                            obs_dict = {}
                            if mcfg.human_act_as_cond:
                                if coplanning:
                                    u_h = past_action[2:]
                                obs = np.concatenate((obs, u_h), axis=-1)
                                obs_dict["all"] = obs
                            else:
                                obs_dict["all"] = obs
                            
                            action = policy(ob=obs_dict)
                            plan_cter += 1
                            u_r = torch.from_numpy(action[:2]).unsqueeze(0)
                            if coplanning:
                                u_h = torch.from_numpy(action[2:]).unsqueeze(0)
                            u_all = torch.cat((u_r, u_h), dim=-1)

                            # Update past action for ROBOT
                            past_action[:2] = action[:2]
                            # Update past action for HUMAN input
                            past_action[2:] = u_h if not coplanning else action[2:]

                        elif planner_type == "cogail":

                            obs = np.divide((obs - env.obs_space_low), env.obs_space_range) 
                            obs = obs.unsqueeze(0) ## TODO: verify that batch is 1 for each eval run
                            eval_recurrent_hidden_states = torch.zeros(
                                1, 1, device=device)
                            eval_masks = torch.zeros(1, 1, device=device)
                            with torch.no_grad():
                                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                                    obs,
                                    random_seed,
                                    eval_recurrent_hidden_states,
                                    eval_masks,
                                    deterministic=True)
                                plan_cter += 1
                                u_r = action[:, :2]

                            if coplanning:
                                u_h = action[:, 2:]
                            u_all = torch.cat((u_r, u_h), dim=-1) # (1,4)

                delta_plan = time.time() - start_plan
                print("Planning time: ", delta_plan)
                delta_plan_sum += delta_plan
                if display_pred and planner_type in ["vrnn"] and path is not None:
                    env.update_prediction(path.tolist())

                # -------------------------------------------- UPDATE ENVIRONMENT -------------------------------------------- #

                # Sum forces, then multiply by scaling factor to get desired table velocity
                duration = CONST_DT
                p_des_vel = share_control(u_h.detach().numpy(), u_r.detach().numpy())
                print("u_h", u_h, "u_r", u_r)
                p_des_vel_x = p_des_vel[0]
                p_des_vel_y = p_des_vel[1]
                p_des_ang_vel_z = p_des_vel[2]

                # Publish desired velocity
                move(v_x=p_des_vel_x, v_y=p_des_vel_y, yaw=p_des_ang_vel_z, duration=1.0, curr_pub=pub_p_des_vel)

                end_pub = time.time()
                print("des vel", p_des_vel)
                print("sim pose: ", obs)
                obs, reward, done, info = env.step(u_all.detach().numpy())
                
                obs, past_rod_center_vec = mocap_pose_to_obs(obs, env.map_info, env.grid, past_rod_center_vec)
                env.table.x = obs[0]
                env.table.y = obs[1]
                env.redraw()
                print("mocap pose: ", obs)
                loop_time = time.time() - loop_timer_begin
                print("loop time", loop_time)
                loop_timer_begin = time.time()

                if planner_type in ["diffusion_policy"]:
                    # obs_q[:-1,:] = obs_q[1:,:]
                    # obs_q[-1,:] = obs
                    pass
                # print("action: ", u_all.detach().numpy())
                if planner_type in ["vrnn"]:
                    obs = torch.from_numpy(obs).float()
                    s_queue = update_queue(s_queue, obs.unsqueeze(0))

                n_iter += 1

                if display_past_states:
                    past_states.append(obs.tolist())
                    env.draw_past_states(past_states)

                # Convert obs to torch tensor if not already
                if not isinstance(obs, torch.Tensor) and planner_type not in ["diffusion_policy"]:
                    obs = torch.from_numpy(obs).float()


                if planner_type in ["bc_lstm_gmm", "vrnn"]:
                    trajectory["states"].append(obs)
                elif planner_type in ["cogail"]:
                    trajectory["states"].append(torch.from_numpy(env.current_state))
                else:
                    trajectory["states"].append(obs[-1, ...])

                if robot == "planner":
                    if planner_type == "vrnn":
                        if path is not None:
                            trajectory["plan"].append(torch.tensor(path))
                    elif planner_type == "rrt":
                        trajectory["plan"].append(path.tolist())
                    elif planner_type == "diffusion_policy":
                        trajectory["plan"].append(action_plan[0].tolist())
                trajectory["actions"].append(u_all)
                trajectory["rewards"].append(torch.tensor(reward))

                if done:
                    if planner_type == "diffusion_policy":
                        if info["success"][-1]:
                            success = True
                            # _ = env.reset()
                        else:
                            success = False
                    else:
                        if info["success"]:
                            success = True
                        else:
                            success = False
                    # env.render(mode="human")
                    running = False
                    break

                next_game_tick += CONST_DT
                loops += 1

            if loops == 0:
                continue
            else:
                pass
                # delta_plan_sum = delta_plan_sum / (loops)


            # Update display
            if not done and planner_type not in ["diffusion_policy"]:
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

        rate.sleep()

    stop = time.time()
    duration = stop - start
    
    print("Average planning time per planning loop: ", delta_plan_sum / plan_cter)
    print("Duration of run: ", duration)
    print("Fluency: ", np.max(env.fluency["inter_f"]), np.min(env.fluency["inter_f"]))

    if planner_type == "diffusion_policy":
        env.close()
        del env

    pygame.quit()

    if not (human == "data" and robot == "data"):
        # Save trajectory
        if planner_type in ["bc_lstm_gmm", "cogail"]:
            trajectory["states"] = torch.stack(trajectory["states"], dim=0).numpy()
        elif not isinstance(trajectory["states"], torch.Tensor):
            if planner_type == "diffusion_policy":
                trajectory["states"] = np.array(trajectory["states"])
            # elif planner_type == "cogail":
            #     trajectory["states"] = np.array(trajectory["states"])[:, -1, :]
            elif planner_type == "vrnn":
                trajectory["states"] = torch.stack(trajectory["states"], dim=0).numpy()
        if robot == "planner":
            if planner_type == "vrnn":
                trajectory["plan"] = torch.stack(trajectory["plan"], dim=0).numpy()
        trajectory["actions"] = torch.stack(trajectory["actions"], dim=0).numpy().squeeze()
        trajectory["rewards"] = torch.stack(trajectory["rewards"], dim=0).numpy().squeeze()
        assert info is not None, "Error: info is None"
        if planner_type == "diffusion_policy":
            trajectory["fluency"] = info["fluency"][0]
            trajectory["success"] = info["success"][-1]
            
        else:
            trajectory["fluency"] = info["fluency"]
            trajectory["success"] = info["success"]

        trajectory["done"] = torch.FloatTensor(
            np.array(
                [
                    [float(done)],
                ]
            )
        )
        trajectory["n_iter"] = trajectory["states"].shape[0]
        n_iter = trajectory["n_iter"]
        trajectory["duration"] = duration

    return trajectory, success, n_iter, duration, delta_plan_sum / plan_cter
