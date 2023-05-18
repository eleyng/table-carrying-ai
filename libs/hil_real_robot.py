import copy
import math
import sys
import time
from os.path import dirname, join

import numpy as np
import pygame
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Empty, Float64MultiArray
from sensor_msgs.msg import Joy
import tf
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
                                   send_plan, compute_reward,)

############################################################################################################
# This node does not need to run on real self.robot. It is used to run the planner, so use the best compute you have.
# This node is responsible for:
# 1. Subscribing to table pose
# 2. Subscribing to human action
# 3. Computing and publishing action plan
############################################################################################################



SKIP_FRAME = 3

class Real_HIL_Runner(object):
    """
    Play a with a trained agent on the gym environment.

        Robot is Player 1 (blue), and human is player 2 (orange).

        Two options for human player:

        1) Human (you, the human) plays with the trained planner (self.robot). If keyboard mode, use "WASD" to move (NOT arrow keys).
        2) Human data is played back with the trained planner (self.robot)
        3) Human trained BC policy is played with the trained planner (self.robot) TODO: add this feature

        Each run begins with a reset of the environment, provided a configuration from a previous
        rollout / demo / ground truth (GT) trajectory. The first H steps from the GT trajectory
        are played out in the enviroment, and can be used for comparison. It is also the trajectory
        that the human plays if running this function in option 2).


        Args:
            env: gym environment
            exp_run_mode: "hil" if human in the loop, "replay_traj" if replaying data
            human: "data" if option 2), "real" if option 1), "policy" if option 3)
            model: trained planner model
            self.mcfg: model config
            SELF.SEQ_LEN: length of sequence for the model to use for planning. Prediction period is SELF.SEQ_LEN - H
            H: observation period. THe model uses the past H observations to predict the next SELF.SEQ_LEN - H actions
            self.playback_trajectory: if not None, then this is the trajectory to play back for option 2)
            self.n_steps: max number of steps to run the episode for
            fps: frames per second
            self.display_pred: if True, then display the predicted trajectory on the pygame window
            self.display_gt: if True, then display the ground truth trajectory on the pygame window.
    """

    def __init__(
                self, 
                env=None,
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
                include_interaction_forces_in_rewards=False
        ):
        rate = rospy.Rate(FPS)
        reset_odom()

        #For Real world:
        self.world_inertial_frame = rospy.get_param('world_frame_name', "nice_origin") #this is origin in sim
        self.table_center_frame_name = rospy.get_param('table_center_frame_name', "table_carried")

        # Create Publisher/Subscribers
        # SUBSCRIBERS -- table pose
        self.sub_human_act = rospy.Subscriber("/joy", Joy, self.human_action_cb, queue_size=1)
        # PUBLISHERS -- table velocity
        self.pub_action_plan = rospy.Publisher("/action_plan", Float64MultiArray, queue_size=1)

        # Create tf buffer and listener to nice_origin (RH rule)
        global tfBuffer, listener
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)

        self.planner_type = planner_type
        self.env = env
        self.exp_run_mode = exp_run_mode
        self.human = human
        self.robot = robot
        self.artifact_path = artifact_path
        self.mcfg = mcfg
        self.SEQ_LEN = SEQ_LEN
        self.H = H
        self.skip = skip
        self.num_candidates = num_candidates
        self.playback_trajectory = playback_trajectory
        self.n_steps = n_steps
        self.fps = fps
        self.collision_checking_env = collision_checking_env
        self.display_pred = display_pred
        self.display_gt = display_gt
        self.display_past_states = display_past_states
        self.device = device
        self.include_interaction_forces_in_rewards = include_interaction_forces_in_rewards
        self.u_h_cb = np.zeros((2,))
        self.mocap_table_pose = None



    def mocap_cb(self, msg):
        global mocap_table_pose
        global prev_mocap_table_pose
        global world_inertial_frame
        global table_center_frame_name
        self.mocap_table_pose = msg.pose

    def human_action_cb(self, msg):
        self.u_h_cb = np.array([-msg.axes[0], msg.axes[1]], dtype=np.float32)
        if np.linalg.norm(self.u_h_cb) == 0.:
            pass
        else:
            self.u_h_cb /= np.linalg.norm(self.u_h_cb)

    def transform_lookup(self, frame_parent=None, frame_child=None, time=rospy.Time(), past_time=None):
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
    def mocap_pose_to_obs(self, map_info, grid, past_rod_center_vec=None, past_rod_center_pose=None):
        """ Convert mocap pose to observation in sim
        obs: np.array (7 + 5 + 6 = 18, )
        table x, y, cth, sth, xspeed, yspeed, ang_speed, map_info (5), grid (6)
        Return: np.array (7 + 5 + 6 = 18, )
        """
        start = time.time()
        
        rod_center_vect = self.transform_lookup(frame_parent=self.world_inertial_frame, frame_child=self.table_center_frame_name, time=rospy.Time())
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
        if self.planner_type == "cogail":
            obs_hist = self.env.obs_hist
            obs_hist[: -self.env.state_dim] = obs_hist[self.env.state_dim :]
            obs_hist[-self.env.state_dim :] = new_obs[ : self.env.state_dim]
            new_obs = np.concatenate((obs_hist, self.env.map_info, self.env.grid))

        return new_obs, past_rod_center_vec, past_rod_center_pose
    

        
    def play_hil_planner(self):
        
        # set planner param
        rospy.set_param('/planner_type', str(self.planner_type))

        # Create variables
        global past_rod_center_vec
        past_rod_center_vec = None
        past_rod_center_pose = None
        # u_h_cb = np.zeros((2,))


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
        assert self.human in [
            "data",
            "real",
            "policy",
            "planner",
        ], "human arg must be one of 'data', 'policy', or 'real'"
        if self.human == "real":
            if self.env.control_type == "joystick":
                pass
            elif self.env.control_type == "keyboard":
                keys_to_action = get_keys_to_action()
                relevant_keys = set(sum(map(list, keys_to_action.keys()), []))
                pressed_keys = []
            else:
                raise ValueError("control_type must be 'joystick' or 'keyboard'")
        elif self.human == "policy":
            raise NotImplementedError("BC policy not implemented yet")
        elif self.human == "planner":
            assert (
                self.robot == "planner" and exp_run_mode == "coplanning"
            ), "Must be in co-planning mode if human is planner. Change robot mode to 'planner', and exp_run_mode to 'coplanning'."
        else:
            assert self.playback_trajectory is not None, "Must provide playback trajectory"
            if len(self.playback_trajectory["actions"].shape) == 3:
                print(self.playback_trajectory["actions"].shape)
                self.playback_trajectory["actions"] = self.playback_trajectory["actions"].squeeze() 
            assert (
                self.human == "data"
            ), "human arg must be from 'data' if not 'real' or 'policy'"
        # Set self.n_steps to data limit if using human or self.robot data as control inputs
        if self.human == "data" or self.robot == "data":
            self.n_steps = len(self.playback_trajectory["actions"]) - 1
        coplanning = True if (self.human == "planner" and self.robot == "planner") else False

        # Check for valid self.robot arg
        assert self.robot in [
            "planner",
            "data",
            "real",
        ], "self.robot arg must be one of 'planner' or 'data' or 'real' (for turing test)"
        if self.robot == "real":
            print("Robot is real")
            p1_id = 1

        # -------------------------------------------- SETUP PLANNER -------------------------------------------- #
        if self.robot == "planner":
            def _is_safe(state):
                # Check if state is in collision
                return is_safe(state, self.collision_checking_env)

            path = None
            a_horizon_ct = 0
            a_horizon = 1

            if self.planner_type == "vrnn":
                sys.path.append(join(dirname(__file__), "..", "algo", "planners", "cooperative_planner"))
                from algo.planners.cooperative_planner.models import VRNN
                self.artifact_path = join("/home/eleyng/table-carrying-ai/trained_models/vrnn/model.ckpt")
                model = VRNN.load_from_checkpoint(self.artifact_path, map_location=torch.device('cpu'))
                model.eval()
                model.batch_size = self.num_candidates
                model.skip = self.skip
                
                
            elif self.planner_type == "rrt":
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
                goal_state = [self.env.goal[0], self.env.goal[1], np.pi]
            elif self.planner_type == "diffusion_policy":
                sys.path.insert(0, join(dirname(__file__), "../libs/policy","diffusion_policy"))
                import copy

                import dill
                import hydra
                from omegaconf import OmegaConf

                sys.path.append('/home/eleyng/diffusion_policy')

                import diffusion_policy.common.pytorch_util as ptu
                import diffusion_policy.policy.diffusion_transformer_lowdim_policy as dp_policy
                from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
                from diffusion_policy.gym_util.video_recording_wrapper import (
                    VideoRecordingWrapper,
                    VideoRecorder,
                )
                from gym.wrappers import FlattenObservation

                # load checkpoint
                if self.mcfg.human_act_as_cond:
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
                

                policy : dp_policy.DiffusionTransformerLowdimPolicy
                policy = workspace.model
                
                policy.eval().to(self.device)
                policy.num_inference_steps = 100
                policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

                # global FPS, CONST_DT, MAX_FRAMESKIP, SKIP_FRAME

                # FPS = 30 # ticks per second
                # new_fps = 10
                # CONST_DT = 1/FPS #1/FPS # self.skip ticks
                # MAX_FRAMESKIP = 3 # max frames to self.skip
                # SKIP_FRAME = FPS // new_fps
                steps_per_render = max(10 // FPS, 1)
                
                def env_fn():
                    return MultiStepWrapper(
                        VideoRecordingWrapper(
                            FlattenObservation(
                                self.env
                            ),
                            video_recoder=VideoRecorder.create_h264(
                                fps=FPS,
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
                self.env = env_fn()
                
                        
            elif  self.planner_type == "cogail":
                # import models  from cogail
                sys.path.append('/home/eleyng/cogail-table')
                import os

                from a2c_ppo_acktr.model import Policy

                # policy (action output)
                recode_dim = (15 * 7 + 2 + 3*2 + 3) + 2

                actor_critic = Policy(
                    self.env.full_observation.shape,
                    self.env.action_space,
                    recode_dim,  # changes input states & actions to embedding of size recode_dim
                    base_kwargs={
                        "recurrent": False,
                        "code_size": 2,
                        "base_net_small": True,
                    },
                )
                actor_critic.eval().to(self.device)

                # Load model
                model_path = "/home/eleyng/table-carrying-ai/trained_models/cogail/model.pt"
                ckpt = torch.load(model_path, map_location=torch.device('cpu'))
                actor_critic.load_state_dict(ckpt)

            elif self.planner_type == "bc_lstm_gmm":

                sys.path.append('/home/eleyng/robomimic')
                import robomimic.utils.file_utils as FileUtils
                import robomimic.utils.obs_utils as ObsUtils
                import robomimic.utils.tensor_utils as TensorUtils
                import robomimic.utils.torch_utils as TorchUtils
                from robomimic.algo import RolloutPolicy
                from robomimic.envs.env_base import EnvBase

                if self.mcfg.human_act_as_cond:
                    model_path = "/home/eleyng/table-carrying-ai/trained_models/bc_lstm_gmm/model_human_act_as_cond.pth"
                else:
                    model_path = "/home/eleyng/table-carrying-ai/trained_models/bc_lstm_gmm/model.pth"
                policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=model_path, device=self.device, verbose=True)
                config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
                rollout_horizon = config.experiment.rollout.horizon

                # configre rollout policy
                policy.start_episode()

            else:
                print("Invalid planner type, ", self.planner_type)
                raise ValueError("Invalid planner type")

        # ----------------------------------------------------- SIMULTAOR SETUP -----------------------------------------------#

        # reset environment
        obs = self.env.reset()
        if isinstance(obs, tuple): #change table_env reset(): must output both obs and random_seed
            random_seed = obs[1]
            obs = obs[0]
        if self.planner_type in ["bc_lstm_gmm"]:
            obs_q = np.tile(obs, (30, 1))
            past_action = np.zeros_like(obs.squeeze())[..., :4]
        elif self.planner_type in ["diffusion_policy"]:
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
                if self.mcfg.human_act_as_cond:
                    obs_dict_np["past_action"] = past_action.astype(np.float32)
                obs_dict = ptu.dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0] #.detach().numpy()
                assert action.shape[-1] == 4
                del result
            print('Ready!')
            policy.reset()
            # Convert obs to torch tensor if not already
            if not isinstance(obs, torch.Tensor):
                obs_t = torch.from_numpy(obs).float()

        elif self.planner_type in ["vrnn"]:
            obs = torch.from_numpy(obs).float()
            # Initialize running list of past H steps of observations for model inputs (need tf2model for model input conversion)
            s_queue = torch.zeros(
                (self.mcfg.H // self.mcfg.skip + 1, obs.shape[0]), dtype=torch.float32
            ).to(self.device)
            s_queue = update_queue(s_queue, obs.unsqueeze(0))
            u_queue = torch.zeros((self.mcfg.H // self.mcfg.skip, self.mcfg.ASIZE), dtype=torch.float32).to(
                self.device
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
        plan_cter = 0

        # ----------------------------------------------- SETUP EXPERIMENT VIS -------------------------------------------- #

        # Initialize list of past states visited in the simulator, for visualization purposes
        if self.display_past_states:
            past_states = []
            past_states.append(obs.tolist())

        # Initialize list of ground truth waypoints, if displaying ground truth
        if self.display_gt:
            waypoints_true = self.playback_trajectory["states"].tolist()

        # ------------------------------------------ SETUP DATA STRUCTS FOR MODEL USE ------------------------------------------ #

        


        ### ---------------------------------------------------- GAME LOOP ---------------------------------------------------- ###
        
        action_plan = None
        loop_timer_begin = time.time()
        rate = rospy.Rate(10) #rospy.Rate(FPS) #if self.planner_type != "diffusion_policy" else rospy.Rate(10)
        start = time.time()
        while running and not rospy.is_shutdown():
            # rospy.loginfo("OBS RATE")
            

            loops = 0

            if done:
                # time.sleep(1)
                pygame.quit()
                print("Episode finished after {} timesteps".format(n_iter + 1))
                break

            else:
                if self.display_gt:
                        self.env.draw_gt(waypoints_true)

                    
                # -------------------------------------------- GET HUMAN INPUT -------------------------------------------- #
                if self.human == "real":

                    if self.env.control_type == "joystick":
                        # Get action from joysticksubscriber
                        u_h = torch.from_numpy(self.u_h_cb).unsqueeze(0)
                        print("human joystick action", u_h)
                    else:
                        u_h = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
                        u_h = set_action_keyboard(u_h)
                        u_h = torch.from_numpy(u_h[1, :]).unsqueeze(0)
                elif self.human == "planner":
                    if self.planner_type == "vrnn":
                        if n_iter <= self.mcfg.H + self.mcfg.skip:
                            u_h = self.playback_trajectory["actions"][n_iter, 2:]
                            u_h = torch.from_numpy(u_h).unsqueeze(0)
                    else:
                        pass
                    pass
                else:
                    # If using human data, then get the actions from the playback trajectory
                    assert (
                        self.human == "data"
                    ), "human arg must be from 'data' if not 'real', 'planner', or 'policy'"
                    n_iter = min(
                        n_iter, self.playback_trajectory["actions"].shape[0] - 1
                    )  # Needed to account finish the playback
                    # else:
                    #     assert n_iter < actions.shape[0], "Ran out of human actions from data."
                    u_h = self.playback_trajectory["actions"][n_iter, 2:]
                    u_h_npy = u_h
                    u_h = torch.from_numpy(u_h).unsqueeze(0)
                # interpolation = 0 
                # print("interpolation: ", interpolation)

                # while time.time() > next_game_tick and loops < MAX_FRAMESKIP and not done:

                    
                    
                ### --------------------------------------------- GET ROBOT INPUTS -------------------------------------------- ####

                if (a_horizon_ct < a_horizon) and n_iter != 0 and self.planner_type == "diffusion_policy":
                    # Fetch the next action from the previously planned action plan
                    u_r = torch.from_numpy(
                        action_plan[a_horizon_ct, :2]
                    ).unsqueeze(0)
                    u_r /= np.linalg.norm(u_r)
                    if coplanning:
                        u_h = torch.from_numpy(
                            action_plan[a_horizon_ct, 2:]
                        ).unsqueeze(0)
                    u_all = torch.cat((u_r, u_h), dim=-1)
                    print("U_h no plan", u_h)

                    # Update past action for HUMAN input
                    past_action[:-1, :] = past_action[1:, :]
                    past_action[-1, :2] = action_plan[a_horizon_ct, :2].squeeze()
                    past_action[-1, 2:] = u_h.flatten() if not coplanning else action_plan[a_horizon_ct, 2:]
                    a_horizon_ct += 1 if zoh_ct % SKIP_FRAME == 0 else 0
                    zoh_ct += 1
                    # print("no planning, ")

                else:

                    start_plan = time.time()
                    # -------------------------------------------- PLANNING PERIOD -------------------------------------------- #

                    # If we are in the planning period, then we need to continue updating the state history queue, get the next observation
                    # from the simulator by feeding the human input and self.robot input from PID, which controls to waypoints planned by the model.
                    if self.planner_type == "vrnn":
                        # -------------------------------------------- IF USING VRNN: GET WAYPOINTS -------------------------------------------- #
                

                        if (self.human == "data" and self.robot == "data"):
                            # # Feed first H steps of state history into simulator
                            u_r = torch.from_numpy(
                                self.playback_trajectory["actions"][n_iter, :2]
                            ).unsqueeze(0)
                            # if coplanning:
                            #     u_all = torch.from_numpy(self.playback_trajectory["actions"][n_iter, :]).unsqueeze(0)
                            # else:
                            #     u_all = torch.cat(
                            #         (u_r, u_h), dim=-1
                            #     )  # player 1 is blue, and in this sim human is player 2 (orange)
                            # # Update action history queue
                            # print('n_iter less than {}, horizon {}.'.format(self.mcfg.H + self.mcfg.skip, self.mcfg.SELF.SEQ_LEN))
                            if coplanning:
                                u_all = torch.zeros((1, 4)) 
                            else:
                                # u_r = torch.zeros((1, 2))
                                u_all = torch.cat(
                                    (u_r, u_h), dim=-1
                                )  # player 1 is blue, and in this sim human is player 2 (orange)
                            u_queue = update_queue(u_queue, u_all)

                        else:
                            # print("here")
                            # if (n_iter % self.mcfg.skip != 0) and not (
                            #     human == "data" and self.robot == "data"
                            # ):
                            if a_horizon_ct < a_horizon and path is not None:
                                n_iter += 1
                                a_horizon_ct += 1
                                
                            else:
                                with torch.no_grad():

                                    s_tf = tf2model(s_queue, self.env.obstacles, zero_padding=False).repeat(self.mcfg.BSIZE, 1, 1)
                                    u = u_queue.repeat(self.mcfg.BSIZE, 1, 1).float()
                                    sample = model.sample(s_tf, u, seq_len=self.mcfg.SEQ_LEN)

                                waypoints = tf2sim(
                                    sample[:, :, :4],
                                    s_queue,
                                    (self.mcfg.H // self.mcfg.skip),
                                )

                                # Evaluate the rewards the batch of sampled trajectories using custom reward function
                                eval = np.sum(
                                    np.array(
                                        [
                                            compute_reward(
                                                waypoints[i, :, :4],
                                                self.env.goal,
                                                self.env.obstacles,
                                                interaction_forces=self.include_interaction_forces_in_rewards,
                                                env=self.env,
                                                u_h=None,
                                                collision_checking_env=self.collision_checking_env,
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

                            idx = a_horizon_ct + self.mcfg.lookahead // self.mcfg.skip
                            if coplanning:
                                pid_actions = pid_single_step(
                                    self.env,
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
                                    self.env,
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

                                # u_r = torch.from_numpy(
                                #     self.playback_trajectory["actions"][n_iter, :2]
                                # ).unsqueeze(0)
                                # print("N_ITER", n_iter)
                                print('shapes', u_r.shape, u_h.shape)
                                u_all = torch.cat((u_r, u_h), dim=-1)

                            else:
                                u_all = torch.from_numpy(
                                    np.clip(pid_actions, -1.0, 1.0)
                                ).unsqueeze(0)
                            # u_r[:, :2] = 0
                            u_queue = update_queue(u_queue, u_all)
                            # print('u_all: ', u_all)
                    elif self.planner_type in ["diffusion_policy"]:

                        a_horizon_ct = 0
                        zoh_ct = 1
                        obs_dict_np = {"obs" : obs_model[::SKIP_FRAME, ...].astype(np.float32)}
                        if self.mcfg.human_act_as_cond:
                            obs_dict_np["past_action"] = past_action[::SKIP_FRAME, ...].astype(np.float32)
                            print("past_action: ", past_action[::SKIP_FRAME, ...].astype(np.float32))
                        obs_dict = ptu.dict_apply(obs_dict_np,
                            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                                        # lambda x: x.to(device=self.device))
                        # run policy
                        with torch.no_grad():
                            result = policy.predict_action(obs_dict)
                            plan_cter += 1

                            # np_action_dict = ptu.dict_apply(action_dict,
                            #     lambda x: x.detach().to('cpu').numpy())
                            action_plan = result['action'][0].detach().to('cpu').numpy()

                            # action_plan = np_action_dict['action']
                            u_r = action_plan[a_horizon_ct, :2]
                            u_r /= np.linalg.norm(u_r) if np.sum(u_r) != 0. else 1.0
                            u_r = torch.from_numpy(u_r).unsqueeze(0)
                            print("U_h plan", u_h)
                            if coplanning:
                                print("coplanning")
                                
                                u_h = torch.from_numpy(
                                    action_plan[a_horizon_ct, 2:]
                                ).unsqueeze(0)
                            u_all = torch.cat((u_r, u_h), dim=-1)

                        # update past action
                        past_action[:-1, :] = past_action[1:, :]
                        past_action[-1, :2] = action_plan[a_horizon_ct, :2].squeeze()
                        past_action[-1, 2:] = u_h.flatten() if not coplanning else action_plan[a_horizon_ct, 2:]

                        a_horizon_ct += 1 if zoh_ct % SKIP_FRAME == 0 else 0
                        zoh_ct += 1


                        delta_plan = time.time() - start_plan
                        delta_plan_sum += delta_plan
                        print("DEVICEEEEEEE", self.device)


                    elif self.planner_type == "bc_lstm_gmm":
                        obs_dict = {}
                        if self.mcfg.human_act_as_cond:
                            if coplanning:
                                u_h = past_action[2:]
                            obs = np.concatenate((obs, u_h), axis=-1)
                            obs_dict["all"] = obs
                        else:
                            obs_dict["all"] = obs
                        
                        action = policy(ob=obs_dict)
                        plan_cter += 1
                        u_r = torch.from_numpy(action[:2]).unsqueeze(0)
                        u_r /= np.linalg.norm(u_r)
                        if coplanning:
                            u_h = torch.from_numpy(action[2:]).unsqueeze(0)
                        u_all = torch.cat((u_r, u_h), dim=-1)

                        # Update past action for ROBOT
                        past_action[:2] = action[:2]
                        # Update past action for HUMAN input
                        past_action[2:] = u_h if not coplanning else action[2:]

                    elif self.planner_type == "cogail":

                        obs = np.divide((obs - self.env.obs_space_low), self.env.obs_space_range) 
                        obs = obs.unsqueeze(0) ## TODO: verify that batch is 1 for each eval run
                        eval_recurrent_hidden_states = torch.zeros(
                            1, 1, device=self.device)
                        eval_masks = torch.zeros(1, 1, device=self.device)
                        with torch.no_grad():
                            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                                obs,
                                random_seed,
                                eval_recurrent_hidden_states,
                                eval_masks,
                                deterministic=True)
                            plan_cter += 1
                            u_r = action[:, :2]
                            u_r /= np.linalg.norm(u_r)

                        if coplanning:
                            u_h = action[:, 2:]
                        u_all = torch.cat((u_r, u_h), dim=-1) # (1,4)
                    else:
                        raise Exception("Planner/policy not valid.")

                delta_plan = time.time() - start_plan
                print("Planning time: ", delta_plan)
                delta_plan_sum += delta_plan
                if self.display_pred and self.planner_type in ["vrnn"] and path is not None:
                    self.env.update_prediction(path.tolist())

                # -------------------------------------------- UPDATE ENVIRONMENT -------------------------------------------- #

                # Publish desired actions
                if self.planner_type != "diffusion_policy":
                    action_plan = u_all.detach().numpy()

                send_plan_start = time.time()
                send_plan(plan=action_plan[:a_horizon, :].flatten(), duration=CONST_DT, curr_pub=self.pub_action_plan)
                print("send plan time: ", time.time() - send_plan_start)

                if self.planner_type != "cogail":
                    obs, reward, done, info = self.env.step(u_all.detach().numpy())
                    # print("fluency", info["fluency"]["inter_f"])
                else:
                    obs, reward, succ, done, info, random_seed = self.env.step(u_all.detach().numpy())

                obs, past_rod_center_vec, past_rod_center_pose = self.mocap_pose_to_obs(self.env.map_info, self.env.grid, past_rod_center_vec, past_rod_center_pose)
                self.env.table.x = past_rod_center_pose[0]
                self.env.table.y = past_rod_center_pose[1]
                self.env.table.angle = past_rod_center_pose[2]
                print("angle", self.env.table.angle)
                self.env.redraw()
                print("mocap pose: ", obs)
                loop_time = time.time() - loop_timer_begin
                # print("loop time", loop_time)
                loop_timer_begin = time.time()

                if self.planner_type in ["diffusion_policy"]:
                    # update obs_model
                    obs_model[:-1, ...] = obs_model[1:, ...]
                    obs_model[-1, ...] = obs[-1, ...]
                    obs_t = torch.from_numpy(obs).float()

                elif self.planner_type in ["vrnn"]:
                    obs = torch.from_numpy(obs).float()
                    s_queue = update_queue(s_queue, obs.unsqueeze(0))
                # Convert obs to torch tensor if not already
                elif not isinstance(obs, torch.Tensor) and self.planner_type not in ["diffusion_policy"]:
                    obs = torch.from_numpy(obs).float()
                else:
                    pass
                n_iter += 1

                if self.display_past_states:
                    past_states.append(obs.tolist())
                    self.env.draw_past_states(past_states)

                save_state = torch.from_numpy(np.array([self.env.table.x, self.env.table.y, self.env.table.angle], dtype=np.float32))
                trajectory["states"].append(save_state)

                # if self.planner_type in ["bc_lstm_gmm", "vrnn"]:
                #     trajectory["states"].append(obs)
                # elif self.planner_type in ["cogail"]:
                #     trajectory["states"].append(torch.from_numpy(self.env.current_state))
                # elif self.planner_type in ["diffusion_policy"]:
                #     trajectory["states"].append(obs_t[-1, ...]) #should be torch
                # else:
                #     trajectory["states"].append(obs[-1, ...])

                if self.robot == "planner":
                    if self.planner_type == "vrnn":
                        if path is not None:
                            trajectory["plan"].append(torch.tensor(path))
                    elif self.planner_type == "rrt":
                        trajectory["plan"].append(path.tolist())
                    elif self.planner_type == "diffusion_policy":
                        trajectory["plan"].append(action_plan[a_horizon_ct].tolist())
                trajectory["actions"].append(u_all)
                trajectory["rewards"].append(torch.tensor(reward))

                if done:
                    if self.planner_type == "diffusion_policy":
                        if info["success"][-1]:
                            success = True
                            # _ = self.env.reset()
                        else:
                            success = False
                    else:
                        if info["success"]:
                            success = True
                        else:
                            success = False
                    # self.env.render(mode="human")
                    running = False
                    break

                next_game_tick += CONST_DT
                loops += 1


            # Update display
            if not done and self.planner_type not in ["diffusion_policy"]:
                self.env.redraw()
                clock.tick(FPS)

                # process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if (
                            event.key in relevant_keys
                            and self.human == "real"
                            and self.env.control_type == "keyboard"
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
        # print("Average planning time per planning loop: ", delta_plan_sum / plan_cter)
        # print("Duration of run: ", duration)

        if self.planner_type == "diffusion_policy":
            self.env.close()
            del self.env

        pygame.quit()

        if not (self.human == "data" and self.robot == "data"):
            # Save trajectory
            trajectory["states"] = torch.stack(trajectory["states"], dim=0).numpy()
            # if self.planner_type in ["bc_lstm_gmm", "cogail"]:
            #     trajectory["states"] = torch.stack(trajectory["states"], dim=0).numpy()
            # elif not isinstance(trajectory["states"], torch.Tensor):
            #     if self.planner_type == "diffusion_policy":
            #         trajectory["states"] = np.array(trajectory["states"])
            #     # elif self.planner_type == "cogail":
            #     #     trajectory["states"] = np.array(trajectory["states"])[:, -1, :]
            #     elif self.planner_type == "vrnn":
            #         trajectory["states"] = torch.stack(trajectory["states"], dim=0).numpy()
            if self.robot == "planner":
                if self.planner_type == "vrnn":
                    trajectory["plan"] = torch.stack(trajectory["plan"], dim=0).numpy()
            trajectory["actions"] = torch.stack(trajectory["actions"], dim=0).numpy().squeeze()
            trajectory["rewards"] = torch.stack(trajectory["rewards"], dim=0).numpy().squeeze()
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
            trajectory["n_iter"] = trajectory["states"].shape[0]
            n_iter = trajectory["n_iter"]
            trajectory["duration"] = duration

        return trajectory, success, n_iter, duration, delta_plan_sum / plan_cter