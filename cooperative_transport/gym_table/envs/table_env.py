import os

os.environ["SDL_AUDIODRIVER"] = "dsp"  # for training on cluster
from os import mkdir
from os.path import exists

import math
import pickle
import random
import time
from typing import Dict, Tuple, Union, List
import numpy as np

import gym
from gym import spaces
import pygame

# from gym.envs.classic_control import rendering
from PIL import Image
import cv2 as cv

from cooperative_transport.gym_table.envs.game_objects.game_objects import (
    Obstacle,
    Table,
    Target,
    Agent,
)

from cooperative_transport.gym_table.envs.utils import (
    load_cfg,
    FPS,
    BLACK,
    WINDOW_W,
    WINDOW_H,
    STATE_W,
    STATE_H,
    rad,
    debug_print,
    set_action_keyboard,
    set_action_joystick,
)

VERBOSE = False  # For debugging


def debug_print(*args):
    if not VERBOSE:
        return
    print(*args)


class TableEnv(gym.Env):
    """The table environment.

    This environment consists of two agents rigidly attached to a table.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def handle_kwargs(self, obs, control, map_config, ep):
        self.obs_type = obs  # type of observation space, rgb or discrete
        self.control_type = control  # type of control input, keyboard or joystick
        self.map_cfg = load_cfg(map_config)
        # Episode initiation
        self.ep = ep

    def init_pygame(self):
        """Initialize the environment."""
        if self.render_mode == "headless":
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            print("Running with display.")
        pygame.init()
        self.screen = pygame.display.set_mode(([WINDOW_W, WINDOW_H]))
        self.viewer = None

    def init_data_paths(self, map_config, run_mode, strategy_name):
        map_name = map_config.split("/")[-1].split(".")[0]
        if not exists(os.path.join(os.path.dirname(__file__), run_mode)):
            debug_print("Making base directories.")
            mkdir(os.path.join(os.path.dirname(__file__), run_mode))
        if not exists(os.path.join(os.path.dirname(__file__), run_mode, map_name)):
            mkdir(os.path.join(os.path.dirname(__file__), run_mode, map_name))
        self.base_dirname = os.path.join(
            os.path.dirname(__file__), run_mode, map_name, strategy_name
        )
        self.dirname = os.path.join(
            os.path.dirname(__file__), run_mode, map_name, strategy_name, "trajectories"
        )  # "runs/two-player-bc") #"../results/one-player-bc")
        self.dirname_fluency = os.path.join(
            os.path.dirname(__file__), run_mode, map_name, strategy_name, "fluency"
        )  # "runs/two-player-bc") #"../results/one-player-bc")
        self.dirname_vis = os.path.join(
            os.path.dirname(__file__), run_mode, map_name, strategy_name, "figures"
        )  # "runs/two-player-bc") #"../results/one-player-bc")
        self.map_config_dir = os.path.join(
            os.path.dirname(__file__), run_mode, map_name, strategy_name, "map_cfg"
        )  # "runs/two-player-bc") #"../results/one-player-bc")
        if not exists(self.base_dirname):
            mkdir(self.base_dirname)
            mkdir(self.dirname)
            mkdir(self.dirname_fluency)
            mkdir(self.dirname_vis)
            mkdir(self.map_config_dir)
        debug_print("Saving to directory: ", self.base_dirname)

        self.file_name = os.path.join(self.dirname, "ep_" + str(self.ep) + ".pkl")
        self.config_file_name = os.path.join(
            self.map_config_dir, "config_params-ep_" + str(self.ep) + ".npz"
        )
        self.file_name_fluency = os.path.join(
            self.dirname_fluency, "ep_" + str(self.ep) + ".npz"
        )
        if not os.path.exists(os.path.dirname(self.file_name_fluency)):
            os.makedirs(os.path.dirname(self.file_name_fluency))
        if not os.path.exists(os.path.dirname(self.file_name)):
            os.makedirs(os.path.dirname(self.file_name))
        if not os.path.exists(os.path.dirname(self.config_file_name)):
            os.makedirs(os.path.dirname(self.config_file_name))

        self.dirname_vis_ep = os.path.join(
            self.dirname_vis, "ep_" + str(self.ep) + "_images"
        )
        if not exists(self.dirname_vis_ep):
            debug_print("Making image directory: ", self.dirname_vis)
            mkdir(self.dirname_vis_ep)

        debug_print("Data saved location: ", self.file_name)

    def init_env(self):
        self.init_pygame()
        self.n = 2  # number of players
        # load from saved env config file
        if self.load_map is not None and self.run_mode == "eval":
            map_run = dict(np.load(self.load_map, allow_pickle=True))
            # table init pose
            table_cfg = [
                map_run["table"].item()["x"] / WINDOW_W,
                map_run["table"].item()["y"] / WINDOW_H,
                map_run["table"].item()["angle"],
            ]
            # table goal pose
            goal_cfg = [
                map_run["goal"].item()["goal"][0] / WINDOW_W,
                map_run["goal"].item()["goal"][1] / WINDOW_H,
            ]
            # table obstacles as encoding
            obs_lst_cfg = map_run["obstacles"].item()["obs_lst"]
            num_obs_cfg = map_run["obstacles"].item()["num_obstacles"]
        else:
            # randomly sample new table init position
            table_cfg = self.map_cfg["TABLE"][
                random.sample(range(0, len(self.map_cfg["TABLE"])), 1)[0]
            ]

        self.table = Table(
            x=table_cfg[0] * WINDOW_W,
            y=table_cfg[1] * WINDOW_H,
            angle=table_cfg[2],
            physics_control_type=self.physics_control_type,
        )
        self.config_params = {}
        self.table_params = {}
        self.table_params["x"] = self.table.x
        self.table_params["y"] = self.table.y
        self.table_params["angle"] = self.table.angle
        self.table_init = np.array([self.table.x, self.table.y, self.table.angle])
        debug_print("Table initial configuration: ", self.table_params)

        self.player_1 = Agent()
        self.player_2 = Agent()

        # RANDOM OBSTACLE CONFIG
        self.obs_dim = len(self.map_cfg["OBSTACLES"][0]["POSITIONS"])
        self.num_obstacles = np.random.choice(range(1, self.max_num_obstacles + 1), 1)[
            0
        ]
        self.visible_obs = 1  # float(args["vis"])

        # create obstacle
        self.obs_lst_idx = random.sample(
            range(0, len(self.map_cfg["OBSTACLES"][0]["POSITIONS"])), self.num_obstacles
        )

        if self.load_map is not None and self.run_mode == "eval":
            self.obs_lst_idx = obs_lst_cfg
            self.num_obstacles = num_obs_cfg
        self.obs_lst = [
            self.map_cfg["OBSTACLES"][0]["POSITIONS"][i] for i in self.obs_lst_idx
        ]

        # initialize obstacles & obstacle sprites
        self.obstacles = np.zeros((self.num_obstacles, 2))
        self.obs_sprite = []

        for i in range(len(self.obs_lst)):
            obs = np.array(
                [
                    self.obs_lst[i][0] * WINDOW_W,
                    self.obs_lst[i][1] * WINDOW_H,
                ]
            )
            self.obstacles[i] = obs
            self.obs_sprite.append(
                Obstacle(
                    self.obstacles[i], size=self.map_cfg["OBSTACLES"][0]["SIZES"][0]
                )
            )
        self.obs_params = {}
        self.obs_params["obstacles"] = self.obstacles
        self.obs_params["num_obstacles"] = self.num_obstacles
        self.obs_params["obs_lst"] = self.obs_lst_idx
        self.obs_params["obs_dim"] = self.obs_dim
        debug_print("Obstacle configuration: ", self.obs_params)

        # RANDOM GOAL CONFIG
        goal_rnd = self.map_cfg["GOAL"][
            random.sample(range(0, len(self.map_cfg["GOAL"])), 1)[0]
        ]
        if self.load_map is not None and self.run_mode == "eval":
            goal_rnd = goal_cfg

        debug_print("goal_rnd", goal_rnd)
        self.goal = np.array([goal_rnd[0] * WINDOW_W, goal_rnd[1] * WINDOW_H])
        self.goal_params = {}
        self.goal_params["goal"] = self.goal
        debug_print("Goal configuration: ", self.goal_params)

        # SAVE CONFIGURATION
        self.config_params = {}
        self.config_params["table"] = self.table_params
        self.config_params["obstacles"] = self.obs_params
        self.config_params["goal"] = self.goal_params

        # find dist2goal, dist2obs
        direction = self.goal - np.array([self.table.x, self.table.y])
        self.dist2goal = np.linalg.norm(direction)
        self.avoid = np.array(self.obstacles) - np.array([self.table.x, self.table.y])
        self.dist2obs = np.linalg.norm(self.avoid, axis=1)
        self.wallpts = np.array(
            [[0, 0], [0, WINDOW_H], [WINDOW_W, WINDOW_H], [WINDOW_W, 0], [0, 0]]
        )
        self.target = Target(self.goal)
        self.sprite_list = pygame.sprite.Group()
        self.sprite_list.add(self.table)
        self.sprite_list.add(self.target)
        if self.visible_obs == 1:
            for i in range(self.num_obstacles):
                self.sprite_list.add(self.obs_sprite[i])
        elif self.visible_obs == 0.5:
            for i in range(self.num_obstacles):
                prob = random.randint(0, 1)
                if prob:
                    self.sprite_list.add(self.obs_sprite[i])
        self.done_list = pygame.sprite.Group()
        self.done_list.add(self.target)
        self.done_list.add(self.obs_sprite)
        self.done = False
        self.success = False

        # update metrics through reward
        table_state = np.expand_dims(
            np.array([self.table.x, self.table.y, self.table.angle]), axis=0
        )
        self.update_metrics(table_state)
        # reward = self.compute_reward(np.array([table_state]))

        self.prev_time = time.time()
        self.observation = self.get_state()
        self.n_step = 0

        self.cap = self.player_1.cap  # TODO: match force cap in game_objects
        self.velocity_cap = self.player_1.velocity_cap

    def init_action_space(self):
        # ----------------------------------------------- Action and Observation Spaces -------------------------------------------------------------
        if self.control_type == "keyboard":
            # define action space
            self.action_space = spaces.Discrete(25)
        elif self.control_type == "joystick":
            # continuous action space specified by two pairs of joystick axis
            action_space_low = np.array([-1.0, -1.0, -1.0, -1.0])
            action_space_high = np.array([1.0, 1.0, 1.0, 1.0])
            self.action_space = spaces.Box(
                action_space_low, action_space_high, dtype=np.float32
            )
        else:
            raise NotImplementedError("Unknown control type: %s" % self.control_type)

    def init_observation_space(self):
        # discrete observation space
        if self.obs_type == "discrete":

            # define observation space
            self.obs_space_low = np.array(
                [
                    0.0,  # x
                    0.0,  # y
                    -1.0,  # angle (cos(angle))
                    -1.0,  # angle (sin(angle))
                    0.0,  # target x
                    0.0,  # target y
                    0.0,  # x position of most relevant obstacle
                    0.0,  # y position of most relevant obstacle
                    -2 * np.pi,  # angle (without angle transform)
                ]
            )
            self.obs_space_hi = np.array(
                [
                    WINDOW_W,
                    WINDOW_H,
                    1.0,
                    1.0,
                    WINDOW_W,
                    WINDOW_H,
                    WINDOW_W,
                    WINDOW_H,
                    2 * np.pi,
                ]
            )

        elif self.obs_type == "rgb":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
            )
        else:
            raise NotImplementedError(
                "Unknown observation space type: %s" % self.obs_type
            )

        self.observation_space = spaces.Box(
            self.obs_space_low, self.obs_space_hi, dtype=np.float32
        )

        self.observation_dim = self.observation_space.shape[0]
        self.obs_space_range = self.obs_space_hi - self.obs_space_low

    def __init__(
        self,
        render_mode="gui",
        obs="discrete",
        control="joystick",
        map_config="cooperative_transport/gym_table/config/maps/rnd_obstacle_v2.yml",
        load_map=None,  # npz file storing map configuration form playback
        run_mode="demo",  # "demo", "eval"; for demo data collection or experiments
        strategy_name="success_1",
        ep=0,
        dt=1 / 30,
        physics_control_type="force",
        state_dim=9,
        max_num_obstacles=3,
        set_obs=None,
    ) -> None:
        self.state_dim = state_dim
        self.render_mode = render_mode
        self.dist2wall_list = None
        self.dist2wall = None
        self.avoid = None
        self.done = None
        self.delta_t = dt
        self.n = None
        self.velocity_cap = None
        self.cap = None
        self.n_step = None
        self.observation = None
        self.prev_time = None
        self.success = None
        self.done_list = None
        self.sprite_list = None
        self.target = None
        self.wallpts = None
        self.dist2obs = None
        self.dist2goal = None
        self.goal = None
        self.obs_sprite = None
        self.obstacles = None
        self.visible_obs = None
        self.num_obstacles = None
        self.max_num_obstacles = max_num_obstacles
        self.player_2 = None
        self.player_1 = None
        self.table = None
        self.inter_f = None
        self.dirname_vis_ep = None
        self.file_name_fluency = None
        self.file_name = None
        self.ep = None
        self.viewer = None
        self.screen = None
        self.base_dirname = None
        self.dirname_fluency = None
        self.dirname_vis = None
        self.control_type = None
        self.map_cfg = None
        self.obs_type = None
        self.dirname = None
        self.physics_control_type = physics_control_type
        self.obs_space_dim = None
        self.observation_space = None
        self.obs_space_range = None
        self.obs_space_low = None
        self.obs_space_hi = None
        self.prediction = None
        self.ground_truth_states = None
        self.past_states = None
        self.completed_traj = None
        self.completed_traj_fluency = None

        self.init_pygame()
        self.handle_kwargs(obs, control, map_config, ep)

        self.data = []
        # synthetic data
        self.init_data_paths(map_config, run_mode, strategy_name)

        self.interact_mode = True

        self.cumulative_reward = 0  # episode's cumulative reward
        self.fluency = {
            "inter_f": [],
            "f_del": [],
            "h_idle": [],
            "r_idle": [],
            "conf": [],
        }

        self.load_map = load_map
        self.run_mode = run_mode
        self.init_action_space()
        self.init_observation_space()
        self.init_env()

    def _set_action(self, action):
        if self.control_type == "keyboard":
            return set_action_keyboard(action)
        elif self.control_type == "joystick":
            return set_action_joystick(action)

    def step(
        self, action: List[np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Union[bool, float]]]:
        """Step the environment forward.

        Parameters
        ----------
        action : List[np.ndarray], shape=(4)
            Actions of each the RL agent in UNSCALED FORM i.e. [-1., 1.]. Represents a force vector f in R^2.

        Returns
        -------
        observation : np.ndarray, shape=(6)
            Observations. We currently observe on the full state + player actions. TODO: change this later.
        reward : float
            Reward at this step.
        done : bool
            Boolean indicating whether the episode satisfies an end condition.
        info : Dict[str, Union[bool, float]]
            Information about the run. For now, contains the reward and done bool.
        """
        self.n_step += 1
        debug_print("dt in step: ", self.delta_t)
        action_for_update = self._set_action(
            action
        )  # action is [-1, 1]; so is action_for_update

        if action is not None:
            f1, f2, x, y, angle, vx, vy, angle_speed = self.table.update(
                action_for_update, self.delta_t, self.n_step, self.interact_mode
            )

        self.player_1.f = f1  # note: is action scaled by factor of 50!
        self.player_2.f = f2  # note: is action scaled by factor of 50!
        self.table.x = x
        self.table.y = y
        self.table.angle = angle
        self.table.x_speed = vx
        self.table.y_speed = vy
        self.table.angle_speed = angle_speed

        info = {}

        table_state = np.expand_dims(
            np.array([self.table.x, self.table.y, self.table.angle]), axis=0
        )

        self.update_metrics(table_state)
        self.observation = self.get_state()
        self.done = self.check_collision()
        self.compute_fluency(action)
        reward = self.compute_reward(table_state)
        # reward = 0.0
        self.cumulative_reward += reward
        debug_print(
            "Done? ",
            self.done,
            " Success? ",
            self.success,
            "Cumulative r: ",
            self.cumulative_reward,
        )
        # compute reward BEFORE dumping data to make sure terminal reward is recorded
        self.data.append(
            [self.table.x, self.table.y, self.table.angle]  # pos
            + [self.table.x_speed, self.table.y_speed, self.table.angle_speed]  # vel
            + [action]  # action
            + [reward]  # reward
            + [self.done]  # done
            + [self.success]
            # + ["f1 (RL)", self.player_1.f]
            # + ["f2", self.player_2.f]
            + [self.n_step]  # step
            + [self.delta_t]  # dt
            + [list(self.goal)]  # goal
            + list(self.obstacles)  # obs
            + list(self.wallpts)  # wallpts
            + [self.cumulative_reward]  # cumulative reward task
        )

        info["step"] = self.n_step
        info["reward"] = reward
        info["done"] = self.done
        info["success"] = self.success

        # dump data upon episode complete
        if self.done:
            np.savez(self.file_name_fluency, **self.fluency)
            np.savez(self.config_file_name, **self.config_params)
            pickle.dump(self.data, open(self.file_name, "wb"))
            debug_print("Data saved!")
        else:
            self.redraw()

        return self.observation, reward, self.done, info

    def compute_fluency(self, action):
        if self.control_type == "keyboard":
            return self.compute_fluency_disc(action)
        elif self.control_type == "joystick":
            return self.compute_fluency_cont(action)

    def tf_w2ego(self, vec):
        """Transforms a vector from table frame to ego frame."""
        return np.array(
            [
                vec[0] * np.cos(self.table.angle) + vec[1] * np.sin(self.table.angle),
                -vec[0] * np.sin(self.table.angle) + vec[1] * np.cos(self.table.angle),
            ]
        )

    def compute_fluency_cont(self, action) -> float:
        # Interaction forces calculated from:
        # Kumar, et. al. "Force Distribution in Closed Kinematic Chains", IEEE Journal of Robotics and Automation, 1988.)
        # Interaction forces = (F_1 - F_2) dot (r_1 - r_2)

        self.inter_f = (self.player_1.f - self.player_2.f) @ (
            self.table.table_center_to_player1 - self.table.table_center_to_player2
        )
        self.fluency["inter_f"].append(self.inter_f)
        # Human Idle: if all actions are 0, then it is idle
        if not np.any(self.player_2.f):
            self.fluency["h_idle"].append(1)
        else:
            self.fluency["h_idle"].append(0)
        # Robot Idle
        if not np.any(self.player_1.f):
            self.fluency["r_idle"].append(1)
        else:
            self.fluency["r_idle"].append(0)
        # Concurrent action: when both are acting
        if np.any(self.player_2.f) and np.any(self.player_1.f):
            self.fluency["conf"].append(1)
        else:
            self.fluency["conf"].append(0)
        # Funct. delay: when both are not acting
        if (not np.any(self.player_2.f)) and (not np.any(self.player_1.f)):
            self.fluency["f_del"].append(1)
        else:
            self.fluency["f_del"].append(0)

    def compute_fluency_disc(self, action):
        if action >= 9:
            self.fluency["conf"].append(1)
        else:
            self.fluency["conf"].append(0)
        if action == 0:
            self.fluency["f_del"].append(1)
        else:
            self.fluency["f_del"].append(0)
        if action in [1, 2, 3, 4]:
            self.fluency["r_idle"].append(1)
        else:
            self.fluency["r_idle"].append(0)
        if action in [5, 6, 7, 8]:
            self.fluency["h_idle"].append(1)
        else:
            self.fluency["h_idle"].append(0)
        if action in [10, 13, 20, 23]:
            self.fluency["inter_f"].append(1)
        else:
            self.fluency["inter_f"].append(0)

    def update_metrics(self, states):
        # dist2goal
        self.dist2goal = np.linalg.norm(states[:, :2] - self.goal, axis=1)
        self.dist2obs = np.asarray(
            [
                np.linalg.norm(
                    states[:, :2]
                    - np.array([self.obs_sprite[i].x, self.obs_sprite[i].y]),
                    axis=1,
                )
                for i in range(self.num_obstacles)
            ],
            dtype=np.float32,
        )
        # print("vec dist2obs", self.dist2obs.shape, self.dist2obs)

        self.avoid = np.array(self.obstacles) - np.array([self.table.x, self.table.y])

        # dist2wall
        self.dist2wall_list = np.vstack(
            (
                states[:, 0],
                WINDOW_W - states[:, 0],
                states[:, 1],
                WINDOW_H - states[:, 1],
            )
        ).T
        # print("vec dist2wall!", self.dist2wall_list)

        # self.dist2obs = np.array(self.dist2obs)
        self.dist2wall = np.min(self.dist2wall_list, axis=1, keepdims=True)
        # print("vec dist2wall!!!", self.dist2wall)

    # TODO: clean up
    def compute_reward(self, states=None) -> float:
        # states should be an N x 3 array
        assert (
            len(states.shape) == 2
        ), "state shape mismatch for vectorized compute_reward"
        assert (
            states.shape[1] == 3
        ), "state shape mismatch for vectorized compute_reward"
        assert states is not None, "states parameter cannot be None"

        n = states.shape[0]
        reward = np.zeros(n)
        # slack reward
        reward += -0.1
        if states is not None:
            dg = np.linalg.norm(states[:, :2] - self.goal, axis=1)
        else:
            dg = self.dist2goal
        a = 0.98
        const = 100.0
        r_g = 10.0 * np.power(a, dg - const)
        reward += r_g

        r_obs = np.zeros(n)
        b = -8.0
        c = 0.9
        const = 150.0

        if states is not None:
            d2obs_lst = np.asarray(
                [
                    np.linalg.norm(
                        states[:, :2]
                        - np.array([self.obs_sprite[i].x, self.obs_sprite[i].y]),
                        axis=1,
                    )
                    for i in range(self.num_obstacles)
                ],
                dtype=np.float32,
            )

        # negative rewards for getting close to wall
        for i in range(self.num_obstacles):
            if states is not None:
                d = d2obs_lst[i]
            else:
                d = self.dist2obs[i]
            if d.any() < const:
                r_obs += b * np.power(c, d - const)

        # only consider minimum distance to wall
        # if self.dist2wall[:, 0] < const:
        #     r_obs += b * np.power(c, self.dist2wall[:, 0] - const)

        reward += r_obs
        print("Total step reward: ", self.n_step, reward)
        return reward

    def check_collision(self) -> bool:
        """Check for collisions.

        Returns
        -------
        collided : Boolean
            Whether the table has collided with the obstacles
        """
        # hit_list = pygame.sprite.spritecollide(self.table, self.done_list, False)
        hit_list = pygame.sprite.spritecollide(
            self.table, self.done_list, False, pygame.sprite.collide_mask
        )

        if any(hit_list):
            # if any(pygame.sprite.spritecollide(self.table, self.target_list, False)):
            if any(
                pygame.sprite.spritecollide(
                    self.table, [self.target], False, pygame.sprite.collide_mask
                )
            ):
                self.success = True
                # reward = 80.
                debug_print("HIT TARGET")
            else:
                # reward = -100.
                debug_print("HIT OBSTACLE")
            return True  # , reward
        else:
            # wall collision
            if not self.screen.get_rect().contains(self.table):
                debug_print("HIT WALL")
                return True
            elif self.table.rect.left < 0:
                debug_print("HIT LEFT WALL")
                return True
            elif self.table.rect.right > WINDOW_W:
                debug_print("HIT RIGHT WALL")
                return True
            elif self.table.rect.top < 0:
                debug_print("HIT TOP WALL")
                return True
            elif self.table.rect.bottom > WINDOW_H:
                debug_print("HIT BOTTOM WALL")
                return True
            else:
                return False  # , 0

    def reset(self) -> np.ndarray:
        """Reset the environment.

        Returns
        -------
        observation : np.ndarray, shape=(state_dim,)
            Observation. TODO: make this not return actions and dist2stuff.
        """

        debug_print("Reset episode.\n")

        self.cumulative_reward = 0  # episode's cumulative reward
        self.fluency = {
            "inter_f": [],
            "f_del": [],
            "h_idle": [],
            "r_idle": [],
            "conf": [],
        }

        self.init_env()

        self.ep += 1
        # print("ep:", self.ep)
        self.file_name = os.path.join(self.dirname, "ep_" + str(self.ep) + ".pkl")
        self.config_file_name = os.path.join(
            self.map_config_dir, "config_params-ep_" + str(self.ep) + ".npz"
        )
        self.file_name_fluency = os.path.join(
            self.dirname_fluency, "ep_" + str(self.ep) + ".npz"
        )
        if not os.path.exists(os.path.dirname(self.file_name_fluency)):
            os.makedirs(os.path.dirname(self.file_name_fluency))
        if not os.path.exists(os.path.dirname(self.file_name)):
            os.makedirs(os.path.dirname(self.file_name))
        if not os.path.exists(os.path.dirname(self.config_file_name)):
            os.makedirs(os.path.dirname(self.config_file_name))

        self.data = []

        return self.get_state()

    def mp_check_collision(self, state) -> bool:
        """Check for collisions.

        Returns
        -------
        collided : Boolean
            Whether the table has collided with the obstacles
        """
        # set table position
        self.table.x = state[0]
        self.table.y = state[1]
        self.table.angle = state[2]
        # update sprite
        self.table.image = pygame.transform.rotate(
            self.table.original_img, math.degrees(self.table.angle)
        )
        self.table.rect = self.table.image.get_rect(center=(self.table.x, self.table.y))
        self.table.mask = pygame.mask.from_surface(self.table.image)

        hit_list = pygame.sprite.spritecollide(
            self.table, self.done_list, False, pygame.sprite.collide_mask
        )

        if any(hit_list):
            if not any(
                pygame.sprite.spritecollide(
                    self.table, [self.target], False, pygame.sprite.collide_mask
                )
            ):
                return True
        return False

    @staticmethod
    def standardize(self, ins, mean, std):
        s = np.divide(np.subtract(ins, mean), std)
        return s

    @staticmethod
    def unstandardize(self, ins, mean, std):
        us = np.multiply(ins, std).add(mean)
        return us

    def update_prediction(self, pred):
        self.prediction = pred

    def draw_gt(self, gt):
        self.ground_truth_states = gt

    def draw_past_states(self, past_states):
        self.past_states = past_states

    def redraw(self) -> None:
        """Updates the pygame visualization."""
        self.screen.fill((BLACK))
        # Update table image
        self.sprite_list.draw(self.screen)
        if self.prediction is not None:
            pygame.draw.circle(
                self.screen, (0, 0, 255, 1), [self.table.x, self.table.y], 3
            )

            for p in range(len(self.prediction)):
                pygame.draw.circle(
                    self.screen, (0, 0, 255, 1), self.prediction[p][:2], 1
                )
        if self.ground_truth_states is not None:
            for p in range(len(self.ground_truth_states)):
                pygame.draw.circle(
                    self.screen, (0, 255, 0, 0.5), self.ground_truth_states[p][:2], 2
                )
        if self.past_states is not None:
            for p in range(len(self.past_states)):
                pygame.draw.circle(
                    self.screen, (255, 255, 0, 0.5), self.past_states[p][:2], 2
                )

        pygame.display.update()

    def render(self, mode: str = "human") -> Union[np.ndarray, None]:
        """Renders an image.

        Parameters
        ----------
        mode : str, default="human"
            Render modes. Can be "human" or "rgb_array".

        Returns
        -------
        output : Union[np.ndarray, None]
            If mode is "human", then return nothing and update the viewer.
            If mode is "rgb_array", then return an image as np array to be rendered.
        """
        img = self.get_image()

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((WINDOW_W, WINDOW_H))
        if self.n_step % 10 == 0:
            img.save(os.path.join(self.dirname_vis_ep, str(self.n_step) + ".png"))

        if mode == "human":
            pass
            # if self.viewer is None:
            #     self.viewer = rendering.SimpleImageViewer()
            #     self.viewer.imshow(img)
        elif mode == "rgb_array":
            return img
        else:
            raise NotImplementedError

    @staticmethod
    def get_image() -> np.ndarray:
        """Gets the pygame display img for render.

        Returns
        -------
        img : np.ndarray, shape=(H, W, 3)
            3-channel image of the environment.
        """
        # observation
        img = np.fliplr(
            np.flip(
                np.rot90(
                    pygame.surfarray.array3d(pygame.display.get_surface()).astype(
                        np.uint8
                    )
                )
            )
        )
        return img

    def _get_rgb_state(self):
        """Gets the player state.

        Returns
        -------
        state : np.ndarray, shape=(STATE_H, STATE_W, 3)
            The state created after resizing the rendered image from pygame
        """
        img = self.get_image()
        img = Image.fromarray(img)
        img = img.resize((STATE_H, STATE_W))
        return np.asarray(img)

    def get_state(self) -> np.ndarray:
        if self.obs_type == "discrete":
            return self._get_discrete_state()
        elif self.obs_type == "rgb":
            return self._get_rgb_state()

    def _get_discrete_state(self) -> np.ndarray:
        """Gets the player state.

        Returns
        -------
        state : np.ndarray, shape=(state_dim, )
            The state.
        """
        state = np.zeros(shape=(self.state_dim,), dtype=np.float32)
        state[0] = self.table.x
        state[1] = self.table.y
        state[2] = np.cos(self.table.angle)  # self.table.angle  #
        state[3] = np.sin(self.table.angle)  # self.target.x  #
        state[4] = self.target.x
        state[5] = self.target.y  # self.dist2goal
        dist2obs = np.linalg.norm(self.avoid, axis=1)
        most_relevant_obs_idx = np.argmin(dist2obs)
        most_relevant_obs = self.obstacles[most_relevant_obs_idx]
        state[6] = most_relevant_obs[0]
        state[7] = most_relevant_obs[1]
        state[8] = self.table.angle

        return state
