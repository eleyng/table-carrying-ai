import random
import argparse
import torch
import gym
import time
import pygame
import numpy as np
import os
from os.path import join, exists, isdir, isfile
from os import mkdir, listdir

import sys

# NOTE: Make sure to add the path to the cooperative-planner repo
# sys.path.append("algo/planners")
from algo.planners.cooperative_planner.models import VRNN

from cooperative_transport.gym_table.envs.utils import load_cfg, init_joystick, debug_print, FPS, CONST_DT, MAX_FRAMESKIP, get_keys_to_action
from cooperative_transport.gym_table.envs.custom_rewards import custom_reward_function
from libs.planner.planner_utils import (
    pid_single_step,
)
from libs.utils import play_hil_planner


# Import experiment configs
from configs.experiment.experiment_config import get_experiment_args
from configs.robot.robot_planner_config import get_planner_args
# from configs.robot.robot_policy_config import get_policy_args
from configs.human.human_config import get_human_args


VERBOSE = False # Set to True to print debug info




def play_traj(
    env,
    mode,
    model=None,
    optimizer=None,
    train_stats=None,
    SEQ_LEN=60,
    H=15,
    playback_trajectory=None,
    n_steps=1000,
    zoom=1,
    fps=30,

):

    # record trajectory played
    trajectory = {}
    trajectory["states"] = []
    trajectory["actions"] = []
    trajectory["rewards"] = []

    # reset environment
    obs = env.reset()
    done = False
    running = True
    n_iter = 0
    success = False

    # If playback trajectory is provided as the EGO actions (player 1's), use that instead of ego model
    if mode == "playback_trajectory" or mode == "data_aug":
        actions = playback_trajectory["actions"]
        waypoints = playback_trajectory["states"]

    next_game_tick = time.time()
    clock = pygame.time.Clock()

    # Main Game loop
    while running:
        loops = 0

        if done:
            time.sleep(1)
            pygame.quit()
            break
        else:
            while time.time() > next_game_tick and loops < MAX_FRAMESKIP and not done:
                if mode == "playback_trajectory":
                    if True:  # n_iter < actions.shape[0]:
                        n_iter = min(n_iter, actions.shape[0] - 1)
                        # pdb.set_trace()
                        waypoint = waypoints[n_iter, :4]
                        u_h = actions[n_iter, 2:]
                        # debug_print("waypoint: ", waypoint)
                        pid_actions = pid_single_step(
                            env,
                            waypoint,
                            kp=0.2,
                            ki=0.0,
                            kd=0.0,
                            max_iter=40,
                            dt=CONST_DT,
                            eps=1e-2,
                            linear_speed_limit=[-2.0, 2.0],
                            angular_speed_limit=[-np.pi / 4, np.pi / 4],
                            u_h=u_h,
                        )
                        pid_actions = np.clip(pid_actions, -1.0, 1.0)
                        debug_print(
                            "actual robot action: ",
                            actions[n_iter, :2],
                            "pid action: ",
                            pid_actions,
                            "human action: ",
                            actions[n_iter, 2:],
                        )

                        u_all = [
                            pid_actions[0],
                            pid_actions[1],
                            actions[n_iter, 2],
                            actions[n_iter, 3],
                        ]

                        debug_print("sending actions", u_all)

                        # obs, reward, done, info = env.step(actions[n_iter, :])

                        obs, reward, done, info = env.step(u_all)
                        if done:
                            running = False
                            break
                        trajectory["actions"].append(pid_actions)
                        trajectory["states"].append(obs)

                        # mse = ((actions[n_iter, :] - u_all) ** 2).mean(axis=0)
                        # debug_print("MSE: ", mse)

                # iterate to next step in game loop
                debug_print("Loop: ", loops, "n_iter", n_iter, info)
                next_game_tick += CONST_DT
                loops += 1
            n_iter += 1

            # UPDATE DISPLAY
            if not env.done:
                env.redraw()
                ##if callback is not None:
                ##    callback(prev_obs, obs, action, rew, env_done, info)
                # CLOCK TICK
                clock.tick(FPS)
                if clock.get_fps() > 0:
                    debug_print("Reported dt: ", 1 / clock.get_fps())

                # process pygame events
                for event in pygame.event.get():
                    # test events, set key states
                    if event.type == pygame.KEYDOWN:
                        if event.key == 27:
                            running = False
                    elif event.type == pygame.QUIT:
                        running = False

    # Save trajectory
    trajectory["states"] = np.stack(trajectory["states"], axis=0)
    trajectory["actions"] = np.stack(trajectory["actions"], axis=0)

    return trajectory, success


def main(sysargv):

    exp_args = get_experiment_args()
    print("Begin experiment in {} mode!".format(exp_args.run_mode))
    print("Experiment name: ", exp_args.exp_name)

    SEED = exp_args.seed
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")
    
    # ------------------------ Robot Options ------------------------
    # Comes from `--robot-mode` flag. Robot can be [planner | policy]
    model = None
    if sysargv[2] == "planner":
        args_r = get_planner_args()
        model = VRNN.load_from_checkpoint(
            args_r.artifact_path,
            # hparams=args_r,
        )
    elif sysargv[2] == "policy":
        pass
        # args_r = get_policy_args()
        # TODO: Add BC policy
        # model = VRNN.load_from_checkpoint(
        #     args_r.artifact_path,
        #     hparams=args_r,
        # )
    else:
        assert model is not None, "Robot type not supported"
    print("Robot {0} loaded from {1}: ".format(sysargv[2], args_r.artifact_path))
    model.eval()
    
    # ------------------------ Human Options ------------------------
    # Human play can be interactive, or come from data.
    print("Human actions from {0}: ".format(exp_args.human_control))
    
    # ------------------------ Directories Setup ------------------------
    # Setup directories
    if not isdir(exp_args.results_dir):
        mkdir(exp_args.results_dir)
    save_dir = join(exp_args.results_dir, exp_args.exp_name)
    if not isdir(save_dir):
        mkdir(save_dir)

    # Grab all GT traj files
    FILES = [
        join(exp_args.data_dir, sd, ssd)
        for sd in listdir(exp_args.data_dir)
        if isdir(join(exp_args.data_dir, sd))
        for ssd in listdir(join(exp_args.data_dir, sd))
    ]
    # Grab all GT map files
    MAP_FILES = [
        join(exp_args.map_dir, sd) 
        for sd in listdir(exp_args.map_dir,)
        if isfile(join(exp_args.map_dir, sd))
    ]

    # ------------------------ Experiment Setup ------------------------

    for f_idx in range(len(FILES)):
        f = FILES[f_idx]
        game_str = f.split("/")
        ep = game_str[-1]
        match = [map for map in MAP_FILES if ep == map.split("/")[-1]]

        env = gym.make(
            "cooperative_transport.gym_table:table-v0",
            render_mode="gui",
            control=exp_args.human_control,
            map_config=exp_args.map_config,
            run_mode="eval",
            load_map=match[0],
            run_name=exp_args.run_name,
            ep=exp_args.ep,
            dt=CONST_DT,
        )

        if exp_args.run_mode ==  "hil":

            trajectory, success, n_iter, duration = play_hil_planner(
                env,
                human=exp_args.human_mode,
                robot=exp_args.robot_mode,
                model=model,
                mcfg=args_r,
                SEQ_LEN=args_r.SEQ_LEN,
                H=args_r.H,
                playback_trajectory=dict(np.load(f)),
                display_pred=exp_args.display_pred,
                display_gt=exp_args.display_gt,
                display_past_states=exp_args.display_past_states,
            )
            
            print("Run finished. Task succeeded: {0}. Num steps taken in env: {1}. Episode {2}.".format(success, n_iter, ep))

            save_f = "eval_" + exp_args.run_mode + + "_seed-" + str(exp_args.seed) + "_R-" + \
                exp_args.robot_mode + "_H-" + exp_args.human_mode + "_EP-" + str(exp_args.ep) + ".npz"

            np.savez(
                join(exp_args.save_dir, save_f),
                states=trajectory["states"],
                plan=trajectory["plan"],
                actions=trajectory["actions"],
                rewards=trajectory["rewards"],
                fluency=trajectory["fluency"],
                done=trajectory["done"],
                success=trajectory["success"],
                n_iter=trajectory["n_iter"],
                duration=trajectory["duration"],
            )

        else:

            trajectory, success, n_iter, duration = play_hil_planner(
                env,
                human="data",
                robot="data",
                model=model,
                mcfg=args_r,
                SEQ_LEN=args_r.SEQ_LEN,
                H=args_r.H,
                playback_trajectory=dict(np.load(f)),
                display_pred=exp_args.display_pred,
                display_gt=exp_args.display_gt,
                display_past_states=exp_args.display_past_states,
            )

            print("Run finished. Task succeeded: {0}. Num steps taken in env: {1}. Episode {2}.".format(success, n_iter, ep))

        ep += 1

if __name__ == "__main__":

    main(sys.argv)
