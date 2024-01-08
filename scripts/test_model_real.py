import argparse
import pickle
import random
import sys
import time
from os import listdir, mkdir
from os.path import dirname, isdir, isfile, join

import gym
import numpy as np
import torch

# Import experiment configs
from configs.experiment.real_experiment_config import get_experiment_args
from configs.robot.robot_planner_config import get_planner_args
from cooperative_transport.gym_table.envs.utils import (CONST_DT, FPS,
                                                        WINDOW_H, WINDOW_W, load_cfg)


VERBOSE = False  # Set to True to print debug info
FPS = 30
CONST_DT = 1.0 / FPS


def main(exp_args, exp_name):
    SEED = exp_args.seed
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # ------------------------ Human Options ------------------------
    # Human play can be interactive, or come from data.
    print("Human actions from: {0}. ".format(exp_args.human_control))

    # ------------------------ Directories Setup ------------------------
    # Setup directories
    if not isdir(exp_args.results_dir):
        mkdir(exp_args.results_dir)
    save_dir = join(exp_args.results_dir, exp_name)
    print("Saving results to: {0}".format(save_dir))
    if not isdir(save_dir):
        mkdir(save_dir)

    # Grab all GT traj files
    FILES = [
        # join(exp_args.data_dir, sd, ssd)
        # for sd in listdir(exp_args.data_dir)
        # if isdir(join(exp_args.data_dir, sd))
        # for ssd in listdir(join(exp_args.data_dir, sd))
        join(exp_args.data_dir, sd)
        for sd in listdir(exp_args.data_dir)
    ]
    # Grab all GT map files
    MAP_FILES = [
        join(exp_args.map_dir, sd)
        for sd in listdir(
            exp_args.map_dir,
        )
        if isfile(join(exp_args.map_dir, sd))
    ]

    # ------------------------ Experiment Setup ------------------------
    # Parameters for sampling
    success_rate = 0
    success_duration = 0
    num_success = 0
    avg_plan_time_total = 0
    robot_control = exp_args.robot_mode

    # for f_idx in range(len(FILES)):
    #     # time.sleep(1)
    #     f = FILES[f_idx]
    #     game_str = f.split("/")
    #     ep = game_str[-1]
    #     robot_control = exp_args.robot_mode
    #     match = [map for map in MAP_FILES if ep in map.split("/")[-1]]

    table_cfgs = load_cfg(exp_args.map_config)["TABLE"]
    goal_cfgs = load_cfg(exp_args.map_config)["GOAL"]   
    print(exp_args.map_config)

    for i in range(4):

        if exp_args.planner_type == "cogail":
            seq_length = 15
            from cooperative_transport.gym_table.envs.table_env_cogail import \
                TableEnv
        else:
            seq_length = 1
            from cooperative_transport.gym_table.envs.table_env import TableEnv

        if exp_args.planner_type == "diffusion_policy":
            from libs.hil_real_robot_diffusion import play_hil_planner
        # elif exp_args.planner_type == "bc_lstm_gmm":
        #     from libs.hil_methods_bc_lstm_gmm import play_hil_planner
        else:
            from libs.hil_real_robot import play_hil_planner


        env = TableEnv(
            render_mode=exp_args.render_mode,
            control=exp_args.human_control,
            map_config=exp_args.map_config,
            run_mode="demo",
            run_name=exp_name,
            ep=0,
            dt=CONST_DT,
            seq_length=seq_length,
            set_obs=load_cfg(exp_args.map_config)["OBSTACLES"]["POSITIONS"],
            set_table=table_cfgs[i],
            set_goal=goal_cfgs[0]
        )

        # env = TableEnv(
        #     render_mode=exp_args.render_mode,
        #     control=exp_args.human_control,
        #     map_config=exp_args.map_config,
        #     run_mode="eval",
        #     load_map=match[0],
        #     run_name=exp_name,
        #     ep=exp_args.ep,
        #     dt=CONST_DT,
        #     seq_length=seq_length,
        #     add_buffer=exp_args.add_buffer,
        # )
        
        trajectory, success, n_iter, duration, avg_plan_time = play_hil_planner(
            env,
            exp_run_mode=exp_args.run_mode,
            human=exp_args.human_mode,
            robot=robot_control,
            planner_type=exp_args.planner_type,
            artifact_path=exp_args.artifact_path,
            mcfg=exp_args,
            SEQ_LEN=exp_args.SEQ_LEN,
            H=exp_args.H,
            skip=exp_args.skip,
            num_candidates=exp_args.BSIZE,
            # playback_trajectory=dict(np.load(f, allow_pickle=True)),
            collision_checking_env=None, #collision_checking_env,
            display_pred=exp_args.display_pred,
            display_gt=exp_args.display_gt,
            display_past_states=exp_args.display_past_states,
            include_interaction_forces_in_rewards=exp_args.include_interaction_forces_in_rewards,
            device=torch.device("cuda") if exp_args.planner_type == "diffusion_policy" else "cpu",
        )

        avg_plan_time_total += avg_plan_time

        print(
            "Run finished. Task succeeded: {0}. Duration: {1} sec. Num steps taken in env: {2}. Episode {3}. Avg plan time {4}".format(
                success, duration, n_iter, ep, avg_plan_time
            )
        )

        if not (exp_args.human_mode == "data" and exp_args.robot_mode == "data"):
            if exp_args.robot_mode == "planner":
                robot_str = exp_args.robot_mode + "-" + exp_args.planner_type
            else:
                robot_str = exp_args.robot_mode
            if exp_args.human_mode == "real":
                human_str = exp_args.human_mode + "-" + exp_args.human_control
            else:
                human_str = exp_args.human_mode

            save_f = exp_name + "-" + ep

            if exp_args.planner_type == "rrt":
                pickle.dump(trajectory, open(join(save_dir, save_f), "wb"))
            else:
                np.savez(
                    join(save_dir, save_f),
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

            success_rate += success
            num_success += 1 if success else 0
            success_duration += duration if success else 0
    success_rate /= len(FILES)
    success_duration /= num_success if num_success > 0 else 1
    avg_plan_time_total /= len(FILES)
    print("Success rate: {0}, with {1} total trials. Avg duration of {2} for {3} total successes. Avg plan time {4}.".format(success_rate, len(FILES), success_duration, num_success, avg_plan_time_total))
    with open(join(save_dir, "success_rate_" + str(exp_args.planner_type) + "_exp-" + str(exp_args.data_dir.split("/")[-1]) + "_hact-" + str(exp_args.human_act_as_cond) + ".txt"), "a") as f:
        f.write("{0}    Subject ID: {1}. Success rate: {2}, with {3} total trials. Avg duration of {4} for {5} total successes. Avg plan time {6}.\n".format(time.strftime("%c", time.localtime()), exp_args.subject_id, success_rate, len(FILES), success_duration, num_success, avg_plan_time_total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Table Carrying Experiments.")

    exp_args = parser.add_argument_group("Experiment Settings")
    get_experiment_args(exp_args)
    assert sys.argv[2] in [
        "replay_traj",
        "hil",
        "coplanning",
        "copolicy",
    ], "Run mode not supported"
    if sys.argv[2] == "hil":
        assert (
            sys.argv[6] != "data"
        ), "If --human-mode is 'data', --run-mode should be 'replay_traj', not 'hil'"

    # ------------------------ Robot Options ------------------------
    # Comes from `--robot-mode` flag. Robot can be [planner | policy]
    if sys.argv[4] == "planner" or sys.argv[6] == "planner" or sys.argv[2] == "replay_traj":
        get_planner_args(exp_args)
    elif sys.argv[4] == "policy" or sys.argv[6] == "policy":
        pass
        # get_policy_args(exp_args)
        # TODO: Add BC policy
    elif sys.argv[4] == "data" or sys.argv[6] == "data":
        assert sys.argv[2] == "replay_traj", "If --robot-mode or --human-mode is 'data', --run-mode should be 'replay_traj', not 'hil'"
    else:
        raise ValueError("Robot mode not supported")
    exp_args = parser.parse_args()

    # Check valid experiment modes
    """ If HIL mode, these are the possible combinations:
        - Human real + Robot planner
        - Human real + Robot policy
        - Human real + Robot data

        ** Notes: 
            - if human real, then indicate control type [keyboard | joystick].
            - human/robot policy is currently not supported   


        If replay_traj mode, these are the possible combinations:
        - Human data + Robot data
        - Human data + Robot planner
        - Human data + Robot policy

        If coplanning mode, these are the possible combinations:
        - Human planner + Robot planner
        
    """

    print("Begin experiment in {} mode!".format(exp_args.run_mode))

    if exp_args.run_mode == "hil":
        assert (
            not exp_args.human_mode == "planner"
        ), "Set --run_mode to coplanning if both robot and human is planner"
        assert not (
            exp_args.human_mode == "data"
        ), "HIL mode requires human behaviors not from data (i.e. not open-loop). Set --run_mode to replay_traj instead."

    if exp_args.robot_mode == "planner":
        robot_str = exp_args.robot_mode + "-" + exp_args.planner_type
    else:
        robot_str = exp_args.robot_mode
    human_str = exp_args.human_mode + "-" + exp_args.human_control
    

    exp_name = (
        "eval_"
        + exp_args.run_mode
        + "_seed-"
        + str(exp_args.seed)
        + "_R-"
        + robot_str
        + "_H-"
        + human_str
        + "_exp-"
        + str(exp_args.data_dir.split("/")[-1])
        + "_hact-"
        + str(exp_args.human_act_as_cond)
        + "_subject-"
        + str(exp_args.subject_id)
    )

    print("Experiment name: ", exp_name)

    
    main(exp_args, exp_name)
