import argparse
import random
import torch
import gym
import numpy as np
from os.path import join, isdir, isfile
from os import mkdir, listdir

import sys

# NOTE: Make sure to add the path to the cooperative-planner repo
# sys.path.append("algo/planners")
from algo.planners.cooperative_planner.models import VRNN

from cooperative_transport.gym_table.envs.utils import CONST_DT
from libs.utils import play_hil_planner


# Import experiment configs
from configs.experiment.experiment_config import get_experiment_args
from configs.robot.robot_planner_config import get_planner_args
# from configs.robot.robot_policy_config import get_policy_args


VERBOSE = False # Set to True to print debug info


def main(exp_args, exp_name):
    SEED = exp_args.seed
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")
    model.eval()
    
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
    # Parameters for sampling
    model.batch_size = exp_args.BSIZE
    model.skip = exp_args.skip

    for f_idx in range(len(FILES)):
        f = FILES[f_idx]
        game_str = f.split("/")
        ep = game_str[-1]
        match = [map for map in MAP_FILES if ep == map.split("/")[-1]]

        env = gym.make(
            "cooperative_transport.gym_table:table-v0",
            render_mode=exp_args.render_mode,
            control=exp_args.human_control,
            map_config=exp_args.map_config,
            run_mode="eval",
            load_map=match[0],
            run_name=exp_name,
            ep=exp_args.ep,
            dt=CONST_DT,
        )

        if exp_args.run_mode ==  "hil" or exp_args.run_mode == "coplanning":

            trajectory, success, n_iter, duration = play_hil_planner(
                env,
                exp_run_mode=exp_args.run_mode,
                human=exp_args.human_mode,
                robot=exp_args.robot_mode,
                model=model,
                mcfg=exp_args,
                SEQ_LEN=exp_args.SEQ_LEN,
                H=exp_args.H,
                playback_trajectory=dict(np.load(f)),
                display_pred=exp_args.display_pred,
                display_gt=exp_args.display_gt,
                display_past_states=exp_args.display_past_states,
            )
            
            print("Run finished. Task succeeded: {0}. Num steps taken in env: {1}. Episode {2}.".format(success, n_iter, ep))

            save_f = "eval_" + exp_args.run_mode + "_seed-" + str(exp_args.seed) + "_R-" + \
                exp_args.robot_mode + "_H-" + exp_args.human_mode + "_" + ep

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

        else:

            trajectory, success, n_iter, duration = play_hil_planner(
                env,
                exp_run_mode=exp_args.run_mode,
                human="data",
                robot="data",
                model=model,
                mcfg=exp_args,
                SEQ_LEN=exp_args.SEQ_LEN,
                H=exp_args.H,
                playback_trajectory=dict(np.load(f)),
                display_pred=exp_args.display_pred,
                display_gt=exp_args.display_gt,
                display_past_states=exp_args.display_past_states,
            )

            print("Run finished. Task succeeded: {0}. Num steps taken in env: {1}. Episode {2}.".format(success, n_iter, ep))


if __name__ == "__main__":  
    parser = argparse.ArgumentParser("Table Carrying Experiments.")

    exp_args = parser.add_argument_group("Experiment Settings")
    get_experiment_args(exp_args)
    assert sys.argv[2] in ["replay_traj", "hil", "coplanning"], "Run mode not supported"
    if sys.argv[2] == "replay_traj":
        assert sys.argv[4] == 'data' and sys.argv[6] == "data", "Replay traj mode requires --human-mode and --robot-mode to be data"

    # ------------------------ Robot Options ------------------------
    # Comes from `--robot-mode` flag. Robot can be [planner | policy]
    model = None
    if sys.argv[4] == "planner" or sys.argv[4] == "data":
        get_planner_args(exp_args)
        exp_args = parser.parse_args()
        model = VRNN.load_from_checkpoint(
            exp_args.artifact_path,
        )
    elif sys.argv[4] == "policy":
        pass
        # get_policy_args(exp_args)
        # exp_args = parser.parse_args()
        # TODO: Add BC policy
        # model = BCRNNGMM.load_from_checkpoint(
        #     exp_args.artifact_path,
        # )
    else:
        assert model is not None, "Robot type not supported"

    # Check valid experiment modes
    """ If HIL mode, these are the possible combinations:
        - Human real + Robot planner
        - Human real + Robot policy
        - Human real + Robot data
        - Human data + Robot planner
        - Human data + Robot policy
        ** Notes: 
            - if human real, then indicate control type [keyboard | joystick].
            - human/robot policy is currently not supported by this version    


        If replay_traj mode, these are the possible combinations:
        - Human data + Robot data
        
    """

    print("Begin experiment in {} mode!".format(exp_args.run_mode))

    if exp_args.run_mode == "hil":
        assert not exp_args.human_mode == "planner", "Set run mode to coplanning if both robot and human is planner"
        assert not (exp_args.robot_mode == "data" and exp_args.human_mode == "data"), "HIL mode require that both human and robot are not from data. Use replay_traj mode instead."
    
    exp_name = "eval_" + exp_args.run_mode + "_seed-" + str(exp_args.seed) + "_R-" + \
        exp_args.robot_mode + "_H-" + exp_args.human_mode + "_" + exp_args.human_control
    print("Experiment name: ", exp_name)

    print("Robot {0} loaded from {1}: ".format(exp_args.robot_mode, exp_args.artifact_path))  
    main(exp_args, exp_name)
