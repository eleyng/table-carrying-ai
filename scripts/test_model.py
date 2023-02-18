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
