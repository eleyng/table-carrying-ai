import argparse
import numpy as np
from os import mkdir
from os.path import join, isfile, isdir
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from cooperative_transport.gym_table.envs.utils import (
    WINDOW_W,
    WINDOW_H,
    obstacle_size,
)

from libs.utils import (
    make_video,
)

def vis(args):
    """ Visualize a HIL or standard trajectory. 
        NOTE: You must have a ground truth demo trajectory to compare against, 
        to see potential deviations in behavior for the given 
        map config (initial pose, goal, and obstacle layout). 

        To elaborate, the workflow might go like this:
        1. Collect a trajectory using the HIL or running robot-robot co-policy.
            During this collection, a ground truth map config is loaded, so the 
            trajectory rollout is given the same map config as the ground truth.
        2. Run this script to visualize the trajectory rollout, and compare it 
            to the ground truth trajectory. Multimodal behavior might occur.
    """
    # ------------------------ Directories Setup ------------------------
    # Setup directories
    if not isdir("results"):
            mkdir("results")
    if not isdir(join("results", "plots")):
        mkdir(join("results", "plots"))

    # ------------------------ Load Trajectories ------------------------

    # load traj
    f = args.path_to_traj
    assert isfile(f), "Trajectory file not found in path specified: {0}".format(f)
    # load gt traj
    gt_f = args.path_to_gt
    assert isfile(gt_f), "Ground truth trajectory file not found in path specified: {0}".format(gt_f)
    # load map
    map_f = args.path_to_map
    assert isfile(map_f), "Map file not found in path specified: {0}".format(map_f)

    # ------------------------ Plotting Setup ------------------------

    hspace, vspace = (WINDOW_W / 100, WINDOW_H / 100)
    fig = plt.figure(figsize=(hspace, vspace), dpi=500)
    plt.rcParams["figure.figsize"] = (hspace, vspace)
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 2.5

    # load traj
    traj = dict(np.load(f, allow_pickle=True))["states"]
    gt = dict(np.load(gt_f, allow_pickle=True))["states"]
    # load map info
    map_run = dict(np.load(map_f, allow_pickle=True))
    # table initial pose
    table_init = np.zeros(2)
    table_init[0] = map_run["table"].item()["x"]
    table_init[1] = map_run["table"].item()["y"]
    # table goal pose
    table_goal = np.zeros(2)
    table_goal[0] = map_run["goal"].item()["goal"][0]
    table_goal[1] = map_run["goal"].item()["goal"][1]
    # table obstacles as encoding
    num_obs = map_run["obstacles"].item()["num_obstacles"]
    obs = np.zeros((num_obs, 2))
    obstacles = map_run["obstacles"].item()["obstacles"]

    # ------------------------ Plotting --------------------------------

    for t in range(1, gt.shape[0], args.skip):
        # plot map
        ca = plt.gca()
        ca.add_patch(
            patches.Circle(
                (table_init[0], table_init[1]),
                radius=obstacle_size,
                facecolor=(175 / 255, 175 / 255, 175 / 255, 1.0),  # black
                zorder=0,
            )
        )

        for i in range(obstacles.shape[0]):
            obstacle_w = obstacle_size
            obstacle_h = obstacle_size
            obstacle_x = obstacles[i, 0]  # - obstacle_w / 2.0
            obstacle_y = obstacles[i, 1]  # - obstacle_h / 2.0
            if obstacle_x == 0 or obstacle_y == 0:
                continue
            ca.add_patch(
                patches.Rectangle(
                    (obstacle_x - obstacle_w / 2, obstacle_y + obstacle_h / 2),
                    obstacle_w,
                    obstacle_h,
                    facecolor=(230 / 255, 111 / 255, 81 / 255, 1.0),
                    zorder=0,
                )
            )
        ca.add_patch(
            patches.Rectangle(
                (table_goal[0] - 200 / 2, table_goal[1] - 250 / 2),
                200,
                250,
                facecolor=(242 / 255, 220 / 255, 107 / 255, 1.0),  # gold
                zorder=0,
            )
        )

        plt.gca().set_aspect("equal") 
        plt.xlim([0, WINDOW_W])
        plt.ylim([0, WINDOW_H])
        plt.axis("off")

        H = args.H

        if t < H:

            init_x = gt[:t, 0]
            init_y = gt[:t, 1]

            plt.plot(init_x[0], init_y[0], "o", c="black", markersize=4)
            plt.scatter(init_x, init_y, c="black", s=2)
            plt.plot(
                gt[:t, 0],
                gt[:t, 1],
                c=(42 / 255, 157 / 255, 142 / 255),
                alpha=1.0,
                markersize=5,
                linewidth=3,
            )

        else:
            init_x = gt[:H, 0]
            init_y = gt[:H, 1]

            plt.plot(init_x[0], init_y[0], "o", c="black", markersize=4)
            plt.scatter(init_x, init_y, c="black", s=2)

            plt.plot(
                traj[H - 1 : t, 0],
                traj[H - 1 : t, 1],
                c=(243 / 255, 162 / 255, 97 / 255),
                alpha=1.0,
                markersize=5,
                linewidth=3,
            )

            plt.plot(
                gt[H - 1 : t, 0],
                gt[H - 1 : t, 1],
                c=(42 / 255, 157 / 255, 142 / 255),
                alpha=1.0,
                markersize=5,
                linewidth=3,
            )

        plot_dir = join(
                "results",
                "plots",
                args.path_to_traj.split("/")[-1].split(".")[0], 
            )
        if not isdir(plot_dir):
            mkdir(plot_dir)
        
        plot_name = join(
                "results",
                "plots",
                args.path_to_traj.split("/")[-1].split(".")[0], 
                "{0}".format(t),
            )

        if not isdir("traj_vis_plots"):
            mkdir("traj_vis_plots")

        plt.xlabel("xlabel", fontsize=18)
        plt.ylabel("ylabel", fontsize=16)
        plt.savefig(plot_name, dpi=500)
        plt.close()

    # ------------------------ Video --------------------------------

    if args.video:
        make_video(plot_dir, args.path_to_traj.split("/")[-1].split(".")[0])
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-traj",
        type=str,
        default="results/eval_hil_seed-88_R-planner_H-data/eval_hil_seed-88_R-planner_H-data_ep_373.npz",
        help="Path to trajectory file (.npz)",
    )
    parser.add_argument(
        "--path-to-gt",
        type=str,
        default="datasets/table-demos/table-demos_traj/test/test_holdout/373/ep_373.npz",
        help="Path to trajectory file (.npz)",
    )
    parser.add_argument(
        "--path-to-map",
        type=str,
        default="datasets/table-demos/table-demos_map-cfg/map_cfg/ep_373.npz",
        help="Path to map file (.npz)",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="demo",
        help="Define Run Mode, [demo | eval]. Demo: "
    )
    parser.add_argument(
        "--map_config",
        type=str,
        default="rnd_obstacle_v2",
        help="Map Config File Path",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="random_run_name",
        help="run_name name for data collection. creates a folder with this name in the repo's base directory to store data.",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=0,
        help="episode number of trajectory data.",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=30,
        help="Observation horizon length",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=120,
        help="Prediction horizon length, including observation horizon",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=10,
        help="skip frames",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="convert images to video",
    )
    args = parser.parse_args()
    vis(args)


if __name__ == "__main__":
    main()