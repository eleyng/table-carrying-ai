import argparse


def get_experiment_args(parser):
    parser.add_argument(
        "--test-idx", type=int, default=0, help="Test index for experiment"
    )
    parser.add_argument("--seed", type=int, default=88, help="Seed")

    parser.add_argument(
        "--robot_name_1", type=str, default="locobot1", help="Name of robot 1"
    )
    parser.add_argument(
        "--robot_name_2", type=str, default="locobot2", help="Name of robot 2"
    )

    # ------------------------ EXPERIMENT SETTINGS ------------------------
    parser.add_argument(
        "--exp-name",
        type=str,
        default="simple_map",
        help="Name of experiment. Used to name saved trajs and plots.",
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        default="hil",
        help="Run mode. Options: [hil | replay_traj | coplanning]",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=0,
        help="Episode number to start from when recording data.",
    )

    # ------------------------ ROBOT MODE ------------------------ #
    parser.add_argument(
        "--robot-mode",
        type=str,
        default="planner",
        help="Robot mode. Options: [planner | policy | data]",
    )
    parser.add_argument(
        "--planner-type",
        type=str,
        default="vrnn",
        help="Planner type. Options: [vrnn | rrt | diffusion_policy].",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="diffusion",
        help="Policy type. Options: [diffusion | bc].",
    )

    # ------------------------ HUMAN MODE ------------------------ #
    parser.add_argument(
        "--human-mode",
        type=str,
        default="data",
        help="Human algorithm. Options: [real | data | policy (TODO: add policy)]. \
            If real or data, then must specify human control is keyboard or joystick.",
    )
    parser.add_argument(
        "--human-control",
        type=str,
        default="joystick",
        help="Human control. Options: [keyboard, joystick].",
    )
    parser.add_argument(
        "--subject-id",
        type=int,
        default=0,
        help="Subject ID. Used to load human data.",
    )

    # ------------------------ REWARD FUNCTION SETTINGS ------------------------ #
    parser.add_argument(
        "--include-interaction-forces-in-rewards",
        action="store_true",
        default=False,
        help="Include interaction forces in reward function.",
    )

    # ------------------------ DISPLAY SETTINGS ------------------------ #
    parser.add_argument(
        "--render-mode",
        type=str,
        default="gui",
        help="Render mode. Options: [gui | headless].",
    )
    parser.add_argument(
        "--add-buffer",
        action="store_true",
        default=False,
        help="Add buffer to display.",
    )
    parser.add_argument(
        "--display-pred",
        action="store_true",
        default=False,
        help="Display predicted trajectories in pygame window.",
    )
    parser.add_argument(
        "--display-gt",
        action="store_true",
        default=False,
        help="Display ground truth trajectories in pygame window.",
    )
    parser.add_argument(
        "--display-past-states",
        action="store_true",
        default=False,
        help="Display visited states as the game progesses in pygame window.",
    )

    # ------------------------ DIRECTORIES ------------------------ #
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument("--plot-dir", type=str, default="plots", help="Plots directory")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets/real_test/simple_map/np_trajectories", #unseen_map", #"datasets/table-demos/table-demos_traj/test/test_holdout", # dp_test/strat_0",
        help="Name of of dataset directory storing past trajectory rollouts; full path from base directory",
    )
    parser.add_argument(
        "--map-dir",
        type=str,
        default="datasets/real_test/simple_map/map_cfg", #unseen_map", #"datasets/table-demos/table-demos_map-cfg/map_cfg_rnd_obstacle_v2", #dp_test/strat_0/map_cfg",
        help="Name of directory storing past trajectory rollouts' map configurations; full path from base directory",
    )
    parser.add_argument(
        "--map-config",
        type=str,
        default="datasets/real_test/real_test.yml", #3.yml", #"cooperative_transport/gym_table/config/maps/rnd_obstacle_v2.yml",
        help="Map Config File Path -- stores possible map configurations for the environment, which you can define.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Experiment Configuration.")
