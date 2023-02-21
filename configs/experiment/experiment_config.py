import argparse


def get_experiment_args(parser):
    parser.add_argument("--seed", type=int, default=88, help="Seed")

    # ------------------------ EXPERIMENT SETTINGS ------------------------
    parser.add_argument(
        "--exp-name",
        type=str,
        default="test_p1-planner_p2-human-keyboard",
        help="Name of experiment. Used to name saved trajs and plots.",
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        default="hil",
        help="Run mode. Options: [hil | replay_traj | coplanning]",
    )
    parser.add_argument(
        "--ep", type=int, default=0, help="Episode number to start from when recording data."
    )

    # ------------------------ ROBOT MODE ------------------------ #
    parser.add_argument(
        "--robot-mode",
        type=str,
        default="planner",
        help="Robot mode. Options: [planner | policy | data]",
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


    # ------------------------ REWARD FUNCTION SETTINGS ------------------------ #
    parser.add_argument(
        "--include_interaction_forces_in_rewards",
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
        "--data_dir",
        type=str,
        default="datasets/table-demos/table-demos_traj/test/test_holdout",
        help="Name of of dataset directory storing past trajectory rollouts; full path from base directory",
    )
    parser.add_argument(
        "--map_dir",
        type=str,
        default="datasets/table-demos/table-demos_map-cfg/map_cfg",
        help="Name of directory storing past trajectory rollouts' map configurations; full path from base directory",
    )
    parser.add_argument(
        "--map_config",
        type=str,
        default="cooperative_transport/gym_table/config/maps/rnd_obstacle_v2.yml",
        help="Map Config File Path -- stores possible map configurations for the environment, which you can define.",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Experiment Configuration.")