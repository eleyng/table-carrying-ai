import argparse


def get_planner_args(parser):
    
    # ------------------------ PLANNER SAVED MODEL SETTINGS ------------------------
    # Restore the planner model. Make sure model parameters match.

    parser.add_argument(
        "--artifact-path",
        type=str,
        default="algo/planners/cooperative_planner/trained_models/vrnn_noact/model.ckpt",
        help="Path to model checkpoint to load.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vrnn",
        help="Type of model to train",
    )
    parser.add_argument(
        "--H", type=int, default=30, help="Observation period length, H."
    )
    parser.add_argument(
        "--SEQ-LEN",
        type=int,
        default=120,
        help="Total sequence length, T. Observation period is H, and prediction period is T-H.",
    )
    parser.add_argument(
        "--BSIZE", type=int, default=16, help="Batch size for predictions."
    )
    parser.add_argument("--skip", type=int, default=5, help="Frame skipping.")
    parser.add_argument("--lookahead", type=int, default=1, help="N multiplier on the number of skip frames to plan ahead for.")
    parser.add_argument(
        "--n-layers",
        type=int,
        default=2,
        help="Number of layers for any sequence model used (gru, rnn, etc.)",
    )
    parser.add_argument(
        "--ASIZE", type=int, default=4, help="Dimension of action space."
    )
    parser.add_argument(
        "--LSIZE", type=int, default=8, help="Dimension of state space."
    )
    parser.add_argument(
        "--NLAT", type=int, default=6, help="Dimension of latent space."
    )
    parser.add_argument(
        "--RSIZE",
        type=int,
        default=64,
        help="Number of hidden units for any sequence model used (gru, rnn, etc.)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Must be False during eval",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        default=True,
        help="Restore model. Must be True during eval.",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="VRNN Planner Configuration.")