import argparse


def get_planner_args():
    parser = argparse.ArgumentParser("Table Carrying Experiments Config")
    parser.add_argument("--seed", type=int, default=88, help="Seed")
    
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


    # ------------------------ DON'T CHANGE THESE SETTINGS FOR EVALUATION ------------------------
    
    # parser.add_argument(
    #     "--num-workers",
    #     type=int,
    #     default=1,
    #     help="Num workers for data loading.",
    # )
    # parser.add_argument(
    #     "--test-data",
    #     type=str,
    #     default="test_holdout",
    #     help="specify which test dataset to use. Options: [test_holdout | unseen_map]",
    # )
    # parser.add_argument(
    #     "--transform", type=str, help="Apply transforms to the data", default=None
    # )
    # parser.add_argument(
    #     "--project",
    #     type=str,
    #     default="sbmp-hrcc",
    #     help="Name of project (for wandb logging).",
    # )
    # parser.add_argument(
    #     "--experiment-name",
    #     type=str,
    #     default="test_hil_planner",
    #     help="Name of run (for wandb logging).",
    # )
    # parser.add_argument(
    #     "--include_actions",
    #     action="store_true",
    #     default=False,
    #     help="Whether to include actions in the model. Note: model from 2023 ICRA paper does not include actions.",
    # )
    # parser.add_argument(
    #     "--grad-clip-val",
    #     type=float,
    #     default=0.0,
    #     help="Smoothing constant for RMSProp Optimizer.",
    # )
    # parser.add_argument(
    #     "--epochs",
    #     type=int,
    #     default=200,
    #     help="Number of epochs to train for.",
    # )
    # parser.add_argument(
    #     "--n-layers",
    #     type=int,
    #     default=2,
    #     help="Number of layers for any sequence model used (gru, rnn, etc.)",
    # )
    # parser.add_argument(
    #     "--ASIZE", type=int, default=4, help="Dimension of action space."
    # )
    # parser.add_argument(
    #     "--LSIZE", type=int, default=8, help="Dimension of state space."
    # )
    # parser.add_argument(
    #     "--NLAT", type=int, default=6, help="Dimension of latent space."
    # )
    # parser.add_argument(
    #     "--RSIZE",
    #     type=int,
    #     default=64,
    #     help="Number of hidden units for any sequence model used (gru, rnn, etc.)",
    # )
    # parser.add_argument(
    #     "--emb",
    #     type=int,
    #     default=32,
    #     help="Number of units for emnbedding layer.",
    # )
    # parser.add_argument(
    #     "--weight_init",
    #     type=str,
    #     default="xavier",
    #     help="Weight init for linear layers.",
    # )

    
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

    args = parser.parse_args()
    args.device = "cpu"

    return args
