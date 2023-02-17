import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser("WorldVAE training & validation")
    parser.add_argument("--seed", type=int, default=88, help="Seed")

    # ------------------------ DIRECTORIES ------------------------
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument("--plot-dir", type=str, default="plots", help="Plots directory")
    parser.add_argument(
        "--data-base", type=str, default="datasets", help="Default data directory"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="final_dataset-test_0.05",
        help="Name of of dataset directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="vrnn",
        help="Name of experiment. Used to name saved model and plots.",
    )

    # ------------------------ RUN SETTINGS ------------------------
    # OPTIONS: restore the model (must include artifact-path), train the model, or test the model
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Indicate if loaded model is for training or not. If false, model is set to eval(), and you must specify artifact-path.",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        default=False,
        help="Restore model. If true, must specify artifact-path.",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default=None,
        help="Artifact directory storing model checkpoint to load. Must be located under `trained_models/` directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vrnn",
        help="Type of model to train",
    )

    # ------------------------ DATA ------------------------
    parser.add_argument(
        "--H", type=int, default=30, help="Observation period length, H."
    )
    parser.add_argument(
        "--SEQ-LEN",
        type=int,
        default=120,
        help="Total training sequence length, T. Observation period is H, and prediction period is T-H.",
    )
    parser.add_argument(
        "--skip", type=int, default=5, help="Frame skipping for training."
    )
    parser.add_argument(
        "--BSIZE", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Num workers for data loading.",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="test_holdout",
        help="specify which test dataset to use. Options: [test_holdout | unseen_map]",
    )
    parser.add_argument("--transform", type=str, default=None)

    # ------------------------ LOGGER ------------------------
    parser.add_argument(
        "--project",
        type=str,
        default="sbmp-hrcc",
        help="Name of project (for wandb logging).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name of run (for wandb logging).",
    )

    # ------------------------ TRAINING ------------------------
    parser.add_argument(
        "--include_actions",
        action="store_true",
        default=False,
        help="Whether to include actions in the model. Note: model from 2023 ICRA paper does not include actions.",
    )
    parser.add_argument(
        "--grad-clip-val",
        type=float,
        default=0.0,
        help="Smoothing constant for RMSProp Optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs to train for.",
    )

    # ------------------------ MODEL PARAMS ------------------------
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
        "--emb",
        type=int,
        default=32,
        help="Number of units for emnbedding layer.",
    )
    parser.add_argument(
        "--weight_init",
        type=str,
        default="xavier",
        help="Weight init for linear layers.",
    )

    # ------------------------ OPTIMIZER ------------------------
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="Smoothing constant for RMSProp Optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.2,
        help="Weight decay (L2 regularizer) for RMSProp Optimizer.",
    )

    # ------------------------ LR SCHEDULER ------------------------
    parser.add_argument(
        "--factor",
        type=float,
        default=0.7,
        help="Factor by which the learning rate will be reduced for LRScheduler.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="num epochs of no change before the learning rate will be reduced for LRScheduler.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0008, help="Training learning rate."
    )

    # ------------------------ BETA SCHEDULER ------------------------
    # Beta scheduling to balance KL loss and reconstruction loss training. Starts at beta_min, and increases to beta_max using cyclic annealing.
    # Parameters for Cyclic Annealing VAE (CAVAE) (https://arxiv.org/abs/1903.10145)
    parser.add_argument(
        "--cycle", type=int, default=4, help="Number of cycles for beta scheduler."
    )
    parser.add_argument(
        "--R",
        type=int,
        default=0.5,
        help="Proportion used to increase beta over a cycle.",
    )

    args = parser.parse_args()
    num_avail_gpu = torch.cuda.device_count()
    print("Number of GPUs available: ", num_avail_gpu)

    if num_avail_gpu > 0:
        args.device = "gpu"
    else:
        args.device = "cpu"

    return args
