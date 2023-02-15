import sys

sys.path.append("/home/armlab/table-carrying-sim/scripts")
# import pdb
import random
import argparse
import torch
import yaml
import gym
import time
import copy
import pygame
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numpy.linalg import norm
from os.path import join, exists, dirname, abspath, isdir
from os import mkdir, listdir
import sys
from cooperative_transport.gym_table.envs.utils import load_cfg

sys.path.append("/home/armlab/cooperative-world-models/models/")
from models.mdrnn import MDRNN
from models.bcrnn import BCRNN
from models.simple_linear import SimpleLinear
from models.mlp_rnn_gmm import MLPRNNGMM
from models.vae import VAE
from models.cvae import CVAE
from models.vrnn import VRNN

# from models.vrnn_gmm import VRNNGMM
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = torch.device("cpu")

# get metadata from config file
yaml_filepath = join("cooperative_transport/gym_table/config/inference_params.yml")
meta_cfg = load_cfg(yaml_filepath)
data_base = meta_cfg["data_base"]
dataset = meta_cfg["dataset"]
trained_model = meta_cfg["trained_model"]
save_dir = meta_cfg["save_dir"]
save_dir = save_dir + "-" + dataset
map_cfg = (
    "cooperative_transport/gym_table/config/maps/" + meta_cfg["map_config"] + ".yml"
)  # path to map config file


def main(config):
    # ------------------------
    # Configure experiment
    # ------------------------
    # Modes: data_aug, playback_trajectory, hil

    root = join("datasets", dataset)

    FILES = [
        join(root, sd, ssd, sssd)
        for sd in listdir(root)
        if isdir(join(root, sd))
        for ssd in listdir(join(root, sd))
        if isdir(join(root, sd))
        for sssd in listdir(join(root, sd, ssd))
    ]
    if args.run_mode == "hil":
        model = VRNN(args)
        # elif args.model == "vrnn_gmm":
        #     model = VRNNGMM(args)

    map_root = join(
        "cooperative_transport/gym_table/envs/demo",
        "rnd_obstacle_v2",
        args.strategy,
        "map_cfg",
    )

    map_files = [
        join(map_root, ssd) for ssd in listdir(map_root) if isfile(join(map_root, ssd))
    ]

    debug_print(args.run_mode)
    if not isdir(save_dir):
        mkdir(save_dir)

    # ------------------------
    # 1 LOAD TRAINING DATA STATS
    # ------------------------
    # model

    print("config", config)
    print("dataset: ", join(config.data_base, config.data_dir))

    if config.model == "bcrnn":
        model = BCRNN(config)
    elif config.model == "mdrnn":
        model = MDRNN(config)
    elif config.model == "linear":
        model = SimpleLinear(config)
    elif config.model == "vrnn":
        model = VRNN(config)
    # elif config.model == "vrnn_gmm":
    #     model = VRNNGMM(config)
    else:
        raise ValueError("Model not supported")

    num_avail_gpu = torch.cuda.device_count()
    print("Number of GPUs available: ", num_avail_gpu)

    # ------------------------
    # 2 RESTORE MODEL
    # ------------------------
    # restoring - reference can be retrieved in artifacts panel
    artifact_dir = config.artifact_dir
    print("Artifact downloaded to: ", artifact_dir)
    model = VRNN.load_from_checkpoint(artifact_dir + "/model.ckpt", hparams=config)
    model.eval()

    with torch.no_grad():
        # load test data

        states = torch.randn(1, config.SEQ_LEN, config.LSIZE)
        actions = torch.randn(1, config.SEQ_LEN, config.ASIZE)
        # mean_s = mean_s.to(config.device)
        # std_s = std_s.to(config.device)
        # print("state", state.shape)
        # print("sample", sample.shape)

        states = model.unstandardize(states, model.mean_s, model.std_s)
        with torch.no_grad():
            y_hat = model(states, actions)
            print("y_hat", y_hat)


# Test to run model on a seen map using rrt to control both


if __name__ == "__main__":

    parser = argparse.ArgumentParser("WorldVAE training & validation")
    parser.add_argument(
        "--entity",
        type=str,
        default="hrcc",
        help="Username for wandb",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="exp_dir",
        help="Where things are logged and models are loaded from.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="sbmp-hrcc",
        help="Name of project (for wandb logging).",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of run (for wandb logging).",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="ID of run for resuming training. Just enter the id, without the model name and the version number.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=0,
        help="Version of model for resuming training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Num workers for data loading.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vrnn",
        help="Type of model to train",
    )
    parser.add_argument(
        "--control_type",
        type=str,
        default="continuous",
        help="Type of control: 'continuous' or 'discrete'",
    )
    parser.add_argument(
        "--include_actions",
        type=str,
        default=None,
        help="Whether to include actions in the model",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default="25",
        help="Number of action bins (if discrete actions).",
    )
    parser.add_argument(
        "--data_base",
        type=str,
        default=data_base,  # "/arm/u/eleyng",
        help="Root/base directory of data folder.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Name of of dataset directory.",
    )
    parser.add_argument(
        "--map_cfg_f",
        type=str,
        default="one_obstacle.yml",
        help="File containing map config parameters.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of layers for any sequence model used (gru, rnn, etc.)",
    )
    parser.add_argument(
        "--ASIZE", type=int, default=4, help="Dimension of action space."
    )
    parser.add_argument(
        "--LSIZE", type=int, default=8, help="Dimension of latent space."
    )
    parser.add_argument(
        "--ydim",
        type=int,
        default=14,
        help="Dimension of conditioning variable for cvae",
    )
    parser.add_argument(
        "--RSIZE",
        type=int,
        default=128,
        help="Number of hidden units for any sequence model used (gru, rnn, etc.)",
    )
    parser.add_argument(
        "--emb",
        type=int,
        default=32,
        help="Number of units for emnbedding layer.",
    )
    parser.add_argument(
        "--grad_clip_val",
        type=float,
        default=0.0,
        help="Smoothing constant for RMSProp Optimizer.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Smoothing constant for RMSProp Optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.2,
        help="Weight decay (L2 regularizer) for RMSProp Optimizer.",
    )
    parser.add_argument(
        "--weight_init",
        type=str,
        default="xavier",  # xavier, or default gauss/uniform (None)
        help="Weight init for linear layers.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.8,
        help="Factor by which the learning rate will be reduced for LRScheduler.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="num epochs of no change before the learning rate will be reduced for LRScheduler.",
    )
    parser.add_argument(
        "--BSIZE", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument("--recon", type=str, default="l2", help="loss")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Training learning rate."
    )
    parser.add_argument(
        "--NGAUSS", type=int, default=4, help="Number of State Gaussians in GMM."
    )
    parser.add_argument(
        "--AGAUSS", type=int, default=2, help="Number of Action Gaussians in GMM."
    )
    parser.add_argument(
        "--SEQ_LEN",
        type=int,
        default=60,
        help="Total training sequence length (autoregressive pred len=SEQLEN-H).",
    )
    parser.add_argument("--H", type=int, default=15, help="Observation period length.")
    parser.add_argument(
        "--record_every_k_epochs",
        type=int,
        default=10,
        help="Record/plot/track training for every k epochs.",
    )

    # Parameters for Cyclic Annealing VAE
    parser.add_argument("--cycle", type=int, default=4)
    parser.add_argument("--R", type=int, default=0.5)
    parser.add_argument("--recon_loss", type=str, default="nll")
    parser.add_argument("--opt", type=str, default="Adam")
    parser.add_argument("--transform", type=str, default="min-max")
    parser.add_argument("-results_dir", type=str, default="results")
    parser.add_argument("--plot_dir", type=str, default="plots")
    parser.add_argument("--restore", type=float, default=0)
    parser.add_argument("--sweep", type=float, default=0)

    args = parser.parse_args()

    main(args)
