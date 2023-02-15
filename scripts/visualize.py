import os
import time
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import numpy as np

from numpy.linalg import norm
from os.path import join, exists, dirname, abspath
from os import mkdir
import sys
from utils import (
    BLACK,
    WHITE,
    LIGHTBLUE,
    BLUE,
    LIGHTORANGE,
    ORANGE,
    GREEN,
    WINDOW_W,
    WINDOW_H,
    STATE_W,
    STATE_H,
    rad,
    load_cfg,
)

import pdb

FPS = 30
CONST_DT = 1 / FPS
MAX_FRAMESKIP = 10  # Min Render FPS = FPS / max_frameskip, i.e. framerate can drop until min render FPS
yaml_filepath = join(dirname(__file__), "../config/inference_params.yml")

cfg = load_cfg(yaml_filepath)
control_type = cfg["control_type"]
ep = cfg["ep"]
data_base = cfg["data_base"]
mode = cfg["mode"]
run_mode = cfg["run_mode"]
strategy_name = cfg["strategy_name"]
map_cfg = cfg["map_cfg"]
print(map_cfg)
experiment_name = cfg["experiment_name"]
print(experiment_name)
train_stats_f = cfg["train_stats_f"]
data_dir = cfg["data_dir"]


# synthetic data
map_name = map_cfg.split("/")[-1].split(".")[0]
if not exists(os.path.join(os.path.dirname(__file__), run_mode)):
    print("Making base directories.")
    mkdir(os.path.join(os.path.dirname(__file__), run_mode))
    mkdir(os.path.join(os.path.dirname(__file__), run_mode, map_name))
base_dirname = os.path.join(
    os.path.dirname(__file__), run_mode, map_name, strategy_name
)
dirname = os.path.join(
    os.path.dirname(__file__), run_mode, map_name, strategy_name, "trajectories"
)  # "runs/two-player-bc") #"../results/one-player-bc")
dirname_fluency = os.path.join(
    os.path.dirname(__file__), run_mode, map_name, strategy_name, "fluency"
)  # "runs/two-player-bc") #"../results/one-player-bc")
dirname_vis = os.path.join(
    os.path.dirname(__file__), run_mode, map_name, strategy_name, "figures"
)  # "runs/two-player-bc") #"../results/one-player-bc")
dirname_vis_ep = os.path.join(dirname_vis, "ep_" + str(ep) + "_images")
if not exists(base_dirname):
    mkdir(base_dirname)
    mkdir(dirname)
    mkdir(dirname_fluency)
    mkdir(dirname_vis)
print("Saving to directory: ", base_dirname)


# DIR = dirname_vis_ep
# trajectory_len = len(
#     [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
# )
# print(trajectory_len)

# img = cv.imread(os.path.join(dirname_vis_ep, "step-" + str(1) + ".jpg"))
# img_a = img
# count = 0
# for n_step in range(trajectory_len):
#     if os.path.isfile(os.path.join(dirname_vis_ep, "step-" + str(n_step) + ".jpg")):
#         if n_step % 5 == 0:
#             img_b = cv.imread(
#                 os.path.join(dirname_vis_ep, "step-" + str(n_step) + ".jpg")
#             )
#             a = 0.2
#             b = 1.0 - a
#             img_c = cv.addWeighted(img_b, b, img_a, a, 0)
#             cv.imwrite("combined.png", img_c)
#             img_a = cv.imread("combined.png")
#             print("overlay")

#     else:
#         print("skipping")
#         continue

# cv.imwrite("combined.png", img_c)
# cv.imshow("img", img_c)
# cv.waitKey(0)
# cv.destroyAllWindows()


def sort_files(img_dir):
    sorted_files = sorted(
        os.listdir(img_dir), key=lambda x: int(os.path.splitext(x)[0])
    )
    return sorted_files


def convert_to_transparent(img_dir):
    for i, fn in enumerate(sort_files(img_dir)):
        if i == 0 or i == (len(os.listdir(img_dir)) - 1):
            img = cv.imread(os.path.join(img_dir, fn), 1)
            img = cv.cvtColor(img, cv.COLOR_RGB2RGBA)
            cv.imwrite(os.path.join(img_dir, fn), img)
            continue
        print(os.path.join(img_dir, fn))
        img = cv.imread(os.path.join(img_dir, fn), 1)
        black_mask = np.all(img == 0, axis=-1)
        alpha = np.uint8(np.logical_not(black_mask)) * 255
        img = np.dstack((img, alpha))
        cv.imwrite(os.path.join(img_dir, fn), img)

        # img = cv.imread(os.path.join(img_dir, fn))
        # img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
        # # Slice of alpha channel
        # alpha = img[:, :, 3]
        # alpha[np.all(img[:, :, 0:3] == (0, 0, 0), 2)] = 0
        # cv.imwrite(os.path.join(img_dir, fn), img)


def create_overlay(img_dir, alpha=0.3, skip_frames=1):
    result = None

    sorted_files = sort_files(img_dir)
    print(sorted_files)

    for i, fn in enumerate(sorted_files[::skip_frames]):
        img = Image.open(os.path.join(img_dir, fn))  #
        if result is None:
            result = img
        else:
            pass
    x, y = img.size
    orig_img = Image.open(os.path.join(img_dir, sorted_files[0]))
    result = orig_img
    last_img = Image.open(os.path.join(img_dir, sorted_files[-1]))  # sorted_files
    print(os.path.join(img_dir, sorted_files[-1]))
    result.paste(last_img, (x, y))
    result = Image.blend(result, last_img, alpha=0.4)
    result.save("combined1.png")
    result = Image.open(os.path.join("combined1.png"))
    result.show()

    result2 = None
    for i, fn in enumerate(sorted_files[1:-1:skip_frames]):
        print(i, os.path.join(img_dir, fn))
        img = Image.open(os.path.join(img_dir, fn))
        if result2 is None:
            result2 = img
        else:
            result2.paste(img, mask=img)
            # result.paste(img, (x, y))
            # print("alpha", alpha)
            result2 = Image.blend(result2, img, alpha=alpha)
        # img.show()
    last_img = Image.open(os.path.join(img_dir, sorted_files[-1]))
    result2.paste(last_img, (x, y))
    result2 = Image.blend(result2, last_img, alpha=0.4)
    result2.save("combined2.png")
    result2 = Image.open(os.path.join("combined2.png"))
    # result.show()

    result.paste(result2, (x, y))
    result = Image.blend(result, result2, alpha=0.5)

    result.save("combined.png")


def add_alpha_channel(img_dir, fn):
    img = cv.imread(os.path.join(img_dir, fn), 1)
    img = cv.cvtColor(img, cv.COLOR_RGB2RGBA)
    cv.imwrite(os.path.join(img_dir, fn), img)


def convert_to_transparent(img_dir):
    for i, fn in enumerate(sort_files(img_dir)):
        if i == 0 or i == (len(os.listdir(img_dir)) - 1):
            add_alpha_channel(img_dir=img_dir, fn=fn)
            continue
        print(os.path.join(img_dir, fn))
        img = cv.imread(os.path.join(img_dir, fn), 1)
        black_mask = np.all(img == 0, axis=-1)
        alpha = np.uint8(np.logical_not(black_mask)) * 255
        img = np.dstack((img, alpha))
        cv.imwrite(os.path.join(img_dir, fn), img)

        # img = cv.imread(os.path.join(img_dir, fn))
        # img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
        # # Slice of alpha channel
        # alpha = img[:, :, 3]
        # alpha[np.all(img[:, :, 0:3] == (0, 0, 0), 2)] = 0
        # cv.imwrite(os.path.join(img_dir, fn), img)


def overlay_pred(img_dir, alpha=0.3, traj_file=None):

    table = Image.open("combined.png")
    table.show()
    traj = Image.open(traj_file)
    traj.show()
    add_alpha_channel(img_dir="", fn=traj_file)
    traj.show()

    x, y = table.size
    xp, yp = traj.size
    print(x, y, xp, yp)
    table.paste(traj, (x, y))
    result = Image.blend(table, traj, alpha=0.4)
    result.save("traj_overlay.png")
    result.show()


if __name__ == "__main__":
    traj_file = "batch-0-one_obstacle-traj.png"
    img_dir = dirname_vis_ep  # Input image dir, created by Blender
    alpha = 0.2  # Transparency (between 0-1; 0 less and 1 is more opacity)
    skip_frames = 20  # Overlay every skip_frames
    # convert_to_transparent(img_dir)
    # create_overlay(img_dir, alpha=alpha, skip_frames=skip_frames)
    overlay_pred(img_dir, alpha=alpha, traj_file=traj_file)
