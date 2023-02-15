import pdb
import random
import argparse
import torch
import yaml
import gym
import time
import copy
import pygame
import numpy as np
from numpy.linalg import norm
import os
from os.path import join, exists, dirname, abspath, isdir, isfile
from os import mkdir, listdir
from stable_baselines3.common.env_checker import check_env

import sys

sys.path.append("/home/armlab/cooperative-world-models/models/")
from models.vrnn import VRNN

from cooperative_transport.gym_table.envs.utils import load_cfg, init_joystick

VERBOSE = False


def debug_print(*args):
    if not VERBOSE:
        return
    print(*args)


FPS = 30
CONST_DT = 1 / FPS
MAX_FRAMESKIP = 10  # Min Render FPS = FPS / max_frameskip, i.e. framerate can drop until min render FPS

# TODO: either move all this stuff into a util file as enum or expose it to be configurable
# table parameters
m = 2.0
b = 2.0
I = 1.0
L = 1.0
d = 40

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = torch.device("cpu")
debug_print("device: ", device)

# get metadata from config file
yaml_filepath = join("cooperative_transport/gym_table/config/inference_params.yml")
meta_cfg = load_cfg(yaml_filepath)
data_base = meta_cfg["data_base"]
dataset = meta_cfg["dataset"]
trained_model = meta_cfg["trained_model"]
save_dir = meta_cfg["save_dir"]
save_dir = save_dir
map_cfg = (
    "cooperative_transport/gym_table/config/maps/" + meta_cfg["map_config"] + ".yml"
)  # path to map config file


def get_action_from_wrench(wrench, current_state, u_h):
    # pdb.set_trace()
    new_wx = wrench[0] - u_h[0]
    new_wy = wrench[1] - u_h[1]
    new_wz = wrench[2] - L / 2.0 * (
        np.sin(current_state[2]) * u_h[0] + np.cos(current_state[2]) * u_h[1]
    )
    new_wrench = np.array([new_wx, new_wy, new_wz]).T
    G = (
        [1, 0],
        [0, 1],
        [
            -L / 2.0 * np.sin(current_state[2]),
            -L / 2.0 * np.cos(current_state[2]),
        ],
    )
    F_des = np.linalg.pinv(G).dot(new_wrench)
    debug_print("FDES: ", F_des)
    # f1_x, f1_y = F_des[0], F_des[1]

    return F_des  # in world frame


def get_joint_action_from_wrench(wrench, current_state):
    # pdb.set_trace()
    new_wx = wrench[0]
    new_wy = wrench[1]
    new_wz = wrench[2]
    new_wrench = np.array([new_wx, new_wy, new_wz]).T
    G = (
        [1, 0, 0],
        [0, 1, 0],
        [
            -L / 2.0 * np.sin(current_state[2]),
            -L / 2.0 * np.cos(current_state[2]),
            L / 2.0 * np.sin(current_state[2]),
            L / 2.0 * np.cos(current_state[2]),
        ],
    )
    F_des = np.linalg.pinv(G).dot(new_wrench)
    debug_print("FDES: ", F_des)
    # f1_x, f1_y = F_des[0], F_des[1]

    return F_des  # in world frame


def pid_single_step(
    env,
    waypoint,
    kp=0.5,
    ki=0.0,
    kd=0.0,
    max_iter=3,
    dt=CONST_DT,
    eps=1e-2,
    linear_speed_limit=[-2.0, 2.0],
    angular_speed_limit=[-np.pi / 8, np.pi / 8],
    u_h=None,
):
    ang = np.arctan2(waypoint[3], waypoint[2])

    # ang[ang < 0] += 2 * np.pi
    if ang < 0:
        ang += 2 * np.pi

    waypoint = np.array([waypoint[0], waypoint[1], ang])

    curr_target = waypoint

    curr_state = np.array([env.table.x, env.table.y, env.table.angle])
    error = curr_target - curr_state
    wrench = kp * error
    # Get actions from env
    if u_h is None:
        raise ValueError("u_h was never passed to pid.")
    F_des_r = get_action_from_wrench(wrench, curr_state, u_h)
    return F_des_r


def update_queue(a, x):
    return torch.cat([a[1:, :], x], dim=0)


def tf2model(state_data):
    # must remove theta obs (last dim of state_data)
    # takes observation (only map info) and returns ego-centric vector to obs/goal for use in model
    state_xy = state_data[:, :2]
    state_th = state_data[:, 2:4]
    state_data_ego_pose = np.diff(state_xy, axis=0)
    state_data_ego_th = np.diff(state_th, axis=0)
    goal_lst = np.empty(shape=(state_data_ego_pose.shape[0], 2), dtype=np.float32)
    obs_lst = np.empty(shape=(state_data_ego_pose.shape[0], 2), dtype=np.float32)

    for t in range(state_data_ego_pose.shape[0]):
        p_ego2obs_world = state_data[t, 6:8] - state_xy[t, :]
        # print("p_ego2obs_world", p_ego2obs_world.shape)
        p_ego2goal_world = state_data[t, 4:6] - state_xy[t, :]

        cth = np.cos(state_data[t, 8])
        sth = np.sin(state_data[t, 8])
        # goal & obs in ego frame
        obs_lst[t, 0] = cth * p_ego2obs_world[0] + sth * p_ego2obs_world[1]
        obs_lst[t, 1] = -sth * p_ego2obs_world[0] + cth * p_ego2obs_world[1]

        goal_lst[t, 0] = cth * p_ego2goal_world[0] + sth * p_ego2goal_world[1]
        goal_lst[t, 1] = -sth * p_ego2goal_world[0] + cth * p_ego2goal_world[1]

    qm = 10
    goal_lst = goal_lst / qm
    obs_lst = obs_lst / qm

    state = np.concatenate(
        (
            state_data_ego_pose,
            state_data_ego_th,
            goal_lst,
            obs_lst,
        ),
        axis=1,
    )

    return torch.as_tensor(state)


def tf2sim(sample, init_state, H):
    x = init_state[-1, 0] + torch.cumsum(sample[:, H:, 0], dim=0).detach().cpu().numpy()

    y = init_state[-1, 1] + torch.cumsum(sample[:, H:, 1], dim=0).detach().cpu().numpy()
    cth = (
        init_state[-1, 2] + torch.cumsum(sample[:, H:, 2], dim=0).detach().cpu().numpy()
    )
    sth = (
        init_state[-1, 3] + torch.cumsum(sample[:, H:, 3], dim=0).detach().cpu().numpy()
    )
    # pdb.set_trace()
    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    cth = np.expand_dims(cth, axis=-1)
    sth = np.expand_dims(sth, axis=-1)
    waypoints_wf = np.concatenate((x, y, cth, sth), axis=-1)

    return waypoints_wf


def tf_ego2w(obs_data_w, pred_ego):
    # takes model output and returns world-centric vector to obs/goal for use in pid controller
    cth = obs_data_w[2]
    sth = obs_data_w[3]

    p_wayptFromTable_w_x = obs_data_w[0] + cth * pred_ego[0] - sth * pred_ego[1]
    p_wayptFromTable_w_y = obs_data_w[1] + sth * pred_ego[0] + cth * pred_ego[1]

    p_table_w = obs_data_w[:2]

    p_waypt_w = p_table_w + np.concatenate(
        (
            p_wayptFromTable_w_x,
            p_wayptFromTable_w_y,
        ),
    )

    return p_waypt_w


def compute_reward(env=None, states=None) -> float:
    # states should be an N x 3 array
    assert len(states.shape) == 2, "state shape mismatch for vectorized compute_reward"
    assert states.shape[1] == 3, "state shape mismatch for vectorized compute_reward"
    assert states is not None, "states parameter cannot be None"
    n = states.shape[0]
    reward = np.zeros(n)
    # slack reward
    reward += -0.1
    dg = np.linalg.norm(states[:, :2] - env.goal, axis=1)

    a = 0.98
    const = 100.0
    r_g = 10.0 * np.power(a, dg - const)
    reward += r_g

    r_obs = np.zeros(n)
    b = -8.0
    c = 0.9
    const = 150.0
    d2obs_lst = np.asarray(
        [
            np.linalg.norm(
                states[:, :2] - np.array([env.obs_sprite[i].x, env.obs_sprite[i].y]),
                axis=1,
            )
            for i in range(env.num_obstacles)
        ],
        dtype=np.float32,
    )

    # negative rewards for getting close to wall
    for i in range(env.num_obstacles):
        d = d2obs_lst[i]
        r_obs += b * np.power(c, d - const)

    reward += r_obs

    return reward


def play_hil(
    env,
    ego="pid",
    model=None,
    mcfg=None,
    optimizer=None,
    train_stats=None,
    SEQ_LEN=60,
    H=15,
    playback_trajectory=None,
    n_steps=1000,
    zoom=1,
    fps=30,
    display_pred=True,
    display_gt=False,
):

    # record trajectory played
    trajectory = {}
    trajectory["states"] = []
    trajectory["plan"] = []
    trajectory["actions"] = []
    trajectory["rewards"] = []
    trajectory["fluency"] = []
    trajectory["terminal"] = []

    # reset environment
    obs = env.reset()
    done = False
    n_iter = 0
    running = True
    next_game_tick = time.time()
    clock = pygame.time.Clock()
    success = True

    # Initialize human input controller
    joysticks = init_joystick()  ## FIXME
    p2_id = 0

    # Initialize ego controls -  first 15 steps
    # assert train_stats is not None, "Missing train_stats file"
    # Keep running list of past H steps of simulator observations (need tf2model for model input conversion)
    s_queue = torch.zeros(
        (mcfg.H // mcfg.skip + 1, mcfg.LSIZE + 1), dtype=torch.float32
    ).to(device)
    past_states = []
    past_states.append(obs.tolist())
    obs = torch.from_numpy(obs).float()
    s_queue = update_queue(s_queue, obs.unsqueeze(0))
    u_queue = torch.zeros((mcfg.H // mcfg.skip, mcfg.ASIZE), dtype=torch.float32).to(
        device
    )
    h = None

    # COMPARE
    actions = playback_trajectory["actions"]
    waypoints_true = playback_trajectory["states"].tolist()
    delta_plan_sum = 0
    # Main Game loop
    while running:

        loops = 0

        if done:
            time.sleep(1)
            pygame.quit()
            break

        else:
            start = time.time()

            while time.time() > next_game_tick and loops < MAX_FRAMESKIP and not done:
                if display_gt:
                    env.draw_gt(waypoints_true)

                if h is None:
                    h = torch.zeros(
                        mcfg.n_layers, mcfg.BSIZE, mcfg.RSIZE, device=device
                    )
                # get human inputs
                u_h = np.array(
                    [
                        joysticks[p2_id].get_axis(0),
                        joysticks[p2_id].get_axis(1),
                    ]
                )
                u_h = torch.from_numpy(np.clip(u_h, -1.0, 1.0)).unsqueeze(0)
                debug_print("U_h", u_h)

                # if at start, n_iter < H, then populate states with state history and step
                if n_iter <= mcfg.H + mcfg.skip:
                    if n_iter % mcfg.skip != 0:
                        n_iter += 1
                        continue
                    # feed first H steps of state history into simulator
                    u_r = torch.from_numpy(actions[n_iter, :2]).unsqueeze(0)
                    u_all = torch.cat(
                        (u_r, u_h), dim=-1
                    )  # player 1 is blue, and in this sim human is player 2 (orange)
                    debug_print("U_all", u_all)

                    ## DEBUG
                    # n_iter_act = min(n_iter, actions.shape[0] - 1)
                    # pdb.set_trace()
                    # u_all = torch.from_numpy(actions[n_iter_act, :]).unsqueeze(0)
                    ## END DEBUG
                    u_queue = update_queue(u_queue, u_all)

                    # Update env with actions
                    obs, reward, done, info = env.step(list(u_all.squeeze()))
                    past_states.append(obs.tolist())
                    env.draw_past_states(past_states)

                    # Update obseravations for model

                    obs = torch.from_numpy(obs).float()
                    s_queue = update_queue(s_queue, obs.unsqueeze(0))
                    n_iter += 1
                    continue

                else:

                    # get model predictions
                    with torch.no_grad():

                        s_tf = tf2model(s_queue).repeat(mcfg.BSIZE, 1, 1)
                        u = u_queue.repeat(mcfg.BSIZE, 1, 1).float()
                        start_plan = time.time()
                        sample = model.sample(
                            s_tf, u, seq_len=(mcfg.SEQ_LEN // mcfg.skip)
                        )

                    waypoints = tf2sim(
                        sample[:, (mcfg.H // mcfg.skip) :, :4],
                        s_queue,
                        (mcfg.H // mcfg.skip),
                    )
                    # choose trajectory
                    ## TODO: select traj with rwd fn
                    # debug_print("true waypoint: ", waypoints_true[n_iter])
                    eval = np.sum(
                        np.array(
                            [
                                compute_reward(env, waypoints[i, :, :3])
                                for i in range(waypoints.shape[0])
                            ]
                        ),
                        -1,
                    )
                    best_traj = np.argmax(eval)
                    path = waypoints[best_traj, :, :]
                    end_plan = time.time()
                    delta_plan = end_plan - start_plan
                    delta_plan_sum = +delta_plan
                    print("Planning time: ", delta_plan)
                    # get ego action
                    # print(
                    #     "currstate", env.table.x, env.table.y, "pred waypoint: ", path
                    # )
                    # pdb.set_trace()
                    if display_pred:
                        env.update_prediction(path.tolist())
                    time_param = 6
                    pid_actions = pid_single_step(
                        env,
                        path[time_param, :4],
                        kp=0.15,
                        ki=0.0,
                        kd=0.0,
                        max_iter=40,
                        dt=CONST_DT,
                        eps=1e-2,
                        u_h=u_h.squeeze().numpy(),
                    )
                    pid_actions /= np.linalg.norm(pid_actions)

                    # print(
                    #     "true waypoint: ",
                    #     waypoints_true[n_iter][:2],
                    #     path[time_param, :4],
                    #     actions[n_iter, :2],
                    #     pid_actions,
                    # )

                    u_r = torch.from_numpy(np.clip(pid_actions, -1.0, 1.0)).unsqueeze(0)
                    # print("u_r", u_r)
                    # u_r = torch.zeros((1, 2))
                    # get pid actions to set u_r
                    # pdb.set_trace()
                    debug_print("U_r", u_r)

                # pdb.set_trace()
                u_all = torch.cat(
                    (u_r, u_h), dim=-1
                )  # player 1 is blue, and in this sim human is player 2 (orange)
                # print("U_all", u_all)

                ## DEBUG
                n_iter = min(n_iter, actions.shape[0] - 1)
                # pdb.set_trace()
                # u_all = torch.from_numpy(actions[n_iter, :]).unsqueeze(0)
                ## END DEBUG
                u_queue = update_queue(u_queue, u_all)

                # Update env with actions
                obs, reward, done, info = env.step(list(u_all.squeeze()))
                past_states.append(obs.tolist())
                env.draw_past_states(past_states)

                # Update obseravations for model

                obs = torch.from_numpy(obs).float()
                s_queue = update_queue(s_queue, obs.unsqueeze(0))
                # obs_std = model.standardize(obs, model.mean_s, model.std_s)
                # s_queue = update_queue(s_queue, obs_std.unsqueeze(0))

                # check if done/success
                if done:
                    if env.success:
                        success = True
                    else:
                        success = False

                    env.render(mode="human")
                    running = False
                    trajectory["states"].append(obs)
                    trajectory["plan"].append(path)
                    trajectory["actions"].append(u_all)
                    trajectory["rewards"].append(reward)
                    trajectory["terminal"].append(done)
                    break

                trajectory["states"].append(obs)
                trajectory["plan"].append(path)
                trajectory["actions"].append(u_all)
                trajectory["rewards"].append(reward)
                trajectory["terminal"].append(done)

                next_game_tick += CONST_DT
                loops += 1
            if loops == 0:
                continue
            else:

                delta_plan_sum = delta_plan_sum / (loops)
            print("Average planning time: ", delta_plan_sum)
            # iterate to next step in game loop
            debug_print("Loop: ", loops, "n_iter", n_iter, info)
            n_iter += 1

            # UPDATE DISPLAY
            if not env.done:
                env.redraw()
                ##if callback is not None:
                ##    callback(prev_obs, obs, action, rew, env_done, info)
                # CLOCK TICK
                clock.tick(FPS)
                if clock.get_fps() > 0:
                    debug_print("Reported dt: ", 1 / clock.get_fps())

                # process pygame events
                for event in pygame.event.get():
                    # test events, set key states
                    if event.type == pygame.KEYDOWN:
                        if event.key == 27:
                            running = False
                    elif event.type == pygame.QUIT:
                        running = False

            stop = time.time()
            duration = stop - start
            debug_print("Duration: ", duration)

    pygame.quit()

    # Save trajectory
    trajectory["states"] = torch.stack(trajectory["states"], dim=0).numpy()
    trajectory["plan"] = trajectory["plan"]
    trajectory["actions"] = torch.stack(trajectory["actions"], dim=0).numpy()
    trajectory["rewards"] = trajectory["rewards"]
    trajectory["terminal"] = trajectory["terminal"]
    trajectory["fluency"] = env.fluency

    return trajectory, success, n_iter


def play_traj(
    env,
    mode,
    ego="pid",
    model=None,
    optimizer=None,
    train_stats=None,
    SEQ_LEN=60,
    H=15,
    playback_trajectory=None,
    n_steps=1000,
    zoom=1,
    fps=30,
):

    # record trajectory played
    trajectory = {}
    trajectory["states"] = []
    trajectory["actions"] = []
    trajectory["rewards"] = []

    # reset environment
    obs = env.reset()
    done = False
    n_iter = 0
    running = True
    next_game_tick = time.time()
    clock = pygame.time.Clock()
    success = True

    # If playback trajectory is provided, use that instead of model + human control
    if mode == "playback_trajectory" or mode == "data_aug":
        actions = playback_trajectory["actions"]
        waypoints = playback_trajectory["states"]

    # Main Game loop
    while running:
        loops = 0

        if done:
            time.sleep(1)
            pygame.quit()
            break
        else:
            while time.time() > next_game_tick and loops < MAX_FRAMESKIP and not done:
                if mode == "playback_trajectory":
                    if True:  # n_iter < actions.shape[0]:
                        n_iter = min(n_iter, actions.shape[0] - 1)
                        # pdb.set_trace()
                        waypoint = waypoints[n_iter, :4]
                        u_h = actions[n_iter, 2:]
                        # debug_print("waypoint: ", waypoint)
                        pid_actions = pid_single_step(
                            env,
                            waypoint,
                            kp=0.2,
                            ki=0.0,
                            kd=0.0,
                            max_iter=40,
                            dt=CONST_DT,
                            eps=1e-2,
                            linear_speed_limit=[-2.0, 2.0],
                            angular_speed_limit=[-np.pi / 4, np.pi / 4],
                            u_h=u_h,
                        )
                        pid_actions = np.clip(pid_actions, -1.0, 1.0)
                        debug_print(
                            "actual robot action: ",
                            actions[n_iter, :2],
                            "pid action: ",
                            pid_actions,
                            "human action: ",
                            actions[n_iter, 2:],
                        )

                        u_all = [
                            pid_actions[0],
                            pid_actions[1],
                            actions[n_iter, 2],
                            actions[n_iter, 3],
                        ]

                        debug_print("sending actions", u_all)

                        # obs, reward, done, info = env.step(actions[n_iter, :])

                        obs, reward, done, info = env.step(u_all)
                        if done:
                            running = False
                            break
                        trajectory["actions"].append(pid_actions)
                        trajectory["states"].append(obs)

                        # mse = ((actions[n_iter, :] - u_all) ** 2).mean(axis=0)
                        # debug_print("MSE: ", mse)

                # iterate to next step in game loop
                debug_print("Loop: ", loops, "n_iter", n_iter, info)
                next_game_tick += CONST_DT
                loops += 1
            n_iter += 1

            # UPDATE DISPLAY
            if not env.done:
                env.redraw()
                ##if callback is not None:
                ##    callback(prev_obs, obs, action, rew, env_done, info)
                # CLOCK TICK
                clock.tick(FPS)
                if clock.get_fps() > 0:
                    debug_print("Reported dt: ", 1 / clock.get_fps())

                # process pygame events
                for event in pygame.event.get():
                    # test events, set key states
                    if event.type == pygame.KEYDOWN:
                        if event.key == 27:
                            running = False
                    elif event.type == pygame.QUIT:
                        running = False

    # Save trajectory
    trajectory["states"] = np.stack(trajectory["states"], axis=0)
    trajectory["actions"] = np.stack(trajectory["actions"], axis=0)

    return trajectory, success


def main(args):
    seed = 3
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ------------------------
    # Configure experiment
    # ------------------------
    # Modes: data_aug, playback_trajectory, hil

    root = join("datasets", dataset)

    FILES = [
        join(root, sd, ssd)
        for sd in listdir(root)
        if isdir(join(root, sd))
        for ssd in listdir(join(root, sd))
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
        print(save_dir)
        mkdir(save_dir)

    if args.run_mode != "playback_trajectory":
        num_avail_gpu = torch.cuda.device_count()
        debug_print("Number of GPUs available: ", num_avail_gpu)

        # ------------------------
        # 2 RESTORE MODEL
        # ------------------------
        # restoring - reference can be retrieved in artifacts panel
        # artifact_dir = "local-exp/bcrnn/bcrnn.ckpt"
        artifact_dir = "local-exp/vrnn_noact/model.ckpt"
        # artifact_dir = "local-exp/vrnn_noact/model.ckpt"
        # artifact_dir = "local-exp/hiafh75s/checkpoints/epoch=10-step=792.ckpt"
        # "local-exp/test-mapstuff/sbmp-hrcc/2n6tqru9/checkpoints/epoch=22-step=1288.ckpt"
        # artifact_dir = "local-exp/testmapstuff_bigdata/sbmp-hrcc/16y7q4ma/checkpoints/epoch=0-step=339.ckpt"
        model = VRNN.load_from_checkpoint(
            artifact_dir,
            hparams=args,
        )
        print("RESTORED MODEL from: ", artifact_dir)
        model.eval()

        ep = 0
        for f_idx in range(len(FILES)):
            f = FILES[f_idx]
            # pdb.set_trace()

            game_str = f.split("/")
            game = game_str[-1]
            match = [match for match in map_files if game in match]
            print("match", match)
            print("file", f)
            if "112" not in match[0]:
                continue

            env = gym.make(
                "cooperative_transport.gym_table:table-v0",
                obs=args.obs,
                control=args.control,
                map_config=args.map_config,
                run_mode="eval",
                # load_map=match[0],
                strategy_name=args.strategy,
                ep=ep,
                dt=CONST_DT,
                physics_control_type=args.physics_control_type,
            )

            trajectory, success, n_t = play_hil(
                env,
                ego="pid",
                model=model,
                mcfg=args,
                playback_trajectory=dict(np.load(f)),
                SEQ_LEN=args.SEQ_LEN,
                H=args.H,
            )
            if not exists(join(save_dir, *f.split("/")[3:4])):
                mkdir(join(save_dir, *f.split("/")[3:4]))  # test_holdout subdir
            if not exists(join(save_dir, join(*f.split("/")[3:5]))):
                mkdir(
                    join(save_dir, join(*f.split("/")[3:5]))
                )  # specific test run subdir

            # pdb.set_trace()
            with open(join(save_dir, "record.txt"), "a") as recordfile:
                if success:
                    recordfile.write("Success: " + str(n_t) + " " + f + "\n")
                else:
                    recordfile.write("Failure: " + str(n_t) + " " + f + "\n")

            debug_print("Saving run...\n", f)

            np.savez(
                join(save_dir, join(*f.split("/")[3:5]), f.split("/")[-1]),
                states=trajectory["states"],
                plan=trajectory["plan"],
                actions=trajectory["actions"],
                rewards=trajectory["rewards"],
                fluency=trajectory["fluency"],
                terminal=trajectory["terminal"],
            )
            print(
                "Saved to ",
                join(save_dir, join(*f.split("/")[3:5]), f.split("/")[-1]),
            )
            ep += 1

    else:

        for f in range(len(FILES)):
            game_str = f.split("/")
            game = game_str[-1].split(".")
            match = [match for match in map_files if game[0] in match]
            # pdb.set_trace()
            env = gym.make(
                "cooperative_transport.gym_table:table-v0",
                obs=args.obs,
                control=args.control,
                map_config=args.map_config,
                run_mode="eval",
                load_map=match[0],
                strategy_name=args.strategy,
                ep=args.ep,
                dt=CONST_DT,
                physics_control_type=args.physics_control_type,
            )
            # env.load_map = match[0]
            # env.run_mode = "eval"
            # debug_print(env.load_map, env.run_mode)
            # pdb.set_trace()

            # map_run = dict(np.load(match[0], allow_pickle=True))
            # pdb.set_trace()
            if exists(join(save_dir, join(*f.split("/")[-2:-1]))):
                # pdb.set_trace()
                debug_print("Failed trajectory previously processed. Skipping ", f)
                continue
            else:
                if exists(join(save_dir, join(*f.split("/")[-2:-1]), f.split("/")[-1])):
                    debug_print(
                        "Successful trajectory previously processed. Skipping ", f
                    )
                    continue
                else:
                    mkdir(join(save_dir, join(*f.split("/")[-2:-1])))
                debug_print("Running simulation on action trajectory: ", f)
                trajectory, success = play_traj(
                    env,
                    args.run_mode,
                    model=None,
                    optimizer=None,
                    train_stats=None,
                    playback_trajectory=dict(np.load(f)),
                    SEQ_LEN=args.SEQ_LEN,
                    H=args.H,
                    zoom=1,
                    fps=30,
                )
                if success:
                    # pdb.set_trace()
                    debug_print("Saving successful run...\n", f)
                    if not isdir(join(save_dir, f.split("/")[1])):
                        mkdir(join(save_dir, f.split("/")[1]))

                    if not isdir(join(save_dir, *f.split("/")[1:-1])):
                        mkdir(join(save_dir, *f.split("/")[1:-1]))

                    np.savez(
                        join(save_dir, join(*f.split("/")[1:-1]), f.split("/")[-1]),
                        states=trajectory["states"],
                        actions=trajectory["actions"],
                        # rewards=rewards,
                        # terminal=terminal,
                    )
                    debug_print(
                        "Saved to ",
                        join(save_dir, join(*f.split("/")[1:]), f.split("/")[-1]),
                    )
                else:
                    debug_print("Failed run. Not saved. Continue data collection.")
                    continue
                # env.reset()
        debug_print("DONE.")


if __name__ == "__main__":
    # ------------------------
    # 1 DEFINE ARGS
    # ------------------------
    # model
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="experiment seed")

    parser.add_argument(
        "--run_mode",
        type=str,
        default="hil",  # data_aug, playback_trajectory, HIL
        help="run mode",
    )
    parser.add_argument(
        "--obs",
        type=str,
        default="discrete",
        help="Define Observation Space, discrete/rgb",
    )
    parser.add_argument(
        "--control",
        type=str,
        default="joystick",
        help="Define Control Input, keyboard/joystick",
    )
    parser.add_argument(
        "--map_config",
        type=str,
        default=map_cfg,
        help="Map Config File Path",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="success_1",
        help="strat",
    )

    parser.add_argument(
        "--model_cfg",
        type=str,
        default=None,
        help="path to model_cfg",
    )

    parser.add_argument(
        "--trained_model",
        type=str,
        default=None,
        help="path to trained model",
    )

    parser.add_argument(
        "--ep",
        type=int,
        default=0,
        help="ep",
    )

    parser.add_argument(
        "--physics_control_type",
        type=str,
        default="force",
        help="ep",
    )

    # Args from model
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
        default="2vhongks",
        help="ID of run for resuming training. Just enter the id, without the model name and the version number.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=7,
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
        default=dataset,
        help="Name of of dataset directory.",
    )
    parser.add_argument(
        "--map_cfg_f",
        type=str,
        default="rnd_obstacle_v2.yml",
        help="File containing map config parameters.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs to train for.",
    )

    #### MODEL PARAMETERS ####
    parser.add_argument(
        "--skip",
        type=int,
        default=5,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--include_actions",
        type=str,
        default=0,
        help="Whether to include actions in the model",
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
        default=64,
        help="Number of hidden units for any sequence model used (gru, rnn, etc.)",
    )
    parser.add_argument(
        "--NGAUSS", type=int, default=6, help="Number of latents in vae."
    )
    parser.add_argument(
        "--GMM", type=int, default=0, help="Number of State Gaussians in GMM."
    )
    parser.add_argument(
        "--AGAUSS", type=int, default=2, help="Number of Action Gaussians in GMM."
    )
    parser.add_argument(
        "--SEQ_LEN",
        type=int,
        default=150,
        help="Total training sequence length (autoregressive pred len=SEQLEN-H).",
    )
    parser.add_argument("--H", type=int, default=30, help="Observation period length.")
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
        default=0.17,
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
        "--BSIZE", type=int, default=10, help="Batch size for training."
    )
    parser.add_argument("--recon", type=str, default="l2", help="loss")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Training learning rate."
    )

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
    parser.add_argument("--transform", type=str, default="mean-std")
    parser.add_argument("-results_dir", type=str, default="results")
    parser.add_argument("--plot_dir", type=str, default="plots")
    parser.add_argument("--restore", type=float, default=0)
    parser.add_argument("--sweep", type=float, default=0)

    args = parser.parse_args()

    args = parser.parse_args()

    main(args)


# modes
# "playback_trajectory" - sim mode is set to eval, get observations from simulator using trajectory of actions and save the trajectories
# "play_with_human" - get observations from simulator using human player completing task with robot
# "play_with_robot" - get observations from simulator using robot player completing task with robot


# python scripts/test_model.py --run_mode=playback_trajectory --map_config=one_obstacle.yml
