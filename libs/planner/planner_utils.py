import numpy as np
import torch
import pygame

from cooperative_transport.gym_table.envs.utils import (
    CONST_DT,
    L,
    debug_print,
    WINDOW_H,
    WINDOW_W,
)
def env_fn():
    steps_per_render = max(10 // FPS, 1)
    return MultiStepWrapper(
        VideoRecordingWrapper(
            FlattenObservation(
                env
            ),
            video_recoder=VideoRecorder.create_h264(
                fps=fps,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=22,
                thread_type="FRAME",
                thread_count=1,
            ),
            file_path=None,
            steps_per_render=steps_per_render,
        ),
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=2000,
    )

# ------------------------ VRNN Planner Utils  ------------------------
""" For planner used in Ng, et al. Learning to Plan for Human-Robot Cooperative Carrying. (ICRA 2023)."""


def get_action_from_wrench(wrench, current_state, u_h):
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
    # print("F_des: ", F_des)
    # f1_x, f1_y = F_des[0], F_des[1]

    return F_des  # in world frame


def get_joint_action_from_wrench(wrench, current_state):
    # pdb.set_trace()
    new_wx = wrench[0]
    new_wy = wrench[1]
    new_wz = wrench[2]
    new_wrench = np.array([new_wx, new_wy, new_wz]).T
    G = (
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [
            -L / 2.0 * np.sin(current_state[2]),
            -L / 2.0 * np.cos(current_state[2]),
            L / 2.0 * np.sin(current_state[2]),
            L / 2.0 * np.cos(current_state[2]),
        ],
    )
    F_des = np.linalg.pinv(G).dot(new_wrench)
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
    joint=False,
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
    # if u_h is None:
    #     raise ValueError("u_h was never passed to pid.")
    if joint:
        F_des_r = get_joint_action_from_wrench(wrench, curr_state)
    else:
        F_des_r = get_action_from_wrench(wrench, curr_state, u_h)
    return F_des_r


def update_queue(a, x):
    return torch.cat([a[1:, :], x], dim=0)


def tf2model(state_data, obstacles, zero_padding=False):
    # must remove theta obs (last dim of state_data)
    # takes observation (only map info) and returns ego-centric vector to obs/goal for use in model
    if obstacles.shape[0] < 3:
        if zero_padding:
            obstacles_filler = np.zeros(
                shape=(3 - obstacles.shape[0], 2), dtype=np.float32
            )
        else:
            num_fill = 3 - obstacles.shape[0]
            obstacles_filler = np.tile(obstacles[-1, :], (num_fill, 1))
        obstacles = np.concatenate(
            (obstacles, obstacles_filler),
            axis=0,
        )
    state_data = state_data.detach().numpy()
    state_xy = state_data[:, :2]
    state_th = state_data[:, 2:4]
    state_data_ego_pose = np.diff(state_xy, axis=0)
    state_data_ego_th = np.diff(state_th, axis=0)
    goal_lst = np.empty(shape=(state_data_ego_pose.shape[0], 2), dtype=np.float32)
    obs_lst = np.empty(shape=(state_data_ego_pose.shape[0], 6), dtype=np.float32)
    p_ego2obs_world = np.empty(shape=(6, ), dtype=np.float32)
    qo = 8
    qg = 8

    for t in range(state_data_ego_pose.shape[0]):
        p_ego2obs_world[::2] = obstacles.flatten()[::2] - state_xy[t, 0]
        p_ego2obs_world[1::2] = obstacles.flatten()[1::2] - state_xy[t, 1]
        p_ego2goal_world = state_data[t, 4:6] - state_xy[t, :]
        cth = state_data[t, 2]
        sth = state_data[t, 3]
        # rotate goal & obs in ego frame
        obs_lst[t, ::2] = np.asarray(cth * p_ego2obs_world[::2] + sth * p_ego2obs_world[1::2], dtype=np.float32) / qo
        obs_lst[t, 1::2] = np.asarray(-sth * p_ego2obs_world[::2] + cth * p_ego2obs_world[1::2], dtype=np.float32) / qo
        goal_lst[t, 0] = np.asarray(cth * p_ego2goal_world[0] + sth * p_ego2goal_world[1], dtype=np.float32) / qg
        goal_lst[t, 1] = np.asarray(-sth * p_ego2goal_world[0] + cth * p_ego2goal_world[1], dtype=np.float32) / qg

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
    x = init_state[-1, 0] + torch.cumsum(sample[:, H:, 0], dim=1).detach().cpu().numpy()

    y = init_state[-1, 1] + torch.cumsum(sample[:, H:, 1], dim=1).detach().cpu().numpy()
    cth = (
        init_state[-1, 2] + torch.cumsum(sample[:, H:, 2], dim=1).detach().cpu().numpy()
    )
    sth = (
        init_state[-1, 3] + torch.cumsum(sample[:, H:, 3], dim=1).detach().cpu().numpy()
    )
    # pdb.set_trace()
    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    cth = np.expand_dims(cth, axis=-1)
    sth = np.expand_dims(sth, axis=-1)
    waypoints_wf = np.concatenate((x, y, cth, sth), axis=-1)

    return waypoints_wf


def tfego2w(obs_data_w, pred_ego):
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




def is_safe(state, collision_checking_env=None):
    """
    Customize state validity checker function for external planners
    Input: current state
    Output: True if current state is valid (collision free, etc.)
    """
    collision, _ = collision_checking_env.mp_check_collision_and_success(state)
    return not collision


def mp_check_collision_and_success(env, state) -> bool:
    """Check for collisions and success.

    Returns
    -------
    collided : Boolean
        Whether the table has collided with the obstacles
    success : Boolean
        Whether the table has reached the target
    """
    # set table position
    env.table.x = state[0]
    env.table.y = state[1]
    env.table.angle = state[2]
    # update sprite
    env.table.image = pygame.transform.rotate(
        env.table.original_img, np.degrees(env.table.angle)
    )
    env.table.rect = env.table.image.get_rect(center=(env.table.x, env.table.y))
    env.table.mask = pygame.mask.from_surface(env.table.image)

    hit_list = pygame.sprite.spritecollide(
        env.table, env.done_list, False, pygame.sprite.collide_mask
    )

    collision = False
    success = False

    if any(hit_list):

        collision = True
        if any(
            pygame.sprite.spritecollide(
                env.table, [env.target], False, pygame.sprite.collide_mask
            )
        ):
            success = True
            debug_print("HIT TARGET")
        else:
            debug_print("HIT OBSTACLE")
    else:
        # wall collision
        if (
            not env.screen.get_rect().contains(env.table)
            or env.table.rect.left < 0
            or env.table.rect.right > WINDOW_W
            or env.table.rect.top < 0
            or env.table.rect.bottom > WINDOW_H
        ):

            collision = True
            debug_print("HIT WALL")

    return collision, success
