import numpy as np
import torch

from cooperative_transport.gym_table.envs.utils import CONST_DT, L

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
    # f1_x, f1_y = F_des[0], F_des[1]

    return F_des  # in world frame



def get_joint_action_from_wrench(wrench, current_state):
    # pdb.set_trace()
    new_wx = wrench[0]
    new_wy = wrench[1]
    new_wz = wrench[2]
    new_wrench = np.array([new_wx, new_wy, new_wz]).T
    G = (
        [1, 0, 0, 0],
        [0, 1, 0, 0],
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
    if u_h is None:
        raise ValueError("u_h was never passed to pid.")
    if joint:
        F_des_r = get_joint_action_from_wrench(wrench, curr_state)
    else:
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

    qm = 8
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