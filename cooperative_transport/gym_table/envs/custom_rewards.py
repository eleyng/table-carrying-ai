import numpy as np
from libs.planner.planner_utils import pid_single_step
from cooperative_transport.gym_table.envs.utils import L, CONST_DT

## Define custom reward functions here
def custom_reward_function(states, goal, obs, interaction_forces=None, vectorized=False, env=None, skip=5, u_h=None):
    # states should be an N x state_dim array
    assert (
        len(states.shape) == 2
    ), "state shape mismatch for compute_reward. Expected (n, {0}), where n is the set of states you are evaluating. Got {1}".format(
        states.shape
    )

    assert states is not None, "states parameter cannot be None"

    n = states.shape[0]
    reward = np.zeros(n)
    # slack reward
    reward += -0.1

    dg = np.linalg.norm(states[:, :2] - goal, axis=1)

    a = 0.98
    const = 100.0
    r_g = 10.0 * np.power(a, dg - const)
    reward += r_g

    r_obs = np.zeros(n)
    b = -100.0
    c = 0.98
    const = 200.0

    num_obstacles = obs.shape[0]
    if states is not None:
        d2obs_lst = np.asarray(
            [
                np.linalg.norm(states[:, :2] - obs[i, :], axis=1)
                for i in range(num_obstacles)
            ],
            dtype=np.float32,
        )

    # negative rewards for getting close to wall
    for i in range(num_obstacles):
        d = d2obs_lst[i]
        if d.any() < const:
            r_obs += b * np.power(c, d - const)

    reward += r_obs

    if interaction_forces is not None:
        if states.shape[0] == 1:
            interaction_forces = 0 #FIXME: hack to avoid error when running in env withtout planning
            pass
        else:
            pid_actions = pid_single_step(
                                env,
                                states[skip, :4],
                                kp=0.15,
                                ki=0.0,
                                kd=0.0,
                                max_iter=40,
                                dt=CONST_DT,
                                eps=1e-2,
                                u_h=u_h.squeeze().numpy(),
                            )
            pid_actions /= np.linalg.norm(pid_actions)
            interaction_forces = compute_interaction_forces(states[skip, :4], pid_actions, u_h.detach().numpy().squeeze())
            reward += interaction_forces_reward(interaction_forces)

    if vectorized:
        return reward
    else:
        return reward[0]


def interaction_forces_reward(interaction_forces):
    # interaction forces penalty : penalize as interaction forces stray from zero
    penalty = 0.5 * (interaction_forces ** 2)

    return -penalty

def compute_interaction_forces(table_state, f1, f2):
    table_center_to_player1 = np.array(
            [
                table_state[0] + (L/2) * table_state[2],
                table_state[1] + (L/2) * table_state[3],
            ]
        )
    table_center_to_player2 = np.array(
        [
            table_state[0] - (L/2) * table_state[2],
            table_state[1] - (L/2) * table_state[3],
        ]
    )
    inter_f = (f1 - f2) @ (
            table_center_to_player1 - table_center_to_player2
    )
    return inter_f
