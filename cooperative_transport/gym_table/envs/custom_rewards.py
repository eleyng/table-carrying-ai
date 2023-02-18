import numpy as np

## Define custom reward functions here
def custom_reward_function(states, goal, obs, interaction_forces=None, vectorized=False):

    assert (
        len(states.shape) == 2
    ), "state shape mismatch for compute_reward. Expected (n, {0}), where n is the batch size you are evaluating. Got {1}".format(
        states.shape[1],
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
    b = -8.0
    c = 0.9
    const = 150.0

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
        reward += interaction_forces_reward(interaction_forces)

    if vectorized:
        return reward
    else:
        return reward[0]


def interaction_forces_reward(interaction_forces):
    # interaction forces penalty : penalize as interaction forces stray from zero
    penalty = 0.5 * (interaction_forces ** 2)

    return -penalty

### NOTE: See other possible fluency metrics under compute_fluency_metrics in table_env.py