"""
BC Agent training from imitation library: https://github.com/HumanCompatibleAI/imitation

This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from cooperative_transport.gym_table.envs.utils import CONST_DT

env = gym.make(
        "cooperative_transport.gym_table:table-v0",
        render_mode="gui",
        control="joystick",
        map_config="cooperative_transport/gym_table/config/maps/rnd_obstacle_v2.yml",
        run_mode="demo",
        load_map=None,
        run_name="train_bc",
        ep=0,
        dt=CONST_DT,
        include_interaction_forces_in_rewards=False
    )
rng = np.random.default_rng(88)


def train_expert():
    print("Training a expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=88,
        batch_size=1000,
        ent_coef=0.01,
        learning_rate=0.001,
        n_epochs=100,
        n_steps=1000,
    )
    expert.learn(total_timesteps=10000000, progress_bar=True)  # Note: change this to 1000000 to train a decent expert.
    return expert


def sample_expert_transitions():
    expert = train_expert()

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    return rollout.flatten_trajectories(rollouts)


transitions = sample_expert_transitions()
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)

print("Evaluating a policy before training")

reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=10,
    render=True,
)
print(f"Reward before training: {reward}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=100)

bc_trainer.save_policy("bc.pt")

reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=10,
    render=True,
)
print(f"Reward after training: {reward}")