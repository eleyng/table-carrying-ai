import gym
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy

### NOTE: This test only checks for env properties. We do not expect SAC to converge to a successful policy in this test.

env = gym.make("cooperative_transport.gym_table:table-v0")

print("Select random actions and step.")
obs = env.reset()
n_steps = 10
for _ in range(n_steps):
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(info)
    if done:
        obs = env.reset()

print("Check complete. Starting training.", reward)

model = SAC(
    MlpPolicy, env, buffer_size=5, batch_size=10, verbose=1
)  # batch consist of bsz number of (s,a,r,s') tuples

model.learn(total_timesteps=100000, log_interval=1)
model.save("sac_tabletest")

print("Training complete, and model saved.", "\n"*100)

del model  # remove to demonstrate saving and loading

model = SAC.load("sac_tabletest")

print("Model loaded.")

obs = env.reset()
print("Predict mode.")
num_eval_episodes = 0
while num_eval_episodes < 10:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
        num_eval_episodes += 1

print("Test complete.")
