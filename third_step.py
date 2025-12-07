import gymnasium as gym
from stable_baselines3 import PPO, A2C

env = gym.make("LunarLander-v3", render_mode="human")
env.reset()

models_dir = "models/PPO"
model_path = f"{models_dir}/250000"

model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
  obs = env.reset()[0]
  done = False
  while not done:
    env.render()
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated


env.close()