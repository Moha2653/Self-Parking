from parking_env import ParkingEnv
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os

models_dir = "models/parking_PPO"
logdir = "logs/parking_PPO"

if not os.path.exists(models_dir):
  os.makedirs(models_dir)

if not os.path.exists(logdir):
  os.makedirs(logdir)

def register_env():
  try: 
    register(
      id='ParkingEnv-v0',
      entry_point='parking_env:ParkingEnv',
    )
  except Exception:
    pass

def make_env():
  env = ParkingEnv()
  env = Monitor(env)
  return env

def main():
  register_env()
  env = DummyVecEnv([make_env])

  model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir,
              learning_rate=3e-4, n_steps=2048, batch_size=64, 
              n_epochs=10, clip_range=0.2)

  TIMESTEPS = 10000
  for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="parking_PPO")
    model.save(f"{models_dir}/parking_PPO_{TIMESTEPS*i}")

  test_env = ParkingEnv()
  obs, _ = test_env.reset()

  for step in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
      break

if __name__ == "__main__":
  main()