import os
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
 
from parking_env import ParkingEnv
 
models_dir = "models/PPO"
# models_dir = "models/A2C"
logdir = "logs"
 
 
if not os.path.exists(models_dir):
  os.makedirs(models_dir)
 
if not os.path.exists(logdir):
  os.makedirs(logdir)
 
env = DummyVecEnv([lambda: Monitor(ParkingEnv(robustness=True), logdir)])
env.reset()
 
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
 
# model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
 
 
TIMESTEPS = 10000
for i in range(1, 1000000):
    # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
 
print("--- TRAINING FINISHED ---")
 
env.close()