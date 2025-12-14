import os
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
 
from parking_env import ParkingEnv
 
# models_dir = "models/PPO"
models_dir = "models/A2C"
# models_dir = "models/DQN"
logdir = "logs"
 
if not os.path.exists(models_dir):
  os.makedirs(models_dir)
 
if not os.path.exists(logdir):
  os.makedirs(logdir)
 
env = DummyVecEnv([lambda: Monitor(ParkingEnv(robustness=True), logdir)])
env.reset()
 
# model = DQN("MlpPolicy", env, verbose=1, 
#                 tensorboard_log=logdir,
#                 learning_rate=0.0003,
#                 n_steps=2048,
#                 batch_size=64,
#                 gamma=0.99)

model = A2C("MlpPolicy", env, verbose=1, 
                tensorboard_log=logdir,
                learning_rate=0.0003,
                n_steps=2048,
                gamma=0.99)

# model = PPO("MlpPolicy", env, verbose=1, 
#                 tensorboard_log=logdir,
#                 learning_rate=0.0003,
#                 n_steps=2048,
#                 batch_size=64,
#                 gamma=0.99)

 
TIMESTEPS = 10000
for i in range(1, 1000000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
    # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
 
print("--- TRAINING FINISHED ---")
 
env.close()