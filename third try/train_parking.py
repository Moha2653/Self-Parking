import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from parking_env import ParkingEnv

def main():
    # Directories
    log_dir = "./logs_parking/"
    models_dir = "./models_parking/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Initialize Env (With Robustness Bonus)
    env = DummyVecEnv([lambda: ParkingEnv(robustness=True)])

    # Callback to save best model
    eval_callback = EvalCallback(env, best_model_save_path=models_dir,
                                 log_path=log_dir, eval_freq=5000,
                                 deterministic=True, render=False)

    # Define PPO Model
    model = PPO("MlpPolicy", env, verbose=1, 
                tensorboard_log=log_dir,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                gamma=0.99)

    print("--- STARTING TRAINING ---")
    print("Run in terminal: tensorboard --logdir=./logs_parking/")
    
    # Train
    total_timesteps = 100000 
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    print("--- TRAINING FINISHED ---")
    
    # Save Final Model
    model.save(f"{models_dir}/ppo_parking_final")
    env.close()

if __name__ == "__main__":
    main()