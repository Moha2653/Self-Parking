# evaluate_policy.py
import csv
from pathlib import Path
import argparse
import numpy as np
from stable_baselines3 import PPO, SAC
from parking_env import ParkingEnv
import gymnasium as gym

def evaluate(model_path, n_episodes=20, render=False):
    # detect algo by name
    if "sac" in str(model_path).lower():
        ModelCls = SAC
    else:
        ModelCls = PPO
    model = ModelCls.load(str(model_path))
    results = []
    for ep in range(n_episodes):
        obs, _ = ParkingEnv().reset()
        done = False
        total_reward = 0.0
        steps = 0
        success = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = ParkingEnv().step(obs) if False else (None,0,True,False,{})
            # We must interact with a real env instance - create new for each episode:
            env = ParkingEnv(render_mode="human" if render else None)
            obs, _ = env.reset()
            break
        # simplified evaluation (call the helper below in practice)
        results.append({"episode": ep+1, "total_reward": 0, "steps":0, "success": False})
    # write CSV stub
    out = Path("evaluation_results.csv")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("Wrote", out)

if __name__ == "__main__":
    print("This script contains an evaluation stub. Use the evaluate function in your notebook or adapt quickly.")
