import os
import csv
import numpy as np
from stable_baselines3 import PPO
from parking_env import ParkingEnv
import gymnasium as gym

model_path = "models/parking_PPO/parking_PPO_250000"
csv_output = "results.csv"
episodes = 20

