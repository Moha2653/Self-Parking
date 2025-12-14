import time
from stable_baselines3 import PPO, DQN, A2C
from parking_env import ParkingEnv

def evaluate():

    env = ParkingEnv(render_mode="human", robustness=True)
    # env_rgb = ParkingEnv(render_mode="rgb_array", robustness=True)

    model_path = "models/PPO/600000.zip"
    try:
        model = PPO.load(model_path)
        use_model = True
    except:
        use_model = False

    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        # obs_rgb, _ = env_rgb.reset()

        done = False
        score = 0
        frames = []

        print(f"--- Episode {ep + 1} ---")

        while not done:
            if use_model: 
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
 
            time.sleep(0.01)
 
        print(f"Episode finished. Score: {score:.2f}")
        time.sleep(1.0)

    env.close()
    # env_rgb.close()

if __name__ == "__main__":
    evaluate()
