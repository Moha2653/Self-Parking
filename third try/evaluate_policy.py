import time
import gymnasium as gym
from stable_baselines3 import PPO
from parking_env import ParkingEnv

def evaluate():
    # Render mode Human to see the window
    env = ParkingEnv(render_mode="human", robustness=True)

    model_path = "./models_parking/ppo_parking_final.zip" 
    
    # Check if model exists
    try:
        model = PPO.load(model_path)
        print(f"Model loaded: {model_path}")
        use_model = True
    except:
        print("WARNING: Model not found. Running random actions for testing.")
        use_model = False

    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        
        print(f"--- Episode {ep + 1} ---")
        
        while not done:
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            
            # Optional: Slow down slightly if too fast
            # time.sleep(0.01)
        
        print(f"Episode finished. Score: {score:.2f}")
        time.sleep(1.0)

    env.close()

if __name__ == "__main__":
    evaluate()