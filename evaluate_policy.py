import time
import imageio
from stable_baselines3 import PPO
from parking_env import ParkingEnv

def evaluate():
    #env = ParkingEnv(render_mode="human", robustness=True)
    env_rgb = ParkingEnv(render_mode="rgb_array", robustness=True)

    model_path = "models/PPO/1150000.zip"
    try:
        model = PPO.load(model_path)
        use_model = True
    except:
        use_model = False

    episodes = 5
    for ep in range(episodes):
        #obs, _ = env.reset()
        obs_rgb, _ = env_rgb.reset()
        done = False
        score = 0
        frames = []

        print(f"--- Episode {ep + 1} ---")

        while not done:

            # Acción del modelo usando obs_rgb
            if use_model:
                action, _ = model.predict(obs_rgb, deterministic=True)
            else:
                action = env_rgb.action_space.sample()

            # Step usando solo env_rgb
            obs_rgb, reward, terminated, truncated, info = env_rgb.step(action)
            done = terminated or truncated
            score += reward

            # Capturar frame
            frame = env_rgb.render()
            frames.append(frame)

            time.sleep(0.01)

        # Guardar GIF del episodio
        gif_name = f"parking_episode_{ep+1}.gif"
        imageio.mimsave(gif_name, frames, fps=30)
        print(f"GIF saved: {gif_name}")

        print(f"Episode finished. Score: {score:.2f}")
        time.sleep(1.0)

    #env.close()
    env_rgb.close()

if __name__ == "__main__":
    evaluate()
