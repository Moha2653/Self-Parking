import time
import numpy as np
import imageio
from parking_env import ParkingEnv
from evo_policy import EvoPolicy


def evaluate():
    env = ParkingEnv(render_mode="human", robustness=True)
    env_rgb = ParkingEnv(render_mode="rgb_array", robustness=True)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    evo_policy = EvoPolicy(obs_dim, act_dim)
    evo_policy.set_weights(
        np.load("models/EVO/best_gen_407.npy")
    )

    episodes = 5

    for ep in range(episodes):
        obs, _ = env.reset()
        obs_rgb, _ = env_rgb.reset()
        done = False
        score = 0.0
        frames = []

        print(f"\n--- Episode {ep + 1} ---")

        while not done:
            action = evo_policy.act(obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            obs_rgb, reward_rgb, terminated_rgb, truncated_rgb, _ = env_rgb.step(action)

            done = terminated or truncated
            score += reward

            frame = env_rgb.render()
            if frame is not None:
                frames.append(frame)

            time.sleep(0.01)

        gif_name = f"gad_parking_{ep+1}.gif"
        if len(frames) > 0:
            imageio.mimsave(gif_name, frames, fps=30)

        print(f"Episode finished. Score: {score:.2f}")
        time.sleep(0.1)

    env.close()
    env_rgb.close()


if __name__ == "__main__":
    evaluate()
