# train_parking.py
import argparse
from pathlib import Path
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from parking_env import ParkingEnv

ENV_ID = "ParkingEnv-v0"

def register_env():
    try:
        register(id=ENV_ID, entry_point="parking_env:ParkingEnv")
    except Exception:
        pass

def make_env(seed=0, render_mode=None, noise=False):
    def _init():
        env = ParkingEnv(render_mode=None, seed=seed, noise=noise)
        env = Monitor(env)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo","sac"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--save-dir", type=str, default="./models")
    parser.add_argument("--tensorboard", type=str, default="./tb_logs")
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()

    register_env()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([make_env(seed=0, noise=args.noise)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    if args.algo == "ppo":
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=args.tensorboard,
                    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10)
    else:
        model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log=args.tensorboard,
                    learning_rate=3e-4, train_freq=1)

    total_timesteps = args.timesteps
    print(f"Training {args.algo} for {total_timesteps} timesteps (noise={args.noise})")
    model.learn(total_timesteps=total_timesteps)

    model_path = save_dir / f"{args.algo}_parking_{total_timesteps}.zip"
    model.save(str(model_path))
    # save VecNormalize stats
    vec_env.save(str(save_dir / "vecnormalize.pkl"))
    print("Saved model to", model_path)

if __name__ == "__main__":
    main()
