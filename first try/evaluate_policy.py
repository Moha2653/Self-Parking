import os
import csv
import numpy as np
from stable_baselines3 import PPO
from parking_env import ParkingEnv

# ============================================================
# CONFIGURACIONES
# ============================================================

MODEL_PATH = "models/parking_PPO/parking_PPO_10000"   # <-- Cambia si quieres otro modelo
N_EPISODES = 20
CSV_OUTPUT = "evaluation_results.csv"

# ============================================================
# EVALUACIÓN
# ============================================================

def evaluate_model(model_path, n_episodes=10, output_csv="eval.csv"):
    # Cargar modelo
    print(f"Cargando modelo: {model_path}")
    model = PPO.load(model_path)

    # Crear ambiente
    env = ParkingEnv()

    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        success = False
        collision = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            # Check condiciones especiales
            if info.get("collision", False):
                collision = True
                done = True

            if info.get("success", False):
                success = True
                done = True

            if terminated or truncated:
                done = True

        results.append({
            "episode": ep + 1,
            "total_reward": total_reward,
            "steps": steps,
            "success": success,
            "collision": collision
        })

        print(f"Ep {ep+1}/{n_episodes} | Reward={total_reward:.2f} | Steps={steps} "
              f"| Success={success} | Collision={collision}")

    # Guardar CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResultados guardados en: {output_csv}")

    # Estadísticas globales
    rewards = [r["total_reward"] for r in results]
    success_rate = sum(r["success"] for r in results) / n_episodes * 100
    collision_rate = sum(r["collision"] for r in results) / n_episodes * 100

    print("\n================= RESUMEN =================")
    print(f"Reward promedio: {np.mean(rewards):.2f}")
    print(f"Tasa de éxito: {success_rate:.1f}%")
    print(f"Tasa de colisión: {collision_rate:.1f}%")
    print("===========================================")


if __name__ == "__main__":
    evaluate_model(MODEL_PATH, N_EPISODES, CSV_OUTPUT)
