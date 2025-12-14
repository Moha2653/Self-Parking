import pygad
import numpy as np
from parking_env import ParkingEnv
from evo_policy import EvoPolicy
from torch.utils.tensorboard import SummaryWriter
import os

ENV_EPISODES = 3
MAX_STEPS = 600

env = ParkingEnv(robustness=True)

LOG_DIR = "logs/EVO"
MODEL_DIR = "models/EVO"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

writer = SummaryWriter(LOG_DIR)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

policy = EvoPolicy(obs_dim, act_dim)

GENOME_SIZE = int(
    obs_dim * 32 +
    32 +
    32 * act_dim +
    act_dim
)

def fitness_function(ga, solution, solution_idx):
    policy.set_weights(solution)
    total_reward = 0.0

    for _ in range(ENV_EPISODES):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

    return total_reward / ENV_EPISODES
generation_counter = {"gen": 0}

def on_generation(ga_instance):
    gen = generation_counter["gen"]

    best_fitness = ga_instance.best_solution()[1]
    avg_fitness = np.mean(ga_instance.last_generation_fitness)

    writer.add_scalar("Fitness/Best", best_fitness, gen)
    writer.add_scalar("Fitness/Average", avg_fitness, gen)

    solution, _, _ = ga_instance.best_solution()
    np.save(f"{MODEL_DIR}/best_gen_{gen}.npy", solution)

    print(
        f"[GEN {gen}] "
        f"Best: {best_fitness:.2f} | "
        f"Avg: {avg_fitness:.2f}"
    )

    generation_counter["gen"] += 1


ga = pygad.GA(
    num_generations=1000,
    num_parents_mating=10,
    fitness_func=fitness_function,
    sol_per_pop=30,
    num_genes=GENOME_SIZE,
    init_range_low=-1.0,
    init_range_high=1.0,
    parent_selection_type="sss",
    keep_parents=2,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=10,
    on_generation=on_generation
)

ga.run()

solution, solution_fitness, _ = ga.best_solution()
np.save(f"{MODEL_DIR}/best_final.npy", solution)

writer.add_text(
    "Training",
    f"Final fitness: {solution_fitness:.2f}"
)

writer.close()

print("Best fitness:", solution_fitness)
env.close()
