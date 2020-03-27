from multiprocessing import Pool
import random
import numpy as np
import torch
from utils import Arguments, Experiment
from train import Individual

args = Arguments(
    seed=0,
    artifact_path="./artifacts",
    env_id="CarRacing-v0",
    obs_size=64,
    latent_size=128,
    hidden_size=256,
    action_size=3,
    discrete_vae=True,
    mut_mode="MUT-MOD",
    mut_pow=0.01,
    num_generations=1200,
    pop_size=200,
    num_workers=10,
    time_limit=1000,
    num_evals=1,
    early_termination=True,
    num_topk=3,
    num_evals_elite=20,
    trunc_thresh=100,
    sample_size=2,
)

experiment = Experiment(args)

random.seed(args.seed)
np.random.seed(seed=args.seed)
torch.manual_seed(args.seed)

population = [
    Individual(
        args.obs_size,
        args.latent_size,
        args.hidden_size,
        args.action_size,
        args.discrete_vae,
        args.mut_mode,
        args.mut_pow,
    )
    for _ in range(args.pop_size)
]

for gen in range(args.num_generations):
    pool = Pool(args.num_workers)

    for ind in population:
        ind.run_solution(
            pool,
            time_limit=args.time_limit,
            num_evals=args.num_evals,
            early_termination=args.early_termination,
            force_eval=True,
        )

    fitnesses = []
    for ind in population:
        ind.is_elite = False
        mean_fitness, _ = ind.evaluate_solution(args.num_evals)
        fitnesses.append(mean_fitness)

    population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

    # reevaluate top k individuals

    topk = population[: args.num_topk]

    for ind in topk:
        ind.run_solution(
            pool,
            time_limit=args.time_limit,
            num_evals=args.num_evals_elite,
            early_termination=args.early_termination,
            force_eval=False,
        )

    topk_fitnesses = []
    for ind in topk:
        mean_fitness, _ = ind.evaluate_solution(args.num_evals_elite)
        topk_fitnesses.append(mean_fitness)

    topk = sorted(topk, key=lambda ind: ind.fitness, reverse=True)

    elite = topk[0]
    elite.is_elite = True
    experiment.update_best(elite, elite.fitness)

    pool.close()  # evaluation done!

    # log generation statistics

    experiment.log(
        gen,
        fit_mean=np.mean(fitnesses),
        fit_std=np.std(fitnesses),
        topk_fit_mean=np.mean(topk_fitnesses),
        topk_fit_std=np.std(topk_fitnesses),
        elite_fit=elite.fitness,
    )

    # reproduce

    if len(population) > args.trunc_thresh - 1:
        del population[args.trunc_thresh - 1 :]
    population.append(elite)

    offsprings = []

    while len(offsprings) < args.trunc_thresh:
        samples = random.sample(population, args.sample_size)
        samples = sorted(samples, key=lambda ind: ind.fitness, reverse=True)
        selected = samples[0]

        # an elite with the highest fitness wins
        for ind in samples:
            if ind.is_elite:
                selected = ind
                break

        child = selected.clone()
        child.mutate()

        offsprings.append(child)

    population.extend(offsprings)
