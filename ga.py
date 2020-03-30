from multiprocessing import Pool
import random
import numpy as np
from train import Individual


def ga_run(args, experiment):
    population = [
        Individual(
            args.mut_mode,
            args.mut_pow,
            obs_size=args.obs_size,
            latent_size=args.latent_size,
            hidden_size=args.hidden_size,
            action_size=args.action_size,
            discrete_vae=args.discrete_vae,
        )
        for _ in range(args.pop_size)
    ]

    for gen in range(args.num_generations):

        # evaluate the population

        fitnesses = []

        with Pool(args.num_workers) as pool:
            for ind in population:
                ind.run_solution(
                    pool,
                    time_limit=args.time_limit,
                    num_evals=args.num_evals,
                    force_eval=True,
                )

            for ind in population:
                ind.is_elite = False
                mean_fitness, _ = ind.evaluate_solution(args.num_evals)
                fitnesses.append(mean_fitness)

        population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

        # reevaluate top k individuals

        topk = population[: args.num_topk]

        topk_fitnesses = []

        with Pool(args.num_workers) as pool:
            for ind in topk:
                ind.run_solution(
                    pool,
                    time_limit=args.time_limit,
                    num_evals=args.num_evals_elite,
                    force_eval=False,
                )

            for ind in topk:
                mean_fitness, _ = ind.evaluate_solution(args.num_evals_elite)
                topk_fitnesses.append(mean_fitness)

        topk = sorted(topk, key=lambda ind: ind.fitness, reverse=True)

        elite = topk[0]
        elite.is_elite = True
        experiment.update_best(elite, elite.fitness)

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
            selected = samples[0]  # select one with the highest fitness

            # an elite with the highest fitness wins
            for ind in samples:
                if ind.is_elite:
                    selected = ind
                    break

            child = selected.clone()
            child.mutate()

            offsprings.append(child)

        population.extend(offsprings)
