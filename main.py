# xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python main.py

from multiprocessing import set_start_method
import random
import numpy as np
import torch
import gym
from utils import Arguments, Experiment
from ga import ga_run

torch.set_num_threads(1)

# disable OpenAI Gym logger
gym.logger.set_level(40)

if __name__ == "__main__":
    set_start_method("forkserver", force=True)

    args = Arguments(
        seed=0,
        artifact_path="./artifacts",
        env_id="CarRacing-v0",
        plastic=True,
        obs_size=64,
        latent_size=128,
        hidden_size=256,
        action_size=3,
        discrete_vae=True,
        mut_mode="MUT-MOD",
        mut_pow=0.01,
        num_generations=200,
        pop_size=200,
        num_workers=16,
        time_limit=1000,
        num_evals=1,
        num_topk=3,
        num_evals_elite=20,
        num_evals_global=100,
        trunc_thresh=100,
        sample_size=2,
    )

    experiment = Experiment(args)

    random.seed(args.seed)
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)

    # run genetic algorithm
    ga_run(args, experiment)

    # evaluate the final model
    global_elite = Individual(
        args.plastic,
        args.mut_mode,
        args.mut_pow,
        obs_size=args.obs_size,
        latent_size=args.latent_size,
        hidden_size=args.hidden_size,
        action_size=args.action_size,
        discrete_vae=args.discrete_vae,
    )

    global_elite.load_solution(experiment.final_model_path)

    with Pool(args.num_workers) as pool:
        global_elite.run_solution(
            pool,
            time_limit=args.time_limit,
            num_evals=args.num_evals_global,
            early_termination=False,
            force_eval=True,
        )
        mean, std = global_elite.evaluate_solution(args.num_evals_global)

    experiment.log(f"Final performance: {mean:d} ± {std:d}")
