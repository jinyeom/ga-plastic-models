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
        trunc_thresh=100,
        sample_size=2,
    )

    experiment = Experiment(args)

    random.seed(args.seed)
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)

    # run genetic algorithm
    ga_run(args, experiment)
