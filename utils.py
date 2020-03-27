import json
import logging
import math
from pathlib import Path
import sys
import uuid

import torch
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate
from termcolor import colored


class Arguments:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Experiment:
    def __init__(self, args):
        self.args = args

        self.artifact_path = Path(args.artifact_path or "/tmp")
        self.exp_id = f"{args.env_id}-{uuid.uuid4().hex}"

        self.exp_path = self.artifact_path / self.exp_id
        self.exp_path.mkdir(parents=True, exist_ok=False)

        # set up logger
        self.logger = logging.getLogger(self.exp_id)
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(self.exp_path / "progress.log")
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]:\n%(message)s\n")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # set up TensorBoard
        self.writer = SummaryWriter(self.exp_path)

        # current best performance
        self.global_best_fitness = -math.inf

        colored_exp_id = colored(self.exp_id, "green")
        print(f"Starting experiment: {colored_exp_id}")

        table = [(k, v) for k, v in self.args.__dict__.items()]
        print(tabulate(table, tablefmt="fancy_grid", stralign="center", numalign="center"))

        with open(self.exp_path / "config.json", "w") as f:
            f.write(json.dumps(self.args.__dict__, indent=4, sort_keys=True))

    def log(self, global_step, **kwargs):
        kwargs["global_step"] = global_step
        kwargs["global_best_fitness"] = self.global_best_fitness

        keys, values = tuple(zip(*kwargs.items()))
        keys = list(keys)
        values = list(values)

        self.logger.info(
            tabulate(
                [values],
                headers=keys,
                tablefmt="fancy_grid",
                floatfmt=".5f",
                stralign="center",
                numalign="center",
            )
        )

        for k, v in kwargs.items():
            self.writer.add_scalar(k, v, global_step=global_step)

    def checkpoint(self, global_step, model, optimizer, loss):
        checkpoint = {
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        checkpoint_path = self.exp_path / f"checkpoint_{global_step}.pth"
        torch.save(checkpoint, checkpoint_path)

    def update_best(self, ind, fitness):
        if fitness > self.global_best_fitness:
            self.logger.info(f"Improvement detected: {self.global_best_fitness:.5f} --> {fitness:.5f}")
            final_model_path = self.exp_path / f"model_final.pth"
            torch.save(ind.state_dict(), final_model_path)
            self.global_best_fitness = fitness
