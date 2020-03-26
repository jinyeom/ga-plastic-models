import copy
import torch
from torchvision import transforms
import numpy as np
import gym
from modules import VAE, MDNRNNCell, Controller

ASIZE = 3  # action size
LSIZE = 32  # latent size (128 for discrete)
RSIZE = 256  # hidden size
RED_SIZE = 64  # obs size
SIZE = 64  #

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((RED_SIZE, RED_SIZE)),
        transforms.ToTensor(),
    ]
)


class RolloutGenerator:
    def __init__(self, time_limit, discrete_VAE):
        self.time_limit = time_limit
        self.discrete_VAE = discrete_VAE

        self.env = gym.make("CarRacing-v0")

        # Because the represenation is discrete, we increase the size of the latent vector
        if self.discrete_VAE:
            LSIZE = 128

        self.vae = VAE(LSIZE)
        self.mdnrnn = MDNRNNCell(LSIZE + ASIZE, RSIZE, LSIZE, 5)
        self.controller = Controller(LSIZE, RSIZE, ASIZE)

    def get_action_and_transition(self, obs, hidden):
        latent_mu = self.vae.encode(obs, reparameterize=False)

        if self.discrete_VAE:
            latent_mu = torch.tanh(latent_mu)
            bins = np.array([-1.0, 0.0, 1.0])
            newdata = bins[np.digitize(latent_mu, bins[1:])] + 1
            latent_mu = torch.from_numpy(newdata).float()

        action = self.controller(latent_mu, hidden[0])

        rnn_input = torch.cat([latent_mu, action], dim=1)
        logpi, mu, sigma, hidden = self.mdnrnn(rnn_input, hidden)

        return action.squeeze().cpu().numpy(), hidden

    def rollout(self, render=False, early_termination=True):
        self.env = gym.make("CarRacing-v0")

        obs = self.env.reset()
        hidden = [torch.zeros(1, RSIZE) for _ in range(2)]  # (h, c)

        t = 0
        neg_count = 0
        fitness = 0
        done = False

        self.env.render("rgb_array")

        while not done:
            if render:
                self.env.render("human")

            obs = transform(obs).unsqueeze(0)

            with torch.no_grad():
                # steering: real valued in [-1, 1]
                # gas: real valued in [0, 1]
                # break: real valued in [0, 1]
                action, hidden = self.get_action_and_transition(obs, hidden)

            obs, reward, done, _ = self.env.step(action)
            fitness += reward

            # count how many times the car did not get a reward
            # (e.g. was outside track)
            if reward < 0:
                neg_count += 1

            # early termination for speeding up evaluation
            if early_termination and (neg_count > 20 or t > self.time_limit):
                done = True

            t += 1

        self.env.close()

        return fitness, None


def fitness_eval_parallel(pool, r_gen, early_termination=True):
    return pool.apply_async(r_gen.rollout, args=(False, early_termination))


class GAIndividual:
    def __init__(self, time_limit, setting, multi=True, discrete_VAE=False):
        self.time_limit = time_limit
        self.setting = setting
        self.multi = multi
        self.discrete_VAE = discrete_VAE
        self.mutation_power = 0.01

        self.r_gen = RolloutGenerator(time_limit, discrete_VAE)

        self.async_results = []
        self.calculated_results = {}

    def run_solution(self, pool, evals=5, early_termination=True, force_eval=False):
        if force_eval:
            self.calculated_results.pop(evals, None)

        if evals in self.calculated_results:  # already calculated results
            return

        self.async_results = []

        for i in range(evals):
            if self.multi:
                results = fitness_eval_parallel(pool, self.r_gen, early_termination)
            else:
                results = self.r_gen.rollout(False, early_termination)

            self.async_results.append(results)

    def evaluate_solution(self, evals):
        if evals in self.calculated_results:  # already calculated?
            mean_fitness, std_fitness = self.calculated_results[evals]
        else:
            if self.multi:
                results = [t.get()[0] for t in self.async_results]
            else:
                results = [t[0] for t in self.async_results]

            mean_fitness = np.mean(results)
            std_fitness = np.std(results)

            self.calculated_results[evals] = (mean_fitness, std_fitness)

        self.fitness = -mean_fitness

        return mean_fitness, std_fitness

    def load_solution(self, filename):
        s = torch.load(filename)

        self.r_gen.vae.load_state_dict(s["vae"])
        self.r_gen.controller.load_state_dict(s["controller"])
        self.r_gen.mdnrnn.load_state_dict(s["mdnrnn"])

    def clone_individual(self):
        child_solution = GAIndividual(
            self.time_limit,
            self.setting,
            multi=self.multi,
            discrete_VAE=self.discrete_VAE,
        )

        child_solution.fitness = self.fitness
        child_solution.r_gen.controller = copy.deepcopy(self.r_gen.controller)
        child_solution.r_gen.vae = copy.deepcopy(self.r_gen.vae)
        child_solution.r_gen.mdrnn = copy.deepcopy(self.r_gen.mdnrnn)

        return child_solution

    def mutate_params(self, params):
        for key in params:
            params[key] += torch.from_numpy(
                np.random.normal(0, 1, params[key].size()) * self.mutation_power
            ).float()

    def mutate(self):
        if self.setting == 0:  # MUT-ALL
            self.mutate_params(self.r_gen.controller.state_dict())
            self.mutate_params(self.r_gen.vae.state_dict())
            self.mutate_params(self.r_gen.mdnrnn.state_dict())

        if self.setting == 1:  # MUT-MOD
            c = np.random.randint(3)
            if c == 0:
                self.mutate_params(self.r_gen.vae.state_dict())
            elif c == 1:
                self.mutate_params(self.r_gen.mdnrnn.state_dict())
            else:
                self.mutate_params(self.r_gen.controller.state_dict())
