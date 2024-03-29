import torch
from torchvision import transforms
import numpy as np
import gym
from modules import VAE, MDNRNNCell, NPRNNCell, Controller


class WorldModelAgent:
    def __init__(self, obs_size, latent_size, hidden_size, action_size, discrete_vae):
        self.obs_size = obs_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.discrete_vae = discrete_vae

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((obs_size, obs_size)),
                transforms.ToTensor(),
            ]
        )

        self.V = VAE(latent_size)
        self.M = MDNRNNCell(latent_size + action_size, hidden_size, latent_size, 5)
        self.C = Controller(latent_size, hidden_size, action_size)

    def act(self, obs, hidden):
        with torch.no_grad():
            z = self.V.encode(obs, reparameterize=False)

            if self.discrete_vae:
                z = torch.tanh(z)
                bins = np.array([-1.0, 0.0, 1.0])
                newdata = bins[np.digitize(z, bins[1:])] + 1
                z = torch.from_numpy(newdata).float()

            # steering: real valued in [-1, 1]
            # gas: real valued in [0, 1]
            # break: real valued in [0, 1]
            action = self.C(z, hidden[0])

            rnn_input = torch.cat([z, action], dim=1)
            logpi, mu, sigma, hidden = self.mdnrnn(rnn_input, hidden)
            # NOTE[jinyeom]: the MDN head doesn't do anything...

        return action, hidden

    def rollout(self, time_limit, render=False, early_termination=True):
        env = gym.make("CarRacing-v0", verbose=0)

        obs = env.reset()
        hidden = (
            torch.zeros(1, self.hidden_size),  # h
            torch.zeros(1, self.hidden_size),  # c
        )

        fitness = 0  # reward sum
        neg_count = 0  # for early termination

        t = 0
        done = False

        while not done:
            if render:
                env.render("human")

            obs = self.transform(obs).unsqueeze(0)
            action, hidden = self.get_action(obs, hidden)
            action = action.squeeze().cpu().numpy()

            obs, reward, done, _ = env.step(action)
            fitness += reward
            neg_count = neg_count + 1 if reward < 0 else 0  # only consecutive

            # early termination for speeding up evaluation
            if early_termination and (neg_count > 20 or t > time_limit):
                done = True

            t += 1

        env.close()
        return fitness


class PlasticModelAgent:
    def __init__(self, obs_size, latent_size, hidden_size, action_size, discrete_vae):
        self.obs_size = obs_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.discrete_vae = discrete_vae

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((obs_size, obs_size)),
                transforms.ToTensor(),
            ]
        )

        self.V = VAE(latent_size)
        self.M = NPRNNCell(latent_size + action_size + 1, hidden_size, latent_size, 5)
        self.C = Controller(latent_size, hidden_size, action_size)

    def encode_obs(self, obs):
        obs = self.transform(obs).unsqueeze(0)

        with torch.no_grad():
            latent_mu = self.V.encode(obs, reparameterize=False)

        if self.discrete_vae:
            latent_mu = torch.tanh(latent_mu)
            bins = np.array([-1.0, 0.0, 1.0])
            newdata = bins[np.digitize(latent_mu, bins[1:])] + 1
            latent_mu = torch.from_numpy(newdata).float()

        return latent_mu

    def act_and_predict(self, z, err, h, hebb):
        with torch.no_grad():
            # steering: real valued in [-1, 1]
            # gas: real valued in [0, 1]
            # break: real valued in [0, 1]
            action = self.C(z, h)

            # (encoded observation, action, previous prediction error)
            rnn_input = torch.cat([z, action, err], dim=1)

            pred, h, m, hebb = self.M(rnn_input, h, hebb)

        return action, pred, h, m, hebb

    def rollout(self, time_limit, render=False, early_termination=True):
        env = gym.make("CarRacing-v0", verbose=0)

        obs = env.reset()

        z = self.encode_obs(obs)
        h = torch.zeros(1, self.hidden_size)
        hebb = torch.zeros(1, self.hidden_size, self.hidden_size)

        # "prediction error" from the memory model
        err = torch.zeros(1, 1)

        fitness = 0  # reward sum
        neg_count = 0  # for early termination

        t = 0
        done = False

        while not done:
            if render:
                env.render("human")

            action, pred, h, m, hebb = self.act_and_predict(z, err, h, hebb)
            # if self.discrete_vae:
            #     pred = torch.sigmoid(pred)

            action = action.squeeze().cpu().numpy()
            obs, reward, done, _ = env.step(action)

            z = self.encode_obs(obs)

            # compare the "prediction" and the next observation
            err = torch.mean((pred - z) ** 2, dim=1, keepdim=True)

            fitness += reward
            neg_count = neg_count + 1 if reward < 0 else 0  # only consecutive

            # early termination for speeding up evaluation
            if early_termination and (neg_count > 20 or t > time_limit):
                done = True

            t += 1

        env.close()
        return fitness


class Individual:
    def __init__(self, plastic, mut_mode, mut_pow, **agent_kwargs):
        self.plastic = plastic
        self.mut_mode = mut_mode
        self.mut_pow = mut_pow
        self.agent_kwargs = agent_kwargs

        if self.plastic:
            self.agent = PlasticModelAgent(**agent_kwargs)
        else:
            self.agent = WorldModelAgent(**agent_kwargs)

        self.async_results = []
        self.calculated_results = {}

        self.fitness = None
        self.is_elite = None

    def __repr__(self):
        s = "Individual("
        if self.fitness is not None:
            s += f"fitness={self.fitness}"
        if self.is_elite is not None:
            s += f", is_elite={self.is_elite}"
        s += ")"
        return s

    def run_solution(
        self,
        pool,
        time_limit=1000,
        num_evals=5,
        early_termination=True,
        force_eval=False,
    ):
        if force_eval:
            # remove existing results, so that it can be evaluated again
            self.calculated_results.pop(num_evals, None)

        if num_evals not in self.calculated_results:
            self.async_results = []
            for _ in range(num_evals):
                func = self.agent.rollout
                args = (time_limit, False, early_termination)
                results = pool.apply_async(func, args=args)
                self.async_results.append(results)

    def evaluate_solution(self, num_evals):
        if num_evals in self.calculated_results:
            mean_fitness, std_fitness = self.calculated_results[num_evals]

        else:
            results = [t.get() for t in self.async_results]
            mean_fitness = np.mean(results)
            std_fitness = np.std(results)
            self.calculated_results[num_evals] = (mean_fitness, std_fitness)

        self.fitness = mean_fitness

        return mean_fitness, std_fitness

    def load_solution(self, filename):
        state_dict = torch.load(filename)
        self.agent.V.load_state_dict(state_dict["V"])
        self.agent.M.load_state_dict(state_dict["M"])
        self.agent.C.load_state_dict(state_dict["C"])

    def state_dict(self):
        return {
            "V": self.agent.V.state_dict(),
            "M": self.agent.M.state_dict(),
            "C": self.agent.C.state_dict(),
        }

    def clone(self):
        child = Individual(
            self.plastic, self.mut_mode, self.mut_pow, **self.agent_kwargs
        )

        child.agent.V.load_state_dict(self.agent.V.state_dict())
        child.agent.M.load_state_dict(self.agent.M.state_dict())
        child.agent.C.load_state_dict(self.agent.C.state_dict())

        child.fitness = self.fitness

        return child

    def mutate_params(self, params):
        for key in params:
            noise = np.random.normal(0, 1, params[key].size()) * self.mut_pow
            params[key] += torch.from_numpy(noise).float()

    def mutate(self):
        if self.mut_mode == "MUT-ALL":
            self.mutate_params(self.agent.V.state_dict())
            self.mutate_params(self.agent.M.state_dict())
            self.mutate_params(self.agent.C.state_dict())

        elif self.mut_mode == "MUT-MOD":
            c = np.random.randint(3)
            if c == 0:
                self.mutate_params(self.agent.V.state_dict())
            elif c == 1:
                self.mutate_params(self.agent.M.state_dict())
            else:
                self.mutate_params(self.agent.C.state_dict())

        else:
            raise NotImplementedError
