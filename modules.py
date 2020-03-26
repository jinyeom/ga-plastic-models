import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(2 * 2 * 256, latent_size)
        self.fc_logsigma = nn.Linear(2 * 2 * 256, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)
        return mu, logsigma


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x


class VAE(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def encode(self, x, reparameterize=False):
        mu, logsigma = self.encoder(x)
        if reparameterize:
            sigma = logsigma.exp()
            eps = torch.randn_like(sigma)
            z = eps.mul(sigma).add_(mu)
            return z
        return mu

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma


class MDNRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mixture_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mixture_size = mixture_size

        self.rnn = nn.LSTMCell(latent_size + actions, hiddens)

        self.pi = nn.Linear(hidden_size, mixture_size)
        self.mu = nn.Linear(hidden_size, mixture_size * output_size)
        self.logsigma = nn.Linear(hidden_size, mixture_size * output_size)

    def forward(self, x, hidden):
        h, c = self.rnn(x, hidden)

        logpi = F.log_softmax(self.pi(h).view(-1, self.mixture_size), dim=1)
        mu = self.mu(h).view(-1, self.mixture_size, self.output_size)
        sigma = self.logsigma(h).exp().view(-1, self.mixture_size, self.output_size)

        return logpi, mu, sigmas, (h, c)


class Controller(nn.Module):
    def __init__(self, latent_size, hidden_size, action_size):
        super().__init__()
        self.fc = nn.Linear(latent_size + hidden_size, action_size)

    def forward(self, z, h):
        x = torch.cat([z, h], dim=1)
        return self.fc(x)
