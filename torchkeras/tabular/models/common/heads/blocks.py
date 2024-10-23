import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions import Categorical

from torchkeras.tabular.models.common.heads import config as head_config
from torchkeras.tabular.utils import _initialize_layers, _linear_dropout_bn

def config_link(r):
    """This is a helper function decorator to link the config to the head."""

    def wrapper(f):
        f.config_template = r
        return f

    return wrapper


class Head(nn.Module):
    def __init__(self, layers, config_template, **kwargs):
        super().__init__()
        self.layers = layers
        self._config_template = config_template

    def forward(self, x):
        return self.layers(x)


class LinearHead(Head):
    _config_template = head_config.LinearHeadConfig

    def __init__(self, in_units: int, output_dim: int, config, **kwargs):
        # Linear Layers
        _layers = []
        _curr_units = in_units
        for units in config.layers.split("-"):
            try:
                int(units)
            except ValueError:
                if units == "":
                    continue
                else:
                    raise ValueError(f"Invalid units {units} in layers {config.layers}")
            _layers.extend(
                _linear_dropout_bn(
                    config.activation,
                    config.initialization,
                    config.use_batch_norm,
                    _curr_units,
                    int(units),
                    config.dropout,
                )
            )
            _curr_units = int(units)
        # Appending Final Output
        _layers.append(nn.Linear(_curr_units, output_dim))
        linear_layers = nn.Sequential(*_layers)
        _initialize_layers(config.activation, config.initialization, linear_layers)
        super().__init__(
            layers=linear_layers,
            config_template=head_config.LinearHeadConfig,
        )


ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
LOG2PI = math.log(2 * math.pi)


class MixtureDensityHead(nn.Module):
    _config_template = head_config.MixtureDensityHeadConfig

    def __init__(self, config, **kwargs):
        self.hparams = config
        super().__init__()
        self._build_network()

    def _build_network(self):
        self.pi = nn.Linear(self.hparams.input_dim, self.hparams.num_gaussian)
        nn.init.normal_(self.pi.weight)
        self.sigma = nn.Linear(
            self.hparams.input_dim,
            self.hparams.num_gaussian,
            bias=self.hparams.sigma_bias_flag,
        )
        self.mu = nn.Linear(self.hparams.input_dim, self.hparams.num_gaussian)
        nn.init.normal_(self.mu.weight)
        if self.hparams.mu_bias_init is not None:
            for i, bias in enumerate(self.hparams.mu_bias_init):
                nn.init.constant_(self.mu.bias[i], bias)

    def forward(self, x):
        pi = self.pi(x)
        sigma = self.sigma(x)
        # Applying modified ELU activation
        sigma = nn.ELU()(sigma) + 1 + 1e-15
        mu = self.mu(x)
        return pi, sigma, mu

    def gaussian_probability(self, sigma, mu, target, log=False):
        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

        Arguments:
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
                size, G is the number of Gaussians, and O is the number of
                dimensions per Gaussian.
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
                number of Gaussians, and O is the number of dimensions per Gaussian.
            target (BxI): A batch of target. B is the batch size and I is the number of
                input dimensions.
        Returns:
            probabilities (BxG): The probability of each point in the probability
                of the distribution in the corresponding sigma/mu index.

        """
        target = target.expand_as(sigma)
        if log:
            ret = -torch.log(sigma) - 0.5 * LOG2PI - 0.5 * torch.pow((target - mu) / sigma, 2)
        else:
            ret = (ONEOVERSQRT2PI / sigma) * torch.exp(-0.5 * ((target - mu) / sigma) ** 2)
        return ret  # torch.prod(ret, 2)

    def log_prob(self, pi, sigma, mu, y):
        log_component_prob = self.gaussian_probability(sigma, mu, y, log=True)
        log_mix_prob = torch.log(nn.functional.gumbel_softmax(pi, tau=self.hparams.softmax_temperature, dim=-1) + 1e-15)
        return torch.logsumexp(log_component_prob + log_mix_prob, dim=-1)

    def sample(self, pi, sigma, mu):
        """Draw samples from a MoG."""
        categorical = Categorical(pi)
        pis = categorical.sample().unsqueeze(1)
        sample = Variable(sigma.data.new(sigma.size(0), 1).normal_())
        # Gathering from the n Gaussian Distribution based on sampled indices
        sample = sample * sigma.gather(1, pis) + mu.gather(1, pis)
        return sample

    def generate_samples(self, pi, sigma, mu, n_samples=None):
        if n_samples is None:
            n_samples = self.hparams.n_samples
        samples = []
        softmax_pi = nn.functional.gumbel_softmax(pi, tau=self.hparams.softmax_temperature, dim=-1)
        assert (softmax_pi < 0).sum().item() == 0, "pi parameter should not have negative"
        for _ in range(n_samples):
            samples.append(self.sample(softmax_pi, sigma, mu))
        samples = torch.cat(samples, dim=1)
        return samples

    def generate_point_predictions(self, pi, sigma, mu, n_samples=None):
        # Sample using n_samples and take average
        samples = self.generate_samples(pi, sigma, mu, n_samples)
        if self.hparams.central_tendency == "mean":
            y_hat = torch.mean(samples, dim=-1)
        elif self.hparams.central_tendency == "median":
            y_hat = torch.median(samples, dim=-1).values
        return y_hat.unsqueeze(1)

