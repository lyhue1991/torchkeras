from dataclasses import dataclass, field
from typing import List, Optional

# from typing import Any, Dict, Iterable, List, Optional


@dataclass
class LinearHeadConfig:
    """A model class for Linear Head configuration; serves as a template and documentation. The models take a
    dictionary as input, but if there are keys which are not present in this model class, it'll throw an exception.

    Args:
        layers (str): Hyphen-separated number of layers and units in the classification/regression head.
                E.g. 32-64-32. Default is just a mapping from intput dimension to output dimension

        activation (str): The activation type in the classification head. The default activation in PyTorch
                like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

        dropout (float): probability of a classification element to be zeroed.

        use_batch_norm (bool): Flag to include a BatchNorm layer after each Linear Layer+DropOut

        initialization (str): Initialization scheme for the linear layers. Defaults to `kaiming`. Choices
                are: [`kaiming`,`xavier`,`random`].

    """

    layers: str = field(
        default="",
        metadata={
            "help": "Hyphen-separated number of layers and units in the classification/regression head. eg. 32-64-32."
            " Default is just a mapping from intput dimension to output dimension"
        },
    )
    activation: str = field(
        default="ReLU",
        metadata={
            "help": "The activation type in the classification head. The default activation in PyTorch"
            " like ReLU, TanH, LeakyReLU, etc."
            " https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "probability of an classification element to be zeroed."},
    )
    use_batch_norm: bool = field(
        default=False,
        metadata={"help": "Flag to include a BatchNorm layer after each Linear Layer+DropOut"},
    )
    initialization: str = field(
        default="kaiming",
        metadata={
            "help": "Initialization scheme for the linear layers. Defaults to `kaiming`",
            "choices": ["kaiming", "xavier", "random"],
        },
    )


@dataclass
class MixtureDensityHeadConfig:
    """MixtureDensityHead configuration.

    Args:
        num_gaussian (int): Number of Gaussian Distributions in the mixture model. Defaults to 1

        sigma_bias_flag (bool): Whether to have a bias term in the sigma layer. Defaults to False

        mu_bias_init (Optional[List]): To initialize the bias parameter of the mu layer to predefined
                cluster centers. Should be a list with the same length as number of gaussians in the mixture
                model. It is highly recommended to set the parameter to combat mode collapse. Defaults to None

        weight_regularization (Optional[int]): Whether to apply L1 or L2 Norm to the MDN layers. Defaults
                to L2. Choices are: [`1`,`2`].

        lambda_sigma (Optional[float]): The regularization constant for weight regularization of sigma
                layer. Defaults to 0.1

        lambda_pi (Optional[float]): The regularization constant for weight regularization of pi layer.
                Defaults to 0.1

        lambda_mu (Optional[float]): The regularization constant for weight regularization of mu layer.
                Defaults to 0

        softmax_temperature (Optional[float]): The temperature to be used in the gumbel softmax of the
                mixing coefficients. Values less than one leads to sharper transition between the multiple
                components. Defaults to 1

        n_samples (int): Number of samples to draw from the posterior to get prediction. Defaults to 100

        central_tendency (str): Which measure to use to get the point prediction. Defaults to mean. Choices
                are: [`mean`,`median`].

        speedup_training (bool): Turning on this parameter does away with sampling during training which
                speeds up training, but also doesn't give you visibility on train metrics. Defaults to False

        log_debug_plot (bool): Turning on this parameter plots histograms of the mu, sigma, and pi layers
                in addition to the logits(if log_logits is turned on in experment config). Defaults to False

        input_dim (int): The input dimensions to the head. This will be automatically filled in while
                initializing from the `backbone.output_dim`

    """

    num_gaussian: int = field(
        default=1,
        metadata={
            "help": "Number of Gaussian Distributions in the mixture model. Defaults to 1",
        },
    )
    sigma_bias_flag: bool = field(
        default=False,
        metadata={
            "help": "Whether to have a bias term in the sigma layer. Defaults to False",
        },
    )
    mu_bias_init: Optional[List] = field(
        default=None,
        metadata={
            "help": "To initialize the bias parameter of the mu layer to predefined cluster centers."
            " Should be a list with the same length as number of gaussians in the mixture model."
            " It is highly recommended to set the parameter to combat mode collapse. Defaults to None",
        },
    )

    weight_regularization: Optional[int] = field(
        default=2,
        metadata={
            "help": "Whether to apply L1 or L2 Norm to the MDN layers. Defaults to L2",
            "choices": [1, 2],
        },
    )

    lambda_sigma: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The regularization constant for weight regularization of sigma layer. Defaults to 0.1",
        },
    )
    lambda_pi: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The regularization constant for weight regularization of pi layer. Defaults to 0.1",
        },
    )
    lambda_mu: Optional[float] = field(
        default=0,
        metadata={
            "help": "The regularization constant for weight regularization of mu layer. Defaults to 0",
        },
    )
    softmax_temperature: Optional[float] = field(
        default=1,
        metadata={
            "help": "The temperature to be used in the gumbel softmax of the mixing coefficients."
            " Values less than one leads to sharper transition between the multiple components. Defaults to 1",
        },
    )
    n_samples: int = field(
        default=100,
        metadata={
            "help": "Number of samples to draw from the posterior to get prediction. Defaults to 100",
        },
    )
    central_tendency: str = field(
        default="mean",
        metadata={
            "help": "Which measure to use to get the point prediction. Defaults to mean",
            "choices": ["mean", "median"],
        },
    )
    speedup_training: bool = field(
        default=False,
        metadata={
            "help": "Turning on this parameter does away with sampling during training which speeds up training,"
            " but also doesn't give you visibility on train metrics. Defaults to False",
        },
    )
    log_debug_plot: bool = field(
        default=False,
        metadata={
            "help": "Turning on this parameter plots histograms of the mu, sigma, and pi layers in addition"
            " to the logits(if log_logits is turned on in experment config). Defaults to False",
        },
    )
    input_dim: int = field(
        default=None,
        metadata={
            "help": "The input dimensions to the head. This will be automatically filled in while initializing"
            " from the `backbone.output_dim`",
        },
    )
    _probabilistic: bool = field(default=True)


# if __name__ == "__main__":
#     from pytorch_tabular.utils import generate_doc_dataclass
#     print(generate_doc_dataclass(MixtureDensityHeadConfig))
