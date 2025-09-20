# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Category Embedding Model Config."""
from dataclasses import dataclass, field

from torchkeras.tabular.config import ModelConfig


@dataclass
class CategoryEmbeddingModelConfig(ModelConfig):
    """CategoryEmbeddingModel configuration.

    Args:
        layers (str): DEPRECATED: Hyphen-separated number of layers and units in the classification head. E.g. 32-64-32.
                Defaults to 128-64-32

        activation (str): DEPRECATED: The activation type in the classification head. The default activation in PyTorch
                like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
                Defaults to ReLU

        use_batch_norm (bool): DEPRECATED: Flag to include a BatchNorm layer after each Linear Layer+DropOut. Defaults
                to False

        initialization (str): DEPRECATED: Initialization scheme for the linear layers. Defaults to `kaiming`. Choices
                are: [`kaiming`,`xavier`,`random`].

        dropout (float): DEPRECATED: probability of a classification element to be zeroed. This is added to each
                linear layer. Defaults to 0.0


        task (str): Specify whether the problem is regression or classification. `backbone` is a task which
                considers the model as a backbone to generate features. Mostly used internally for SSL and related
                tasks. Choices are: [`regression`,`classification`,`backbone`].

        head (Optional[str]): The head to be used for the model. Should be one of the heads defined in
                `pytorch_tabular.models.common.heads`. Defaults to  LinearHead. Choices are:
                [`None`,`LinearHead`,`MixtureDensityHead`].

        head_config (Optional[Dict]): The config as a dict which defines the head. If left empty, will be
                initialized as default linear head.

        embedding_dims (Optional[List]): The dimensions of the embedding for each categorical column as a
                list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of
                the categorical column using the rule min(50, (x + 1) // 2)

        embedding_dropout (float): Dropout to be applied to the Categorical Embedding. Defaults to 0.0

        batch_norm_continuous_input (bool): If True, we will normalize the continuous layer by passing it
                through a BatchNorm layer.

        loss (Optional[str]): The loss function to be applied. By Default, it is MSELoss for regression and
                CrossEntropyLoss for classification. Unless you are sure what you are doing, leave it at MSELoss
                or L1Loss for regression and CrossEntropyLoss for classification

    """

    layers: str = field(
        default="128-64-32",
        metadata={
            "help": (
                "Hyphen-separated number of layers and units in the classification"
                " head. eg. 32-64-32. Defaults to 128-64-32"
            )
        },
    )
    activation: str = field(
        default="ReLU",
        metadata={
            "help": (
                "The activation type in the classification head. The default"
                " activation in PyTorch like ReLU, TanH, LeakyReLU, etc."
                " https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity."
                " Defaults to ReLU"
            )
        },
    )
    use_batch_norm: bool = field(
        default=False,
        metadata={"help": ("Flag to include a BatchNorm layer after each Linear Layer+DropOut." " Defaults to False")},
    )
    initialization: str = field(
        default="kaiming",
        metadata={
            "help": ("Initialization scheme for the linear layers. Defaults to `kaiming`"),
            "choices": ["kaiming", "xavier", "random"],
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={
            "help": (
                "probability of an classification element to be zeroed."
                " This is added to each linear layer. Defaults to 0.0"
            )
        },
    )

    # def __post_init__(self):
    #     deprecated_args = [
    #         "layers",
    #         "activation",
    #         "use_batch_norm",
    #         "initialization",
    #         "dropout",
    #     ]
    #     # for arg in deprecated_args:
    #     if any([getattr(self, arg) is not None for arg in deprecated_args]):
    #         warnings.warn(
    #             f"{deprecated_args} are deprecated and will be remoevd in next version. "
    #             "Please use 'head' and `head_config` and set deprecated args "
    #             "to `None` to turn off warning. CategoricalEmbedding model is just a "
    #             "linear head with embedding layers."
    #         )
    #     return super().__post_init__()

    _module_src: str = field(default="models.category_embedding")
    _model_name: str = field(default="CategoryEmbeddingModel")
    _backbone_name: str = field(default="CategoryEmbeddingBackbone")
    _config_name: str = field(default="CategoryEmbeddingModelConfig")

