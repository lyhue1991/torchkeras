# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""AutomaticFeatureInteraction Config."""
from dataclasses import dataclass, field
from typing import Optional
from torchkeras.tabular.config import ModelConfig

@dataclass
class DANetConfig(ModelConfig):
    """DANet configuration.

    Args:
        n_layers (int): Number of Blocks in the DANet. 8, 20, 32 are configurations
            the paper evaluated. Defaults to 8

        abstlay_dim_1 (int): The dimension for the intermediate output in the
                first ABSTLAY layer in a Block. Defaults to 32

        abstlay_dim_2 (int): The dimension for the intermediate output in the
                second ABSTLAY layer in a Block. Defaults to 64

        k (int): The number of feature groups in the ABSTLAY layer. Defaults to 5

        dropout_rate (float): Dropout to be applied in the Block. Defaults to 0.1

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

    n_layers: int = field(
        default=8,
        metadata={"help": "Number of Blocks in the DANet. Each block has 2 Abstlay Blocks each. Defaults to 8"},
    )

    abstlay_dim_1: int = field(
        default=32,
        metadata={
            "help": "The dimension for the intermediate output in the first ABSTLAY layer in a Block. Defaults to 32"
        },
    )

    abstlay_dim_2: Optional[int] = field(
        default=None,
        metadata={
            "help": "The dimension for the intermediate output in the second ABSTLAY layer in a Block."
            "If None, it will be twice abstlay_dim_1. Defaults to None"
        },
    )
    k: int = field(
        default=5,
        metadata={"help": "The number of feature groups in the ABSTLAY layer. Defaults to 5"},
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout to be applied in the Block. Defaults to 0.1"},
    )
    block_activation: str = field(
        default="LeakyReLU",
        metadata={
            "help": "The activation type in the classification head. The default activation in PyTorch"
            " like ReLU, TanH, LeakyReLU, etc."
            " https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
        },
    )
    virtual_batch_size: Optional[int] = field(
        default=256,
        metadata={
            "help": "If not None, all BatchNorms will be converted to GhostBatchNorm's "
            " with this virtual batch size. Defaults to None"
        },
    )

    _module_src: str = field(default="models.danet")
    _model_name: str = field(default="DANetModel")
    _backbone_name: str = field(default="DANetBackbone")
    _config_name: str = field(default="DANetConfig")

    def __post_init__(self):
        if self.abstlay_dim_2 is None:
            self.abstlay_dim_2 = self.abstlay_dim_1 * 2
        return super().__post_init__()


