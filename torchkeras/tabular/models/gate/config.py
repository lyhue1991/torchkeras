# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""GatedAdditiveTreeEnsembleConfig Config."""
from dataclasses import dataclass, field

from torchkeras.tabular.config import ModelConfig

@dataclass
class GatedAdditiveTreeEnsembleConfig(ModelConfig):
    """Gated Additive Tree Ensemble configuration.

    Args:
        gflu_stages (int): Number of layers in the feature abstraction layer. Defaults to 6

        gflu_dropout (float): Dropout rate for the feature abstraction layer. Defaults to 0.0

        tree_depth (int): Depth of the tree. Defaults to 5

        num_trees (int): Number of trees to use in the ensemble. Defaults to 20

        binning_activation (str): The binning function to use. Defaults to entmoid. Defaults to sparsemoid.
                Choices are: [`entmoid`,`sparsemoid`,`sigmoid`].

        feature_mask_function (str): The feature mask function to use. Defaults to sparsemax. Choices are:
                [`entmax`,`sparsemax`,`softmax`].

        tree_dropout (float): probability of dropout in tree binning transformation. Defaults to 0.0

        chain_trees (bool): If True, we will chain the trees together. Synonymous to boosting
            (chaining trees) or bagging (parallel trees). Defaults to True

        tree_wise_attention (bool): If True, we will use tree wise attention to combine trees. Defaults to
                True

        tree_wise_attention_dropout (float): probability of dropout in the tree wise attention layer.
                Defaults to 0.0

        share_head_weights (bool): If True, we will share the weights between the heads. Defaults to True


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

    gflu_stages: int = field(
        default=6,
        metadata={"help": "Number of layers in the feature abstraction layer. Defaults to 6"},
    )

    gflu_dropout: float = field(
        default=0.0, metadata={"help": "Dropout rate for the feature abstraction layer. Defaults to 0.0"}
    )

    tree_depth: int = field(default=4, metadata={"help": "Depth of the tree. Defaults to 5"})

    num_trees: int = field(
        default=10,
        metadata={"help": "Number of trees to use in the ensemble. Defaults to 20"},
    )

    binning_activation: str = field(
        default="sparsemoid",
        metadata={
            "help": "The binning function to use. Defaults to entmoid. Defaults to entmoid",
            "choices": ["entmoid", "sparsemoid", "sigmoid"],
        },
    )
    feature_mask_function: str = field(
        default="t-softmax",
        metadata={
            "help": "The feature mask function to use. Defaults to entmax",
            "choices": ["entmax", "sparsemax", "softmax", "t-softmax"],
        },
    )
    gflu_feature_init_sparsity: float = field(
        default=0.3,
        metadata={
            "help": "Only valid for t-softmax. The percentage of features to be dropped in "
            "each GFLU stage. This is just initialized and during learning it may change"
        },
    )
    tree_feature_init_sparsity: float = field(
        default=0.8,
        metadata={
            "help": "Only valid for t-softmax. The perecentge of features to be dropped in "
            "each split in the tree. This is just initialized and during learning it may change"
        },
    )
    learnable_sparsity: bool = field(
        default=True,
        metadata={
            "help": "Only valid for t-softmax. If True, the sparsity parameters will be learned."
            "If False, the sparsity parameters will be fixed to the initial values specified in "
            "`gflu_feature_init_sparsity` and `tree_feature_init_sparsity`"
        },
    )

    tree_dropout: float = field(
        default=0.0,
        metadata={"help": "probability of dropout in tree binning transformation. Defaults to 0.0"},
    )
    chain_trees: bool = field(
        default=True,
        metadata={
            "help": "If True, we will chain the trees together."
            " Synonymous to boosting (chaining trees) or bagging (parallel trees). Defaults to True"
        },
    )
    tree_wise_attention: bool = field(
        default=True,
        metadata={"help": "If True, we will use tree wise attention to combine trees. Defaults to True"},
    )
    tree_wise_attention_dropout: float = field(
        default=0.0,
        metadata={"help": "probability of dropout in the tree wise attention layer. Defaults to 0.0"},
    )
    share_head_weights: bool = field(
        default=True,
        metadata={"help": "If True, we will share the weights between the heads. Defaults to True"},
    )

    _module_src: str = field(default="models.gate")
    _model_name: str = field(default="GatedAdditiveTreeEnsembleModel")
    _backbone_name: str = field(default="GatedAdditiveTreesBackbone")
    _config_name: str = field(default="GatedAdditiveTreeEnsembleConfig")

    def __post_init__(self):
        assert self.tree_depth > 0, "tree_depth should be greater than 0"
        # Either gflu_stages or num_trees should be greater than 0
        assert self.num_trees > 0, (
            "`num_trees` must be greater than 0." "If you want a lighter model which performs better, use GANDALF."
        )
        super().__post_init__()

