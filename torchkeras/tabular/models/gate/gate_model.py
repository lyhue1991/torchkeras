# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
import torch
import torch.nn as nn

from ..common.heads import blocks
from ..common.layers import Add, Embedding1dLayer, GatedFeatureLearningUnit, NeuralDecisionTree
from ..common.layers.activations import entmax15, entmoid15, sparsemax, sparsemoid, t_softmax

from ..base_model import BaseModel

class GatedAdditiveTreesBackbone(nn.Module):
    ACTIVATION_MAP = {
        "entmax": entmax15,
        "sparsemax": sparsemax,
        "softmax": nn.functional.softmax,
        "t-softmax": t_softmax,
    }

    BINARY_ACTIVATION_MAP = {
        "entmoid": entmoid15,
        "sparsemoid": sparsemoid,
        "sigmoid": nn.functional.sigmoid,
    }

    def __init__(
        self,
        cat_embedding_dims: list,
        n_continuous_features: int,
        gflu_stages: int,
        num_trees: int,
        tree_depth: int,
        chain_trees: bool = True,
        tree_wise_attention: bool = False,
        tree_wise_attention_dropout: float = 0.0,
        gflu_dropout: float = 0.0,
        tree_dropout: float = 0.0,
        binning_activation: str = "entmoid",
        feature_mask_function: str = "softmax",
        gflu_feature_init_sparsity: float = 0.3,
        tree_feature_init_sparsity: float = 0.8,
        learnable_sparsity: bool = True,
        batch_norm_continuous_input: bool = True,
        virtual_batch_size: int = None,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()
        assert (
            binning_activation in self.BINARY_ACTIVATION_MAP.keys()
        ), f"`binning_activation should be one of {self.BINARY_ACTIVATION_MAP.keys()}"
        assert (
            feature_mask_function in self.ACTIVATION_MAP.keys()
        ), f"`feature_mask_function should be one of {self.ACTIVATION_MAP.keys()}"

        self.gflu_stages = gflu_stages
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.chain_trees = chain_trees
        self.tree_wise_attention = tree_wise_attention
        self.tree_wise_attention_dropout = tree_wise_attention_dropout
        self.gflu_dropout = gflu_dropout
        self.tree_dropout = tree_dropout
        self.binning_activation = self.BINARY_ACTIVATION_MAP[binning_activation]
        self.feature_mask_function = self.ACTIVATION_MAP[feature_mask_function]
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.n_continuous_features = n_continuous_features
        self.cat_embedding_dims = cat_embedding_dims
        self._embedded_cat_features = sum([y for x, y in cat_embedding_dims])
        self.n_features = self._embedded_cat_features + n_continuous_features
        self.embedding_dropout = embedding_dropout
        self.output_dim = 2**self.tree_depth if self.num_trees > 0 else self.n_features
        self.gflu_feature_init_sparsity = gflu_feature_init_sparsity
        self.tree_feature_init_sparsity = tree_feature_init_sparsity
        self.learnable_sparsity = learnable_sparsity
        self.virtual_batch_size = virtual_batch_size
        self._build_network()

    def _build_network(self):
        if self.gflu_stages > 0:
            self.gflus = GatedFeatureLearningUnit(
                n_features_in=self.n_features,
                n_stages=self.gflu_stages,
                feature_mask_function=self.feature_mask_function,
                dropout=self.gflu_dropout,
                feature_sparsity=self.gflu_feature_init_sparsity,
                learnable_sparsity=self.learnable_sparsity,
            )
        if self.num_trees > 0:
            self.trees = nn.ModuleList(
                [
                    NeuralDecisionTree(
                        depth=self.tree_depth,
                        n_features=self.n_features + 2**self.tree_depth * t if self.chain_trees else self.n_features,
                        dropout=self.tree_dropout,
                        binning_activation=self.binning_activation,
                        feature_mask_function=self.feature_mask_function,
                        feature_sparsity=self.tree_feature_init_sparsity,
                        learnable_sparsity=self.learnable_sparsity,
                    )
                    for t in range(self.num_trees)
                ]
            )
            if self.tree_wise_attention:
                self.tree_attention = nn.MultiheadAttention(
                    embed_dim=self.output_dim,
                    num_heads=1,
                    batch_first=False,
                    dropout=self.tree_wise_attention_dropout,
                )

    def _build_embedding_layer(self):
        return Embedding1dLayer(
            continuous_dim=self.n_continuous_features,
            categorical_embedding_dims=self.cat_embedding_dims,
            embedding_dropout=self.embedding_dropout,
            batch_norm_continuous_input=self.batch_norm_continuous_input,
            virtual_batch_size=self.virtual_batch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gflu_stages > 0:
            x = self.gflus(x)
        if self.num_trees > 0:
            # Decision Tree
            tree_outputs = []
            tree_feature_masks = []  # TODO make this optional and create feat importance
            tree_input = x
            for i in range(self.num_trees):
                tree_output, feat_masks = self.trees[i](tree_input)
                tree_outputs.append(tree_output.unsqueeze(-1))
                tree_feature_masks.append(feat_masks)
                if self.chain_trees:
                    tree_input = torch.cat([tree_input, tree_output], 1)
            tree_outputs = torch.cat(tree_outputs, dim=-1)
            if self.tree_wise_attention:
                tree_outputs = tree_outputs.permute(2, 0, 1)
                tree_outputs, _ = self.tree_attention(tree_outputs, tree_outputs, tree_outputs)
                tree_outputs = tree_outputs.permute(1, 2, 0)
            return tree_outputs
        else:
            return x

    @property
    def feature_importance_(self):
        return self.gflus.feature_mask_function(self.gflus.feature_masks).sum(dim=0).detach().cpu().numpy()


class CustomHead(nn.Module):
    """Custom Head for GATE.

    """

    def __init__(self, input_dim: int, hparams):
        super().__init__()
        self.hparams = hparams
        self.input_dim = input_dim
        if self.hparams.share_head_weights:
            self.head = self._get_head_from_config()
        else:
            self.head = nn.ModuleList([self._get_head_from_config() for _ in range(self.hparams.num_trees)])
        # random parameter with num_trees elements
        self.eta = nn.Parameter(torch.rand(self.hparams.num_trees, requires_grad=True))
        if self.hparams.task == "regression":
            self.T0 = nn.Parameter(torch.rand(self.hparams.output_dim), requires_grad=True)

    def _get_head_from_config(self):
        _head_callable = getattr(blocks, self.hparams.head)
        return _head_callable(
            in_units=self.input_dim,
            output_dim=self.hparams.output_dim,
            config=_head_callable._config_template(**self.hparams.head_config),
        )  # output_dim auto-calculated from other configs

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        # B x L x T
        if not self.hparams.share_head_weights:
            # B x T X Output
            y_hat = torch.cat(
                [h(backbone_features[:, :, i]).unsqueeze(1) for i, h in enumerate(self.head)],
                dim=1,
            )
        else:
            # https://discuss.pytorch.org/t/how-to-pass-a-3d-tensor-to-linear-layer/908/6
            # B x T x L -> B x T x Output
            y_hat = self.head(backbone_features.transpose(2, 1))

        # applying weights to each tree and summing up
        # ETA
        y_hat = y_hat * self.eta.reshape(1, -1, 1)
        # summing up
        y_hat = y_hat.sum(dim=1)

        if self.hparams.task == "regression":
            y_hat = y_hat + self.T0
        return y_hat


class GatedAdditiveTreeEnsembleModel(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    @property
    def backbone(self):
        return self._backbone

    @property
    def embedding_layer(self):
        return self._embedding_layer

    @property
    def head(self):
        return self._head

    def _build_network(self):
        # Backbone
        self._backbone = GatedAdditiveTreesBackbone(
            n_continuous_features=self.hparams.continuous_dim,
            cat_embedding_dims=self.hparams.embedding_dims,
            gflu_stages=self.hparams.gflu_stages,
            gflu_dropout=self.hparams.gflu_dropout,
            num_trees=self.hparams.num_trees,
            tree_depth=self.hparams.tree_depth,
            tree_dropout=self.hparams.tree_dropout,
            binning_activation=self.hparams.binning_activation,
            feature_mask_function=self.hparams.feature_mask_function,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
            chain_trees=self.hparams.chain_trees,
            tree_wise_attention=self.hparams.tree_wise_attention,
            tree_wise_attention_dropout=self.hparams.tree_wise_attention_dropout,
            gflu_feature_init_sparsity=self.hparams.gflu_feature_init_sparsity,
            tree_feature_init_sparsity=self.hparams.tree_feature_init_sparsity,
            virtual_batch_size=self.hparams.virtual_batch_size,
        )
        # Embedding Layer
        self._embedding_layer = self._backbone._build_embedding_layer()
        # Head
        if self.hparams.num_trees == 0:
            self.T0 = nn.Parameter(torch.rand(self.hparams.output_dim), requires_grad=True)
            self._head = nn.Sequential(self._get_head_from_config(), Add(self.T0))
        else:
            self._head = CustomHead(self.backbone.output_dim, self.hparams)

    def data_aware_initialization(self, dataloader):
        if self.hparams.task == "regression":
            batch = next(iter(dataloader))
            t0 = torch.mean(batch["target"], dim=0)
            if self.hparams.num_trees != 0:
                self.head.T0.data = t0
            else:
                self.T0.data = t0
