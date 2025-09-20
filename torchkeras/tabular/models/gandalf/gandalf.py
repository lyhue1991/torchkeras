# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
import torch
import torch.nn as nn

from ..common.layers import Add, Embedding1dLayer, GatedFeatureLearningUnit
from ..common.layers.activations import t_softmax
from ..base_model import BaseModel

class GANDALFBackbone(nn.Module):
    def __init__(
        self,
        cat_embedding_dims: list,
        n_continuous_features: int,
        gflu_stages: int,
        gflu_dropout: float = 0.0,
        gflu_feature_init_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        batch_norm_continuous_input: bool = True,
        virtual_batch_size: int = None,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()
        self.gflu_stages = gflu_stages
        self.gflu_dropout = gflu_dropout
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.n_continuous_features = n_continuous_features
        self.cat_embedding_dims = cat_embedding_dims
        self._embedded_cat_features = sum([y for x, y in cat_embedding_dims])
        self.n_features = self._embedded_cat_features + n_continuous_features
        self.embedding_dropout = embedding_dropout
        self.output_dim = self.n_continuous_features + self._embedded_cat_features
        self.gflu_feature_init_sparsity = gflu_feature_init_sparsity
        self.learnable_sparsity = learnable_sparsity
        self.virtual_batch_size = virtual_batch_size
        self._build_network()

    def _build_network(self):
        self.gflus = GatedFeatureLearningUnit(
            n_features_in=self.n_features,
            n_stages=self.gflu_stages,
            feature_mask_function=t_softmax,
            dropout=self.gflu_dropout,
            feature_sparsity=self.gflu_feature_init_sparsity,
            learnable_sparsity=self.learnable_sparsity,
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
        return self.gflus(x)

    @property
    def feature_importance_(self):
        return self.gflus.feature_mask_function(self.gflus.feature_masks).sum(dim=0).detach().cpu().numpy()


class GANDALFModel(BaseModel):
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
        self._backbone = GANDALFBackbone(
            cat_embedding_dims=self.hparams.embedding_dims,
            n_continuous_features=self.hparams.continuous_dim,
            gflu_stages=self.hparams.gflu_stages,
            gflu_dropout=self.hparams.gflu_dropout,
            gflu_feature_init_sparsity=self.hparams.gflu_feature_init_sparsity,
            learnable_sparsity=self.hparams.learnable_sparsity,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
            embedding_dropout=self.hparams.embedding_dropout,
            virtual_batch_size=self.hparams.virtual_batch_size,
        )
        # Embedding Layer
        self._embedding_layer = self._backbone._build_embedding_layer()
        # Head
        self.T0 = nn.Parameter(torch.rand(self.hparams.output_dim), requires_grad=True)
        self._head = nn.Sequential(self._get_head_from_config(), Add(self.T0))

    def data_aware_initialization(self, dataloader):
        if self.hparams.task == "regression":
            batch = next(iter(dataloader))
            self.T0.data = torch.mean(batch["target"], dim=0)
