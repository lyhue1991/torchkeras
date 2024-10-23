# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Category Embedding Model."""
import torch
import torch.nn as nn

from torchkeras.tabular.models.common.layers import Embedding1dLayer
from torchkeras.tabular.utils import _initialize_layers, _linear_dropout_bn

from ..base_model import BaseModel


class CategoryEmbeddingBackbone(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.hparams = config
        self._build_network()

    def _build_network(self):
        # Linear Layers
        layers = []
        if hasattr(self.hparams, "_backbone_input_dim"):
            _curr_units = self.hparams._backbone_input_dim  # TODO implement this backdoor in every model?
        else:
            _curr_units = self.hparams.embedded_cat_dim + self.hparams.continuous_dim
        for units in self.hparams.layers.split("-"):
            layers.extend(
                _linear_dropout_bn(
                    self.hparams.activation,
                    self.hparams.initialization,
                    self.hparams.use_batch_norm,
                    _curr_units,
                    int(units),
                    self.hparams.dropout,
                )
            )
            _curr_units = int(units)
        self.linear_layers = nn.Sequential(*layers)
        _initialize_layers(self.hparams.activation, self.hparams.initialization, self.linear_layers)
        self.output_dim = _curr_units

    def _build_embedding_layer(self):
        return Embedding1dLayer(
            continuous_dim=self.hparams.continuous_dim,
            categorical_embedding_dims=self.hparams.embedding_dims,
            embedding_dropout=self.hparams.embedding_dropout,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
            virtual_batch_size=self.hparams.virtual_batch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_layers(x)
        return x


class CategoryEmbeddingModel(BaseModel):
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
        self._backbone = CategoryEmbeddingBackbone(self.hparams)
        # Embedding Layer
        self._embedding_layer = self._backbone._build_embedding_layer()
        # Head
        self.head = self._get_head_from_config()
