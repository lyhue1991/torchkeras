# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
# Inspired by https://github.com/rixwew/pytorch-fm/blob/master/torchfm/model/afi.py
"""AutomaticFeatureInteraction Model."""
import torch
import torch.nn as nn

from ..common.layers import Embedding2dLayer
from torchkeras.tabular.utils import _initialize_layers, _linear_dropout_bn
from ..base_model import BaseModel

class AutoIntBackbone(nn.Module):
    def __init__(self, config):
        """Automatic Feature Interaction Network.

        """
        super().__init__()
        self.hparams = config
        self._build_network()

    def _build_network(self):
        # Deep Layers
        _curr_units = self.hparams.embedding_dim
        if self.hparams.deep_layers:
            # Linear Layers
            layers = []
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
        # Projection to Multi-Headed Attention Dims
        self.attn_proj = nn.Linear(_curr_units, self.hparams.attn_embed_dim)
        _initialize_layers(self.hparams.activation, self.hparams.initialization, self.attn_proj)
        # Multi-Headed Attention Layers
        self.self_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.hparams.attn_embed_dim,
                    self.hparams.num_heads,
                    dropout=self.hparams.attn_dropouts,
                )
                for _ in range(self.hparams.num_attn_blocks)
            ]
        )
        if self.hparams.has_residuals:
            self.V_res_embedding = torch.nn.Linear(
                _curr_units,
                (
                    self.hparams.attn_embed_dim * self.hparams.num_attn_blocks
                    if self.hparams.attention_pooling
                    else self.hparams.attn_embed_dim
                ),
            )
        self.output_dim = (self.hparams.continuous_dim + self.hparams.categorical_dim) * self.hparams.attn_embed_dim
        if self.hparams.attention_pooling:
            self.output_dim = self.output_dim * self.hparams.num_attn_blocks

    def _build_embedding_layer(self):
        return Embedding2dLayer(
            continuous_dim=self.hparams.continuous_dim,
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.embedding_dim,
            shared_embedding_strategy=self.hparams.share_embedding_strategy,
            frac_shared_embed=self.hparams.shared_embedding_fraction,
            embedding_bias=self.hparams.embedding_bias,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
            embedding_dropout=self.hparams.embedding_dropout,
            initialization=self.hparams.embedding_initialization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hparams.deep_layers:
            x = self.linear_layers(x)
        # (N, B, E*) --> E* is the Attn Dimention
        cross_term = self.attn_proj(x).transpose(0, 1)
        if self.hparams.attention_pooling:
            attention_ops = []
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
            if self.hparams.attention_pooling:
                attention_ops.append(cross_term)
        if self.hparams.attention_pooling:
            cross_term = torch.cat(attention_ops, dim=-1)
        # (B, N, E*)
        cross_term = cross_term.transpose(0, 1)
        if self.hparams.has_residuals:
            # (B, N, E*) --> Projecting Embedded input to Attention sub-space
            V_res = self.V_res_embedding(x)
            cross_term = cross_term + V_res
        # (B, NxE*)
        cross_term = nn.ReLU()(cross_term).reshape(-1, self.output_dim)
        return cross_term


class AutoIntModel(BaseModel):
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
        self._backbone = AutoIntBackbone(self.hparams)
        # Embedding Layer
        self._embedding_layer = self._backbone._build_embedding_layer()
        # Head
        self._head = self._get_head_from_config()
