# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
import warnings

import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..common import activations
from ..common.layers import Lambda
from ..common.layers import Embedding1dLayer
from .architecture_blocks import DenseODSTBlock

class NODEBackbone(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.hparams = config
        # self.hparams.output_dim = (0 if self.hparams.output_dim is None else self.hparams.output_dim)
        # For SSL cases where output_dim will be None
        self._build_network()

    def _build_network(self):
        self.hparams.node_input_dim = self.hparams.continuous_dim + self.hparams.embedded_cat_dim
        self.dense_block = DenseODSTBlock(
            input_dim=self.hparams.node_input_dim,
            num_trees=self.hparams.num_trees,
            num_layers=self.hparams.num_layers,
            tree_output_dim=self.hparams.output_dim + self.hparams.additional_tree_output_dim,
            max_features=self.hparams.max_features,
            input_dropout=self.hparams.input_dropout,
            depth=self.hparams.depth,
            choice_function=getattr(activations, self.hparams.choice_function),
            bin_function=getattr(activations, self.hparams.bin_function),
            initialize_response_=getattr(nn.init, self.hparams.initialize_response + "_"),
            initialize_selection_logits_=getattr(nn.init, self.hparams.initialize_selection_logits + "_"),
            threshold_init_beta=self.hparams.threshold_init_beta,
            threshold_init_cutoff=self.hparams.threshold_init_cutoff,
        )
        self.output_dim = self.hparams.output_dim + self.hparams.additional_tree_output_dim

    def _build_embedding_layer(self):
        embedding = Embedding1dLayer(
            continuous_dim=self.hparams.continuous_dim,
            categorical_embedding_dims=self.hparams.embedding_dims,
            embedding_dropout=self.hparams.embedding_dropout,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
            virtual_batch_size=self.hparams.virtual_batch_size,
        )
        return embedding

    def forward(self, x: torch.Tensor):
        x = self.dense_block(x)
        return x


class NODEModel(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def subset(self, x):
        return x[..., : self.hparams.output_dim].mean(dim=-2)

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
        self._backbone = NODEBackbone(self.hparams)
        # Embedding Layer
        self._embedding_layer = self._backbone._build_embedding_layer()
        # average first n channels of every tree, where n is the number of output targets for regression
        # and number of classes for classification
        # Not using config head because NODE has a specific head
        warnings.warn("Ignoring head config because NODE has a specific head which subsets the tree outputs")
        self._head = Lambda(self.subset)
