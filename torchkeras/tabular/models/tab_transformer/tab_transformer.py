# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
# Inspired by implementations
# 1. lucidrains - https://github.com/lucidrains/tab-transformer-pytorch/
# If you are interested in Transformers, you should definitely check out his repositories.
# 2. PyTorch Wide and Deep - https://github.com/jrzaurin/pytorch-widedeep/
# It is another library for tabular data, which supports multi modal problems.
# Check out the library if you haven't already.
# 3. AutoGluon - https://github.com/awslabs/autogluon
# AutoGluon is an AuttoML library which supports Tabular data as well. it is from Amazon Research and is in MXNet
# 4. LabML Annotated Deep Learning Papers - The position-wise FF was shamelessly copied from
# https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/transformers
"""TabTransformer Model."""
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn


from ..common.layers.batch_norm import BatchNorm1d
from ..base_model import BaseModel
from ..common.layers import Embedding2dLayer, TransformerEncoderBlock


class TabTransformerBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.share_embedding_strategy in [
            "add",
            "fraction",
        ], (
            f"`share_embedding_strategy` should be one of `add` or `fraction`,"
            f" not {self.hparams.share_embedding_strategy}"
        )
        self.hparams = config
        self._build_network()

    def _build_network(self):
        self.transformer_blocks = OrderedDict()
        for i in range(self.hparams.num_attn_blocks):
            self.transformer_blocks[f"mha_block_{i}"] = TransformerEncoderBlock(
                input_embed_dim=self.hparams.input_embed_dim,
                num_heads=self.hparams.num_heads,
                ff_hidden_multiplier=self.hparams.ff_hidden_multiplier,
                ff_activation=self.hparams.transformer_activation,
                attn_dropout=self.hparams.attn_dropout,
                ff_dropout=self.hparams.ff_dropout,
                add_norm_dropout=self.hparams.add_norm_dropout,
                keep_attn=False,  # No easy way to convert TabTransformer Attn Weights to Feature Importance
            )
        self.transformer_blocks = nn.Sequential(self.transformer_blocks)
        self.attention_weights = [None] * self.hparams.num_attn_blocks
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = BatchNorm1d(self.hparams.continuous_dim, self.hparams.virtual_batch_size)

        self.output_dim = self.hparams.input_embed_dim * self.hparams.categorical_dim + self.hparams.continuous_dim

    def _build_embedding_layer(self):
        return Embedding2dLayer(
            continuous_dim=0,  # Only passing and embedding categorical features
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.input_embed_dim,
            shared_embedding_strategy=self.hparams.share_embedding_strategy,
            frac_shared_embed=self.hparams.shared_embedding_fraction,
            embedding_bias=self.hparams.embedding_bias,
            embedding_dropout=self.hparams.embedding_dropout,
            initialization=self.hparams.embedding_initialization,
            virtual_batch_size=self.hparams.virtual_batch_size,
        )

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        # (B, N)
        x = None
        if self.hparams.categorical_dim > 0:
            for i, block in enumerate(self.transformer_blocks):
                x_cat = block(x_cat)
            # Flatten (Batch, N_Categorical, Hidden) --> (Batch, N_CategoricalxHidden)
            #from einops import rearrange
            #x = rearrange(x_cat, "b n h -> b (n h)")
            b = x_cat.size(0)
            x = x_cat.view(b,-1)
        
        if self.hparams.continuous_dim > 0:
            if self.hparams.batch_norm_continuous_input:
                x_cont = self.normalizing_batch_norm(x_cont)
            else:
                x_cont = x_cont
            # (B, N, E)
            x = x_cont if x is None else torch.cat([x, x_cont], 1)
        return x


class TabTransformerModel(BaseModel):
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
        self._backbone = TabTransformerBackbone(self.hparams)
        # Embedding Layer
        self._embedding_layer = self._backbone._build_embedding_layer()
        # Head
        self._head = self._get_head_from_config()

    # Redefining forward because this model flow is slightly different
    def forward(self, x: Dict):
        if self.hparams.categorical_dim > 0:
            x_cat = self.embed_input({"categorical": x["categorical"]})
        else:
            x_cat = None
        x = self.compute_backbone({"categorical": x_cat, "continuous": x["continuous"]})
        return self.compute_head(x)

    # Redefining compute_backbone because this model flow is slightly different
    def compute_backbone(self, x: Dict):
        # Returns output
        x = self.backbone(x["categorical"], x["continuous"])
        return x
