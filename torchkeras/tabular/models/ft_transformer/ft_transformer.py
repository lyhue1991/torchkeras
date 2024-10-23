# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Feature Tokenizer Transformer Model."""
from collections import OrderedDict

import torch
import torch.nn as nn

from ..common.layers.batch_norm import BatchNorm1d

from ..base_model import BaseModel
from ..common.layers import AppendCLSToken, Embedding2dLayer, TransformerEncoderBlock


class FTTransformerBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.share_embedding_strategy in [
            "add",
            "fraction",
        ], (
            "`share_embedding_strategy` should be one of `add` or `fraction`,"
            f" not {self.hparams.share_embedding_strategy}"
        )
        self.hparams = config
        self._build_network()

    def _build_network(self):
        self.add_cls = AppendCLSToken(
            d_token=self.hparams.input_embed_dim,
            initialization=self.hparams.embedding_initialization,
        )
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
                keep_attn=self.hparams.attn_feature_importance,  # Can use Attn Weights to derive feature importance
            )
        self.transformer_blocks = nn.Sequential(self.transformer_blocks)
        if self.hparams.attn_feature_importance:
            self.attention_weights_ = [None] * self.hparams.num_attn_blocks
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = BatchNorm1d(self.hparams.continuous_dim, self.hparams.virtual_batch_size)

        self.output_dim = self.hparams.input_embed_dim

    def _build_embedding_layer(self):
        return Embedding2dLayer(
            continuous_dim=self.hparams.continuous_dim,
            categorical_cardinality=self.hparams.categorical_cardinality,
            embedding_dim=self.hparams.input_embed_dim,
            shared_embedding_strategy=self.hparams.share_embedding_strategy,
            frac_shared_embed=self.hparams.shared_embedding_fraction,
            embedding_bias=self.hparams.embedding_bias,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
            embedding_dropout=self.hparams.embedding_dropout,
            initialization=self.hparams.embedding_initialization,
            virtual_batch_size=self.hparams.virtual_batch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.add_cls(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            if self.hparams.attn_feature_importance:
                self.attention_weights_[i] = block.mha.attn_weights
                # self.feature_importance_+=block.mha.attn_weights[:,:,:,-1].sum(dim=1)
                # self._calculate_feature_importance(block.mha.attn_weights)
        if self.hparams.attn_feature_importance:
            self._calculate_feature_importance()
        # Flatten (Batch, N_Categorical, Hidden) --> (Batch, N_CategoricalxHidden)
        # x = rearrange(x, "b n h -> b (n h)")
        # Taking only CLS token for the prediction head
        return x[:, -1]

    # Not Tested Properly
    def _calculate_feature_importance(self):
        n, h, f, _ = self.attention_weights_[0].shape
        device = self.attention_weights_[0].device
        L = len(self.attention_weights_)
        self.local_feature_importance = torch.zeros((n, f), device=device)
        for attn_weights in self.attention_weights_:
            self.local_feature_importance += attn_weights[:, :, :, -1].sum(dim=1)
        self.local_feature_importance = (1 / (h * L)) * self.local_feature_importance[:, :-1]
        self.feature_importance_ = self.local_feature_importance.mean(dim=0).detach().cpu().numpy()
        # self.feature_importance_count_+=attn_weights.shape[0]


class FTTransformerModel(BaseModel):
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
        self._backbone = FTTransformerBackbone(self.hparams)
        # Embedding Layer
        self._embedding_layer = self._backbone._build_embedding_layer()
        # Head
        self._head = self._get_head_from_config()

    def feature_importance(self):
        if self.hparams.attn_feature_importance:
            return super().feature_importance()
        else:
            raise ValueError("If you want Feature Importance, `attn_feature_weights` should be `True`.")
