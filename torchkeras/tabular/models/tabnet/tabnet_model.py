# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""TabNet Model."""
from typing import Dict

import torch
import torch.nn as nn
from ..base_model import BaseModel


class TabNetBackbone(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.hparams = config
        self._build_network()

    def _build_network(self):
        from pytorch_tabnet.tab_network import TabNet
        from pytorch_tabnet.utils import create_group_matrix
        if self.hparams.grouped_features:
            # converting the grouped_features into a nested list of indices
            features = self.hparams.categorical_cols + self.hparams.continuous_cols
            grp_list = [
                [features.index(col) for col in grp if col in features] for grp in self.hparams.grouped_features
            ]
        else:
            # creating a default grp_list with each feature as a group
            grp_list = [[i] for i in range(self.hparams.continuous_dim + self.hparams.categorical_dim)]
        group_matrix = create_group_matrix(
            grp_list,
            self.hparams.continuous_dim + self.hparams.categorical_dim,
        )
        self.tabnet = TabNet(
            input_dim=self.hparams.continuous_dim + self.hparams.categorical_dim,
            output_dim=self.hparams.output_dim,
            n_d=self.hparams.n_d,
            n_a=self.hparams.n_a,
            n_steps=self.hparams.n_steps,
            gamma=self.hparams.gamma,
            cat_idxs=list(range(self.hparams.categorical_dim)),
            cat_dims=[cardinality for cardinality, _ in self.hparams.embedding_dims],
            cat_emb_dim=[embed_dim for _, embed_dim in self.hparams.embedding_dims],
            n_independent=self.hparams.n_independent,
            n_shared=self.hparams.n_shared,
            epsilon=1e-15,
            virtual_batch_size=self.hparams.virtual_batch_size,
            momentum=0.02,
            mask_type=self.hparams.mask_type,
            group_attention_matrix=group_matrix,
        )

    def unpack_input(self, x: Dict):
        # unpacking into a tuple
        x = x["categorical"], x["continuous"]
        # eliminating None in case there is no categorical or continuous columns
        x = (item for item in x if len(item) > 0)
        x = torch.cat(tuple(x), dim=1)
        return x

    def forward(self, x: Dict):
        # unpacking into a tuple
        x = self.unpack_input(x)
        # Making two parameters to the right device.
        self.tabnet.embedder.embedding_group_matrix = self.tabnet.embedder.embedding_group_matrix.to(x.device)
        self.tabnet.tabnet.encoder.group_attention_matrix = self.tabnet.tabnet.encoder.group_attention_matrix.to(
            x.device
        )
        # Returns output and Masked Loss. We only need the output
        x, _ = self.tabnet(x)
        return x


class TabNetModel(BaseModel):
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
        # TabNet has its own embedding layer.
        # So we are not using the embedding layer from BaseModel
        self._embedding_layer = nn.Identity()
        self._backbone = TabNetBackbone(self.hparams)
        setattr(self.backbone, "output_dim", self.hparams.output_dim)
        # TabNet has its own head
        self._head = nn.Identity()

    def extract_embedding(self):
        raise ValueError("Extracting Embeddings is not supported by Tabnet. Please use another" " compatible model")
