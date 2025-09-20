# W605
import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from torchkeras.tabular.models.common.layers.batch_norm import BatchNorm1d
from torchkeras.tabular.utils import _initialize_kaiming


# Slight adaptation from https://github.com/jrzaurin/pytorch-widedeep which in turn adapted from AutoGluon
class SharedEmbeddings(nn.Module):
    """Enables different values in a categorical feature to share some embeddings across."""

    def __init__(
        self,
        num_embed: int,
        embed_dim: int,
        add_shared_embed: bool = False,
        frac_shared_embed: float = 0.25,
    ):
        super().__init__()
        assert frac_shared_embed < 1, "'frac_shared_embed' must be less than 1"

        self.add_shared_embed = add_shared_embed
        self.embed = nn.Embedding(num_embed, embed_dim, padding_idx=0)
        self.embed.weight.data.clamp_(-2, 2)
        if add_shared_embed:
            col_embed_dim = embed_dim
        else:
            col_embed_dim = int(embed_dim * frac_shared_embed)
        self.shared_embed = nn.Parameter(torch.empty(1, col_embed_dim).uniform_(-1, 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.embed(X)
        shared_embed = self.shared_embed.expand(out.shape[0], -1)
        if self.add_shared_embed:
            out += shared_embed
        else:
            out[:, : shared_embed.shape[1]] = shared_embed
        return out

    @property
    def weight(self):
        w = self.embed.weight.detach()
        if self.add_shared_embed:
            w += self.shared_embed
        else:
            w[:, : self.shared_embed.shape[1]] = self.shared_embed
        return w


class PreEncoded1dLayer(nn.Module):
    """Takes in pre-encoded categorical variables and just concatenates with continuous variables No learnable
    component."""

    def __init__(
        self,
        continuous_dim: int,
        categorical_dim: Tuple[int, int],
        embedding_dropout: float = 0.0,
        batch_norm_continuous_input: bool = False,
        virtual_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_dim = categorical_dim
        self.batch_norm_continuous_input = batch_norm_continuous_input

        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None
        # Continuous Layers
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = BatchNorm1d(continuous_dim, virtual_batch_size)

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        assert "continuous" in x or "categorical" in x, "x must contain either continuous and categorical features"
        # (B, N)
        continuous_data, categorical_data = (
            x.get("continuous", torch.empty(0, 0)),
            x.get("categorical", torch.empty(0, 0)),
        )
        assert (
            categorical_data.shape[1] == self.categorical_dim
        ), "categorical_data must have same number of columns as categorical embedding layers"
        assert (
            continuous_data.shape[1] == self.continuous_dim
        ), "continuous_data must have same number of columns as continuous dim"
        embed = None
        if continuous_data.shape[1] > 0:
            if self.batch_norm_continuous_input:
                embed = self.normalizing_batch_norm(continuous_data)
            else:
                embed = continuous_data
            # (B, N, C)
        if categorical_data.shape[1] > 0:
            # (B, N, C)
            if embed is None:
                embed = categorical_data
            else:
                embed = torch.cat([embed, categorical_data], dim=1)
        if self.embd_dropout is not None:
            embed = self.embd_dropout(embed)
        return embed


class Embedding1dLayer(nn.Module):
    """Enables different values in a categorical features to have different embeddings."""

    def __init__(
        self,
        continuous_dim: int,
        categorical_embedding_dims: Tuple[int, int],
        embedding_dropout: float = 0.0,
        batch_norm_continuous_input: bool = False,
        virtual_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_embedding_dims = categorical_embedding_dims
        self.batch_norm_continuous_input = batch_norm_continuous_input

        # Embedding layers
        self.cat_embedding_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in categorical_embedding_dims])
        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None
        # Continuous Layers
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = BatchNorm1d(continuous_dim, virtual_batch_size)

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        assert "continuous" in x or "categorical" in x, "x must contain either continuous and categorical features"
        # (B, N)
        continuous_data, categorical_data = (
            x.get("continuous", torch.empty(0, 0)),
            x.get("categorical", torch.empty(0, 0)),
        )
        assert categorical_data.shape[1] == len(
            self.cat_embedding_layers
        ), "categorical_data must have same number of columns as categorical embedding layers"
        assert (
            continuous_data.shape[1] == self.continuous_dim
        ), "continuous_data must have same number of columns as continuous dim"
        embed = None
        if continuous_data.shape[1] > 0:
            if self.batch_norm_continuous_input:
                embed = self.normalizing_batch_norm(continuous_data)
            else:
                embed = continuous_data
            # (B, N, C)
        if categorical_data.shape[1] > 0:
            categorical_embed = torch.cat(
                [
                    embedding_layer(categorical_data[:, i])
                    for i, embedding_layer in enumerate(self.cat_embedding_layers)
                ],
                dim=1,
            )
            # (B, N, C + C)
            if embed is None:
                embed = categorical_embed
            else:
                embed = torch.cat([embed, categorical_embed], dim=1)
        if self.embd_dropout is not None:
            embed = self.embd_dropout(embed)
        return embed


class Embedding2dLayer(nn.Module):
    """Embeds categorical and continuous features into a 2D tensor."""

    def __init__(
        self,
        continuous_dim: int,
        categorical_cardinality: List[int],
        embedding_dim: int,
        shared_embedding_strategy: Optional[str] = None,
        frac_shared_embed: float = 0.25,
        embedding_bias: bool = False,
        batch_norm_continuous_input: bool = False,
        virtual_batch_size: Optional[int] = None,
        embedding_dropout: float = 0.0,
        initialization: Optional[str] = None,
    ):
        """
        Args:
            continuous_dim: number of continuous features
            categorical_cardinality: list of cardinalities of categorical features
            embedding_dim: embedding dimension
            shared_embedding_strategy: strategy to use for shared embeddings
            frac_shared_embed: fraction of embeddings to share
            embedding_bias: whether to use bias in embedding layers
            batch_norm_continuous_input: whether to use batch norm on continuous features
            embedding_dropout: dropout to apply to embeddings
            initialization: initialization strategy to use for embedding layers"""
        super().__init__()
        self.continuous_dim = continuous_dim
        self.categorical_cardinality = categorical_cardinality
        self.embedding_dim = embedding_dim
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.shared_embedding_strategy = shared_embedding_strategy
        self.frac_shared_embed = frac_shared_embed
        self.embedding_bias = embedding_bias
        self.initialization = initialization
        d_sqrt_inv = 1 / math.sqrt(embedding_dim)
        if initialization is not None:
            assert initialization in [
                "kaiming_uniform",
                "kaiming_normal",
            ], "initialization should be either of `kaiming` or `uniform`"
            self._do_kaiming_initialization = True
            self._initialize_kaiming = partial(
                _initialize_kaiming,
                initialization=initialization,
                d_sqrt_inv=d_sqrt_inv,
            )
        else:
            self._do_kaiming_initialization = False

        # cat Embedding layers
        if self.shared_embedding_strategy is not None:
            self.cat_embedding_layers = nn.ModuleList(
                [
                    SharedEmbeddings(
                        c,
                        self.embedding_dim,
                        add_shared_embed=(self.shared_embedding_strategy == "add"),
                        frac_shared_embed=self.frac_shared_embed,
                    )
                    for c in categorical_cardinality
                ]
            )
            if self._do_kaiming_initialization:
                for embedding_layer in self.cat_embedding_layers:
                    self._initialize_kaiming(embedding_layer.embed.weight)
                    self._initialize_kaiming(embedding_layer.shared_embed)
        else:
            self.cat_embedding_layers = nn.ModuleList(
                [nn.Embedding(c, self.embedding_dim) for c in categorical_cardinality]
            )
            if self._do_kaiming_initialization:
                for embedding_layer in self.cat_embedding_layers:
                    self._initialize_kaiming(embedding_layer.weight)
        if embedding_bias:
            self.cat_embedding_bias = nn.Parameter(torch.Tensor(len(self.categorical_cardinality), self.embedding_dim))
            if self._do_kaiming_initialization:
                self._initialize_kaiming(self.cat_embedding_bias)
        # Continuous Embedding Layer
        self.cont_embedding_layer = nn.Embedding(self.continuous_dim, self.embedding_dim)
        if self._do_kaiming_initialization:
            self._initialize_kaiming(self.cont_embedding_layer.weight)
        if embedding_bias:
            self.cont_embedding_bias = nn.Parameter(torch.Tensor(self.continuous_dim, self.embedding_dim))
            if self._do_kaiming_initialization:
                self._initialize_kaiming(self.cont_embedding_bias)
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = BatchNorm1d(continuous_dim, virtual_batch_size)
        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        assert "continuous" in x or "categorical" in x, "x must contain either continuous and categorical features"
        # (B, N)
        continuous_data, categorical_data = (
            x.get("continuous", torch.empty(0, 0)),
            x.get("categorical", torch.empty(0, 0)),
        )
        assert categorical_data.shape[1] == len(
            self.cat_embedding_layers
        ), "categorical_data must have same number of columns as categorical embedding layers"
        assert (
            continuous_data.shape[1] == self.continuous_dim
        ), "continuous_data must have same number of columns as continuous dim"
        embed = None
        if continuous_data.shape[1] > 0:
            cont_idx = torch.arange(self.continuous_dim, device=continuous_data.device).expand(
                continuous_data.size(0), -1
            )
            if self.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)
            embed = torch.mul(
                continuous_data.unsqueeze(2),
                self.cont_embedding_layer(cont_idx),
            )
            if self.embedding_bias:
                embed += self.cont_embedding_bias
            # (B, N, C)
        if categorical_data.shape[1] > 0:
            categorical_embed = torch.cat(
                [
                    embedding_layer(categorical_data[:, i]).unsqueeze(1)
                    for i, embedding_layer in enumerate(self.cat_embedding_layers)
                ],
                dim=1,
            )
            if self.embedding_bias:
                categorical_embed += self.cat_embedding_bias
            # (B, N, C + C)
            if embed is None:
                embed = categorical_embed
            else:
                embed = torch.cat([embed, categorical_embed], dim=1)
        if self.embd_dropout is not None:
            embed = self.embd_dropout(embed)
        return embed
