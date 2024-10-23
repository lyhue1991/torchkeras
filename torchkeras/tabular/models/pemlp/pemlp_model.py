import torch 
from torch import nn
from torch import nn,Tensor 
import torch.nn.functional as F 

from omegaconf import DictConfig
from typing import Any, Dict, List, Optional, Union

from ..base_model import BaseModel
from ..common.layers import Embedding1dLayer

def check_input_shape(x: Tensor, expected_n_features: int) -> None:
    if x.ndim < 1:
        raise ValueError(
            f'The input must have at least one dimension, however: {x.ndim=}'
        )
    if x.shape[-1] != expected_n_features:
        raise ValueError(
            'The last dimension of the input was expected to be'
            f' {expected_n_features}, however, {x.shape[-1]=}'
        )

class Periodic(nn.Module):
    def __init__(self, n_features: int, k: int, sigma: float) -> None:
        if sigma <= 0.0:
            raise ValueError(f'sigma must be positive, however: {sigma=}')
        super().__init__()
        self._sigma = sigma
        self.weight = nn.Parameter(torch.empty(n_features, k))
        self.reset_parameters()

    def reset_parameters(self):
        # extreme values (~0.3% probability) are explicitly avoided just in case.
        bound = self._sigma * 3
        nn.init.trunc_normal_(self.weight, 0.0, self._sigma, a=-bound, b=bound)

    def forward(self, x: Tensor) -> Tensor:
        check_input_shape(x, self.weight.shape[0])
        x = 2 * torch.pi * self.weight * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x

class NLinear(nn.Module):
    """N *separate* linear layers for N feature embeddings."""
    def __init__(self, n: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                'NLinear supports only inputs with exactly one batch dimension,'
                ' so `x` must have a shape like (BATCH_SIZE, N_FEATURES, D_EMBEDDING).'
            )
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        x = x + self.bias
        return x
        

class PeriodicEmbeddings(nn.Module):
    """PL & PLR & PLR(lite) (P ~ Periodic, L ~ Linear, R ~ ReLU) embeddings for continuous features."""

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 24,
        *,
        n_frequencies: int = 48,
        frequency_init_scale: float = 0.01,
        activation: bool = True
    ) -> None:
        """
        Args:
            n_features: the number of features.
            d_embedding: the embedding size.
            n_frequencies: the number of frequencies for each feature.
                (denoted as "k" in Section 3.3 in the paper).
            frequency_init_scale: the initialization scale for the first linear layer
                (denoted as "sigma" in Section 3.3 in the paper).
                **This is an important hyperparameter**,
                see the documentation for details.
            activation: if True, the embeddings is PLR, otherwise, it is PL.
        """
        super().__init__()
        self.periodic = Periodic(n_features, n_frequencies, frequency_init_scale)
        self.linear = NLinear(n_features, 2 * n_frequencies, d_embedding)
        self.activation = nn.SiLU() if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.periodic(x)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MLP(nn.Module):
    def __init__(self, d_in, d_layers, dropout, 
                 d_out = 1):
        super().__init__()
        layers = []
        for d in d_layers:
            layers.append(nn.Linear(d_in, d))
            #layers.append(nn.BatchNorm1d(d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            d_in = d
        self.sequential = nn.Sequential(*layers)
        if d_out is not None:
            self.out = nn.Linear(d_in, d_out)
        else:
            self.out = None 
            
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, d_in)``
        """
        x = self.sequential(x)
        if self.out is not None:
            x = self.out(x)
        return x


class PeMLPBackbone(nn.Module):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__()
        self.hparams = config

        d_in = self.hparams.embedded_cat_dim + self.hparams.continuous_dim
        d_embed = self.hparams.input_embed_dim

        dropout = self.hparams.mlp_dropout
        #d_out = self.hparams.output_dim 

        d_layers = [int(x) for x in self.hparams.mlp_layers.split('-')]
        self.cont_embeddings = PeriodicEmbeddings(
            d_in, d_embed
        )
        d_in_mlp = d_in * d_embed
        
        self.mlp = MLP(
            d_in= d_in_mlp, 
            d_layers = d_layers, 
            dropout = dropout,
            d_out = None)
        
        self.output_dim = d_layers[-1] #输出尺寸
        
    def _build_embedding_layer(self):
        return Embedding1dLayer(
            continuous_dim=self.hparams.continuous_dim,
            categorical_embedding_dims=self.hparams.embedding_dims,
            embedding_dropout=self.hparams.embedding_dropout,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
            virtual_batch_size=self.hparams.virtual_batch_size,
        )
        
    def forward(self, x: Tensor):
        x = self.cont_embeddings(x).flatten(1)
        return self.mlp(x)
        
class PeMLPModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
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
        self._backbone = PeMLPBackbone(self.hparams)
        self._embedding_layer = self._backbone._build_embedding_layer()
        self.head = self._get_head_from_config()
        
    def forward(self, x: Dict) -> Dict[str, Any]:
        x = self.embed_input(x)
        x = self.compute_backbone(x)
        return self.compute_head(x)
        