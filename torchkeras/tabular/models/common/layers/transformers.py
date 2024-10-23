# W605
import math
from typing import Optional

import torch
from torch import einsum, nn

from torchkeras.tabular.utils import _initialize_kaiming

from .gated_units import GEGLU, PositionWiseFeedForward, ReGLU, SwiGLU

GATED_UNITS = {"GEGLU": GEGLU, "ReGLU": ReGLU, "SwiGLU": SwiGLU}


class AddNorm(nn.Module):
    """Applies LayerNorm, Dropout and adds to input.

    Standard AddNorm operations in Transformers

    """

    def __init__(self, input_dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.ln(self.dropout(Y) + X)


class MultiHeadedAttention(nn.Module):
    """Multi Headed Attention Block in Transformers."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        head_dim: int = 16,
        dropout: int = 0.1,
        keep_attn: bool = True,
    ):
        super().__init__()
        assert input_dim % num_heads == 0, "'input_dim' must be multiples of 'num_heads'"
        inner_dim = head_dim * num_heads
        self.n_heads = num_heads
        self.scale = head_dim**-0.5
        self.keep_attn = keep_attn

        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        from einops import rearrange
        h = self.n_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (q, k, v))
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        if self.keep_attn:
            self.attn_weights = attn
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class TransformerEncoderBlock(nn.Module):
    """A single Transformer Encoder Block."""

    def __init__(
        self,
        input_embed_dim: int,
        num_heads: int = 8,
        ff_hidden_multiplier: int = 4,
        ff_activation: str = "GEGLU",
        attn_dropout: float = 0.1,
        keep_attn: bool = True,
        ff_dropout: float = 0.1,
        add_norm_dropout: float = 0.1,
        transformer_head_dim: Optional[int] = None,
    ):
        """
        Args:
            input_embed_dim: The input embedding dimension
            num_heads: The number of attention heads
            ff_hidden_multiplier: The hidden dimension multiplier for the position-wise feed-forward layer
            ff_activation: The activation function for the position-wise feed-forward layer
            attn_dropout: The dropout probability for the attention layer
            keep_attn: Whether to keep the attention weights
            ff_dropout: The dropout probability for the position-wise feed-forward layer
            add_norm_dropout: The dropout probability for the residual connections
            transformer_head_dim: The dimension of the attention heads. If None, will default to input_embed_dim
        """
        super().__init__()
        self.mha = MultiHeadedAttention(
            input_embed_dim,
            num_heads,
            head_dim=input_embed_dim if transformer_head_dim is None else transformer_head_dim,
            dropout=attn_dropout,
            keep_attn=keep_attn,
        )

        try:
            self.pos_wise_ff = GATED_UNITS[ff_activation](
                d_model=input_embed_dim,
                d_ff=input_embed_dim * ff_hidden_multiplier,
                dropout=ff_dropout,
            )
        except (AttributeError, KeyError):
            self.pos_wise_ff = PositionWiseFeedForward(
                d_model=input_embed_dim,
                d_ff=input_embed_dim * ff_hidden_multiplier,
                dropout=ff_dropout,
                activation=getattr(nn, ff_activation)(),
            )
        self.attn_add_norm = AddNorm(input_embed_dim, add_norm_dropout)
        self.ff_add_norm = AddNorm(input_embed_dim, add_norm_dropout)

    def forward(self, x):
        y = self.mha(x)
        x = self.attn_add_norm(x, y)
        y = self.pos_wise_ff(y)
        return self.ff_add_norm(x, y)


class AppendCLSToken(nn.Module):
    """Appends the [CLS] token for BERT-like inference."""

    def __init__(self, d_token: int, initialization: str) -> None:
        """Initialize self."""
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_token))
        d_sqrt_inv = 1 / math.sqrt(d_token)
        _initialize_kaiming(self.weight, initialization, d_sqrt_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        assert x.ndim == 3
        return torch.cat([x, self.weight.view(1, 1, -1).repeat(len(x), 1, 1)], dim=1)
