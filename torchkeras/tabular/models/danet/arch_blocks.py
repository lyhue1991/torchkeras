from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.layers.activations import entmax15
from ..common.layers.batch_norm import GBN


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    return


class LearnableLocality(nn.Module):
    def __init__(self, input_dim, k):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.rand(k, input_dim)))
        self.smax = partial(entmax15, dim=-1)

    def forward(self, x):
        mask = self.smax(self.weight)
        masked_x = torch.einsum("nd,bd->bnd", mask, x)  # [B, k, D]
        return masked_x


class AbstractLayer(nn.Module):
    def __init__(self, base_input_dim, base_output_dim, k, virtual_batch_size, bias=True):
        super().__init__()
        self.masker = LearnableLocality(input_dim=base_input_dim, k=k)
        self.fc = nn.Conv1d(
            base_input_dim * k,
            2 * k * base_output_dim,
            kernel_size=1,
            groups=k,
            bias=bias,
        )
        initialize_glu(self.fc, input_dim=base_input_dim * k, output_dim=2 * k * base_output_dim)
        self.bn = GBN(2 * base_output_dim * k, virtual_batch_size)
        self.k = k
        self.base_output_dim = base_output_dim

    def forward(self, x):
        b = x.size(0)
        x = self.masker(x)  # [B, D] -> [B, k, D]
        x = self.fc(x.view(b, -1, 1))  # [B, k, D] -> [B, k * D, 1] -> [B, k * (2 * D'), 1]
        x = self.bn(x)
        chunks = x.chunk(self.k, 1)  # k * [B, 2 * D', 1]
        x = sum(
            [
                F.relu(torch.sigmoid(x_[:, : self.base_output_dim, :]) * x_[:, self.base_output_dim :, :])
                for x_ in chunks
            ]
        )  # k * [B, D', 1] -> [B, D', 1]
        return x.squeeze(-1)


class BasicBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        abstlay_dim_1,
        abstlay_dim_2,
        k,
        virtual_batch_size,
        fix_input_dim,
        drop_rate,
        block_activation,
    ):
        super().__init__()
        self.conv1 = AbstractLayer(input_dim, abstlay_dim_1, k, virtual_batch_size)
        self.conv2 = AbstractLayer(abstlay_dim_1, abstlay_dim_2, k, virtual_batch_size)

        self.downsample = nn.Sequential(
            nn.Dropout(drop_rate),
            AbstractLayer(fix_input_dim, abstlay_dim_2, k, virtual_batch_size),
        )
        self.block_activation = block_activation

    def forward(self, x, pre_out=None):
        if pre_out is None:
            pre_out = x
        out = self.conv1(pre_out)
        out = self.conv2(out)
        identity = self.downsample(x)
        out += identity
        return self.block_activation(out)
