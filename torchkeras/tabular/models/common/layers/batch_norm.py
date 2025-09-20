import numpy as np
import torch
from torch import nn


class GBN(nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=512):
        super().__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim)

    def forward(self, x):
        if self.training:
            chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)


class BatchNorm1d(nn.Module):
    """BatchNorm1d with Ghost Batch Normalization."""

    def __init__(self, num_features, virtual_batch_size=None):
        super().__init__()
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        if self.virtual_batch_size is None:
            self.bn = nn.BatchNorm1d(self.num_features)
        else:
            self.bn = GBN(self.num_features, self.virtual_batch_size)

    def forward(self, x):
        return self.bn(x)
