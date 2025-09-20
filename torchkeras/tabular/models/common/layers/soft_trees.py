import random
from typing import Callable
from warnings import warn

import numpy as np
import torch
import torch.nn as nn

from torchkeras.tabular.utils import check_numpy

from ..layers import ModuleWithInit
from .activations import RSoftmax, entmax15, sparsemax, sparsemoid

# Neural Oblivious Decision Ensembles
# Author: Sergey Popov, Julian Qian
# https://github.com/Qwicen/node
# For license information, see https://github.com/Qwicen/node/blob/master/LICENSE.md


class ODST(ModuleWithInit):
    def __init__(
        self,
        in_features,
        num_trees,
        depth=6,
        tree_output_dim=1,
        flatten_output=True,
        choice_function=sparsemax,
        bin_function=sparsemoid,
        initialize_response_=nn.init.normal_,
        initialize_selection_logits_=nn.init.uniform_,
        threshold_init_beta=1.0,
        threshold_init_cutoff=1.0,
    ):
        """Oblivious Differentiable Sparsemax Trees. http://tinyurl.com/odst-readmore One can drop (sic!) this module
        anywhere instead of nn.Linear.

        :param in_features: number of features in the input tensor
        :param num_trees: number of trees in this layer
        :param tree_dim: number of response channels in the response of individual tree
        :param depth: number of splits in every tree
        :param flatten_output: if False, returns [..., num_trees, tree_dim],
            by default returns [..., num_trees * tree_dim]
        :param choice_function: f(tensor, dim) -> R_simplex computes feature weights s.t. f(tensor, dim).sum(dim) == 1
        :param bin_function: f(tensor) -> R[0, 1], computes tree leaf weights

        :param initialize_response_: in-place initializer for tree output tensor
        :param initialize_selection_logits_: in-place initializer for logits that select features for the tree
        both thresholds and scales are initialized with data-aware init (or .load_state_dict)
        :param threshold_init_beta: initializes threshold to a q-th quantile of data points
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
            If this param is set to 1, initial thresholds will have the same distribution as data points
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.

        :param threshold_init_cutoff: threshold log-temperatures initializer, in (0, inf)
            By default(1.0), log-remperatures are initialized in such a way that all bin selectors
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
        """
        super().__init__()
        self.depth, self.num_trees, self.tree_dim, self.flatten_output = (
            depth,
            num_trees,
            tree_output_dim,
            flatten_output,
        )
        self.choice_function, self.bin_function = choice_function, bin_function
        self.threshold_init_beta, self.threshold_init_cutoff = (
            threshold_init_beta,
            threshold_init_cutoff,
        )

        self.response = nn.Parameter(torch.zeros([num_trees, tree_output_dim, 2**depth]), requires_grad=True)
        initialize_response_(self.response)

        self.feature_selection_logits = nn.Parameter(torch.zeros([in_features, num_trees, depth]), requires_grad=True)
        initialize_selection_logits_(self.feature_selection_logits)

        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )  # nan values will be initialized on first batch (data-aware init)
        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float("nan"), dtype=torch.float32),
            requires_grad=True,
        )

        # binary codes for mapping between 1-hot vectors and bin indices
        with torch.no_grad():
            indices = torch.arange(2**self.depth)
            offsets = 2 ** torch.arange(self.depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
            self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)
            # ^-- [depth, 2 ** depth, 2]

    def forward(self, input):
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)
        # new input shape: [batch_size, in_features]

        feature_logits = self.feature_selection_logits
        feature_selectors = self.choice_function(feature_logits, dim=0)
        # ^--[in_features, num_trees, depth]

        feature_values = torch.einsum("bi,ind->bnd", input, feature_selectors)
        # ^--[batch_size, num_trees, depth]

        threshold_logits = (feature_values - self.feature_thresholds) * torch.exp(-self.log_temperatures)

        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # ^--[batch_size, num_trees, depth, 2]

        bins = self.bin_function(threshold_logits)
        # ^--[batch_size, num_trees, depth, 2], approximately binary

        bin_matches = torch.einsum("btds,dcs->btdc", bins, self.bin_codes_1hot)
        # ^--[batch_size, num_trees, depth, 2 ** depth]

        response_weights = torch.prod(bin_matches, dim=-2)
        # ^-- [batch_size, num_trees, 2 ** depth]

        response = torch.einsum("bnd,ncd->bnc", response_weights, self.response)
        # ^-- [batch_size, num_trees, tree_dim]

        return response.flatten(1, 2) if self.flatten_output else response

    def initialize(self, input, eps=1e-6):
        # data-aware initializer
        assert len(input.shape) == 2
        if input.shape[0] < 1000:
            warn(
                "Data-aware initialization is performed on less than 1000 data points. This may cause instability."
                "To avoid potential problems, run this model on a data batch with at least 1000 data samples."
                "You can do so manually before training. Use with torch.no_grad() for memory efficiency."
            )
        with torch.no_grad():
            feature_selectors = self.choice_function(self.feature_selection_logits, dim=0)
            # ^--[in_features, num_trees, depth]

            feature_values = torch.einsum("bi,ind->bnd", input, feature_selectors)
            # ^--[batch_size, num_trees, depth]

            # initialize thresholds: sample random percentiles of data
            percentiles_q = 100 * np.random.beta(
                self.threshold_init_beta,
                self.threshold_init_beta,
                size=[self.num_trees, self.depth],
            )
            self.feature_thresholds.data[...] = torch.as_tensor(
                list(
                    map(
                        np.percentile,
                        check_numpy(feature_values.flatten(1, 2).t()),
                        percentiles_q.flatten(),
                    )
                ),
                dtype=feature_values.dtype,
                device=feature_values.device,
            ).view(self.num_trees, self.depth)

            # init temperatures: make sure enough data points are in the linear region of sparse-sigmoid
            temperatures = np.percentile(
                check_numpy(abs(feature_values - self.feature_thresholds)),
                q=100 * min(1.0, self.threshold_init_cutoff),
                axis=0,
            )

            # if threshold_init_cutoff > 1, scale everything down by it
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures) + eps)

    def __repr__(self):
        return "{}(in_features={}, num_trees={}, depth={}, tree_dim={}, flatten_output={})".format(
            self.__class__.__name__,
            self.feature_selection_logits.shape[0],
            self.num_trees,
            self.depth,
            self.tree_dim,
            self.flatten_output,
        )


class NeuralDecisionStump(nn.Module):
    def __init__(
        self,
        n_features: int,
        binning_activation: Callable = entmax15,
        feature_mask_function: Callable = entmax15,
        feature_sparsity: float = 0.8,
        learnable_sparsity: bool = True,
    ):
        super().__init__()
        self._num_cutpoints = 1
        self._num_leaf = 2
        self.n_features = n_features
        self.binning_activation = binning_activation
        self.feature_mask_function = feature_mask_function
        self.feature_sparsity = feature_sparsity
        self.learnable_sparsity = learnable_sparsity
        self._build_network()

    def _build_network(self):
        # sampling a random beta distribution
        # random distribution helps with diversity in trees and feature splits
        alpha = random.uniform(0.5, 10.0)
        beta = random.uniform(0.5, 10.0)
        # with torch.no_grad():
        feature_mask = (
            torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta])).sample((self.n_features,)).squeeze(-1)
        )
        self.feature_mask = nn.Parameter(feature_mask, requires_grad=True)
        if self.feature_mask_function.__name__ == "t_softmax":
            t = RSoftmax.calculate_t(self.feature_mask, r=torch.tensor([self.feature_sparsity]))
            self.t = nn.Parameter(t, requires_grad=self.learnable_sparsity)
        W = torch.linspace(
            1.0,
            self._num_cutpoints + 1.0,
            self._num_cutpoints + 1,
            requires_grad=False,
        ).reshape(1, 1, -1)
        self.register_buffer("W", W)

        cutpoints = torch.rand([self.n_features, self._num_cutpoints])
        # Append zeros to the beginning of each row
        cutpoints = torch.cat([torch.zeros([self.n_features, 1], device=cutpoints.device), cutpoints], 1)
        self.cut_points = nn.Parameter(cutpoints, requires_grad=True)
        self.leaf_responses = nn.Parameter(torch.rand(self.n_features, self._num_leaf), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_mask_function.__name__ == "t_softmax":
            t = torch.relu(self.t)
            feature_mask = self.feature_mask_function(self.feature_mask, t)
        else:
            feature_mask = self.feature_mask_function(self.feature_mask)
        # Repeat W for each batch size using broadcasting
        W = torch.ones(x.size(0), 1, 1, device=x.device) * self.W
        # Binning features
        x = torch.bmm(x.unsqueeze(-1), W) - self.cut_points.unsqueeze(0)
        x = self.binning_activation(x)  # , dim=-1)
        x = x * self.leaf_responses.unsqueeze(0)
        x = (x * feature_mask.reshape(1, -1, 1)).sum(dim=1)
        return x, feature_mask


class NeuralDecisionTree(nn.Module):
    def __init__(
        self,
        depth: int,
        n_features: int,
        dropout: float = 0,
        binning_activation: Callable = entmax15,
        feature_mask_function: Callable = entmax15,
        feature_sparsity: float = 0.8,
        learnable_sparsity: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self._num_cutpoints = 1
        self.n_features = n_features
        self._dropout = dropout
        self.binning_activation = binning_activation
        self.feature_mask_function = feature_mask_function
        self.feature_sparsity = feature_sparsity
        self.learnable_sparsity = learnable_sparsity
        self._build_network()

    def _build_network(self):
        for d in range(self.depth):
            for n in range(max(2 ** (d), 1)):
                self.add_module(
                    f"decision_stump_{d}_{n}",
                    NeuralDecisionStump(
                        self.n_features + (2 ** (d) if d > 0 else 0),
                        self.binning_activation,
                        self.feature_mask_function,
                        self.feature_sparsity,
                        self.learnable_sparsity,
                    ),
                )
        self.dropout = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tree_input = x
        feature_masks = []
        for d in range(self.depth):
            layer_nodes = []
            layer_feature_masks = []
            for n in range(max(2 ** (d), 1)):
                leaf_nodes, feature_mask = self._modules[f"decision_stump_{d}_{n}"](tree_input)
                layer_nodes.append(leaf_nodes)
                layer_feature_masks.append(feature_mask)
            layer_nodes = torch.cat(layer_nodes, dim=1)
            tree_input = torch.cat([x, layer_nodes], dim=1)
            feature_masks.append(layer_feature_masks)
        return self.dropout(layer_nodes), feature_masks
