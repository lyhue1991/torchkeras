# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Base Model."""
import importlib
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd 
import torch
import torch.nn as nn

from torch import Tensor
from torch.optim import Optimizer

from torchkeras.tabular.models.common.heads import blocks
from torchkeras.tabular.models.common.layers import PreEncoded1dLayer
from torchkeras.tabular.utils import reset_all_weights

class BaseModel(nn.Module,metaclass=ABCMeta):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        """Base Model for PyTorch Tabular.

        Args:
            config (dataclass): The configuration for the model.
            kwargs (Dict, optional): Additional keyword arguments.

        """
        super().__init__()
        self.hparams = config
        self._build_network()
        self._setup_loss()

    @abstractmethod
    def _build_network(self):
        pass

    @property
    def backbone(self):
        raise NotImplementedError("backbone property needs to be implemented by inheriting classes")

    @property
    def embedding_layer(self):
        raise NotImplementedError("embedding_layer property needs to be implemented by inheriting classes")

    @property
    def head(self):
        raise NotImplementedError("head property needs to be implemented by inheriting classes")

    def _get_head_from_config(self):
        _head_callable = getattr(blocks, self.hparams.head)
        return _head_callable(
            in_units=self.backbone.output_dim,
            output_dim=self.hparams.output_dim,
            config=_head_callable._config_template(**self.hparams.head_config),
        )  # output_dim auto-calculated from other configs

    def _setup_loss(self):
        try:
            self.loss = getattr(nn, self.hparams.loss)()
        except AttributeError as e:
            print(f"{self.hparams.loss} is not a valid loss defined in the torch.nn module")
            raise e
       
    def compute_loss(self, output: Dict, y: torch.Tensor) -> torch.Tensor:
        """Calculates the loss for the model.

        Args:
            output (Dict): The output dictionary from the model
            y (torch.Tensor): The target tensor

        Returns:
            torch.Tensor: The loss value

        """
        y_hat = output["logits"]
        reg_terms = [k for k, v in output.items() if "regularization" in k]
        reg_loss = 0
        for t in reg_terms:
            if output[t] != 0:
                reg_loss += output[t]
        if self.hparams.task == "regression":
            computed_loss = reg_loss
            for i in range(self.hparams.output_dim):
                _loss = self.loss(y_hat[:, i], y[:, i])
                computed_loss += _loss
        else:
            # TODO loss fails with batch size of 1?
            computed_loss = self.loss(y_hat.squeeze(), y.squeeze()) + reg_loss
        return computed_loss

    
    def data_aware_initialization(self, dataloader):
        """Performs data-aware initialization of the model before fitted"""
        pass

    def compute_backbone(self, x: Dict) -> torch.Tensor:
        # Returns output
        x = self.backbone(x)
        return x

    def embed_input(self, x: Dict) -> torch.Tensor:
        return self.embedding_layer(x)

    def pack_output(self, y_hat: torch.Tensor, backbone_features: torch.tensor) -> Dict[str, Any]:
        """Packs the output of the model.

        Args:
            y_hat (torch.Tensor): The output of the model

            backbone_features (torch.tensor): The backbone features

        Returns:
            The packed output of the model

        """
        # if self.head is the Identity function it means that we cannot extract backbone features,
        # because the model cannot be divide in backbone and head (i.e. TabNet)
        if type(self.head) is nn.Identity:
            return {"logits": y_hat}
        return {"logits": y_hat, "backbone_features": backbone_features}

    def compute_head(self, backbone_features: Tensor) -> Dict[str, Any]:
        """Computes the head of the model.

        Args:
            backbone_features (Tensor): The backbone features

        Returns:
            The output of the model

        """
        y_hat = self.head(backbone_features)
        return self.pack_output(y_hat, backbone_features)

    def forward(self, x: Dict) -> Dict[str, Any]:
        """The forward pass of the model.

        Args:
            x (Dict): The input of the model with 'continuous' and 'categorical' keys

        """
        x = self.embed_input(x)
        x = self.compute_backbone(x)
        return self.compute_head(x)

    @torch.no_grad()
    def predict(self, x: Dict, ret_model_output: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Predicts the output of the model.

        Args:
            x (Dict): The input of the model with 'continuous' and 'categorical' keys

            ret_model_output (bool): If True, the method returns the output of the model

        Returns:
            The output of the model

        """
        ret_value = self.forward(x)
        if ret_model_output:
            return ret_value.get("logits"), ret_value
        return ret_value.get("logits")

    def extract_embedding(self):
        """Extracts the embedding of the model.

        This is used in `CategoricalEmbeddingTransformer`

        """
        if self.hparams.categorical_dim > 0:
            if not isinstance(self.embedding_layer, PreEncoded1dLayer):
                return self.embedding_layer.cat_embedding_layers
            else:
                raise ValueError(
                    "Cannot extract embedding for PreEncoded1dLayer. Please use a different embedding layer."
                )
        else:
            raise ValueError(
                "Model has been trained with no categorical feature and therefore can't be used"
                " as a Categorical Encoder"
            )

    def reset_weights(self):
        reset_all_weights(self.backbone)
        reset_all_weights(self.head)
        reset_all_weights(self.embedding_layer)

    def feature_importance(self) -> pd.DataFrame:
        """Returns a dataframe with feature importance for the model."""
        if hasattr(self.backbone, "feature_importance_"):
            imp = self.backbone.feature_importance_
            n_feat = len(self.hparams.categorical_cols + self.hparams.continuous_cols)
            if self.hparams.categorical_dim > 0:
                if imp.shape[0] != n_feat:
                    # Combining Cat Embedded Dimensions to a single one by averaging
                    wt = []
                    norm = []
                    ft_idx = 0
                    for _, embd_dim in self.hparams.embedding_dims:
                        wt.extend([ft_idx] * embd_dim)
                        norm.append(embd_dim)
                        ft_idx += 1
                    for _ in self.hparams.continuous_cols:
                        wt.extend([ft_idx])
                        norm.append(1)
                        ft_idx += 1
                    imp = np.bincount(wt, weights=imp) / np.array(norm)
                else:
                    # For models like FTTransformer, we dont need to do anything
                    # It takes categorical and continuous as individual 2-D features
                    pass
            importance_df = pd.DataFrame(
                {
                    "Features": self.hparams.categorical_cols + self.hparams.continuous_cols,
                    "importance": imp,
                }
            )
            return importance_df
        else:
            raise ValueError("Feature Importance unavailable for this model.")

