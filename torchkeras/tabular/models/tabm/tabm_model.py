import torch 
from torch import nn,Tensor 
import torch.nn.functional as F 
from typing import Any,  List, Optional, Union
import tabm 

class CatEmbeddings(nn.Module):
    def __init__(
        self,
        categorical_embedding_dims,
        embedding_dropout = 0.0,
    ):
        super().__init__()
        self.categorical_embedding_dims = categorical_embedding_dims
        self.cat_embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in categorical_embedding_dims])
        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None

    def forward(self, x):
        # (B, N)
        x_cat = x
        assert x_cat.shape[1] == len(
            self.cat_embedding_layers
        ), "categorical_data must have same number of columns as categorical embedding layers"
        if x_cat.shape[1] > 0:
            embed = torch.cat(
                [
                    embedding_layer(x_cat[:, i])
                    for i, embedding_layer in enumerate(self.cat_embedding_layers)
                ],
                dim=1,
            )
        else:
            embed = torch.empty(0, 0)
        if self.embd_dropout is not None:
            embed = self.embd_dropout(embed)
        return embed
    
class PeriodicEmbeddings(nn.Module):
    def __init__(
        self, n_features: int, n_frequencies: int, frequency_scale: float
    ) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(
            torch.normal(0.0, frequency_scale, (n_features, n_frequencies))
        )
    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x


class PLREmbeddings(nn.Sequential):
    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
    ) -> None:
        super().__init__(
            PeriodicEmbeddings(n_features, n_frequencies, frequency_scale),
            nn.Linear(2 * n_frequencies, d_embedding),
            nn.ReLU(),
        )


class TabMModel(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.hparams = config
        
        d_numerical = self.hparams.continuous_dim
        categories = self.hparams.categorical_cardinality
        n_classes = self.hparams.output_dim
        d_out = n_classes if n_classes is not None else 1
        d_block = self.hparams.d_block
        n_blocks = self.hparams.n_blocks
        dropout = self.hparams.dropout
        k = self.hparams.k
        scaling_init = self.hparams.scaling_init

        self.cat_embeddings = CatEmbeddings(
            categorical_embedding_dims = self.hparams.embedding_dims
        ) if categories else None
        
        self.num_embeddings = PLREmbeddings(
            n_features = d_numerical,
            n_frequencies = self.hparams.plr_frequencies_num,
            frequency_scale = self.hparams.plr_frequency_scale,
            d_embedding = self.hparams.plr_embed_dim
        ) if d_numerical>0 else None

        d_in = d_numerical * self.hparams.plr_embed_dim + self.hparams.embedded_cat_dim
        self.tabm_core = nn.Sequential(
            tabm.EnsembleView(k=k),   # (B, d_in) -> (B, k, d_in)
            tabm.MLPBackboneBatchEnsemble(
                d_in=d_in,
                n_blocks=n_blocks,
                d_block=d_block,
                dropout=dropout,
                k=k,
                tabm_init=True,
                scaling_init=scaling_init,
                start_scaling_init_chunks=None
            ),
            tabm.LinearEnsemble(512, d_out, k=k) # -> (B, k, d_out)
        )
        self.loss = getattr(nn, self.hparams.loss)()


    def forward(self, inputs):
        x_cat,x_num = inputs["categorical"], inputs["continuous"]
        x = []
        if self.num_embeddings is not None:
            x.append(
                x_num
                if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
            
        if self.cat_embeddings is not None:
            x.append(self.cat_embeddings(x_cat))

        x = torch.cat(x, dim=1)
        logits = self.tabm_core(x)
        
        with torch.no_grad():
            if self.hparams.task == 'binary':
                probs = torch.sigmoid(logits)
            elif self.hparams.task == 'multiclass':
                probs = torch.softmax(logits, dim=-1)
            else:
                probs = logits
            yhat = probs.mean(1)

        return {"logits": logits, "yhat":yhat}

    def compute_loss(self, output, target):
        y_pred = output["logits"]
        # TabM产生k个预测，需要分别训练
        # 回归: (batch_size, k) -> (batch_size * k,)
        # 分类: (batch_size, k, n_classes) -> (batch_size * k, n_classes)
        y_pred = y_pred.flatten(0, 1)
        # 复制目标以匹配预测的形状
        # (batch_size,) -> (batch_size * k,)
        y_true = target.repeat_interleave(self.hparams.k)
        computed_loss = self.loss(y_pred.squeeze(), y_true.squeeze())
        return computed_loss
    
    
    def predict(self, batch):
        self.eval()
        output = self.forward(batch)
        return output['yhat']
    