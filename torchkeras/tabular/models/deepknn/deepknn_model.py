import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from omegaconf import DictConfig
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from ..base_model import BaseModel

class nn_Lambda(nn.Module):
    def __init__(self, fn, /, **kwargs) -> None:
        super().__init__()
        self._function = fn
        self._function_kwargs = kwargs
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        return self._function(x, **self._function_kwargs)


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

class OneHotEncoder(nn.Module):
    cardinalities: Tensor

    def __init__(self, cardinalities: List[int]) -> None:
        super().__init__()
        self.register_buffer('cardinalities', torch.tensor(cardinalities))

    def forward(self, x: Tensor) -> Tensor:
        encoded_columns = [
            F.one_hot(x[..., column], cardinality)
            for column, cardinality in zip(range(x.shape[-1]), self.cardinalities)
        ]
        return torch.cat(encoded_columns, -1)
        
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

class DeepKNNModel(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.hparams = config
        
        d_numerical = self.hparams.continuous_dim
        d_main = self.hparams.hidden_size
        dropout0 = self.hparams.block_dropout0
        categories = self.hparams.categorical_cardinality
        n_classes = self.hparams.output_dim
        self.d_block = int(2.0*d_main)
        
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
        
        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [self.make_block(i > 0) 
             for i in range(self.hparams.encoder_block_num)]
        )

        #R
        self.normalization = nn.LayerNorm(d_main) if self.hparams.encoder_block_num>0 else None 
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes==1
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), nn_Lambda(lambda x: x.squeeze(-2))
            )
        )

        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, self.d_block),
            nn.ReLU(),
            nn.Dropout(dropout0),
            nn.Linear(self.d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(self.hparams.context_dropout)

        #P
        self.blocks1 = nn.ModuleList(
            [self.make_block(True) for _ in range(self.hparams.predictor_block_num)]
        )
        
        self.major_head = nn.Sequential(
            nn.LayerNorm(d_main),
            nn.ReLU(),
            nn.Linear(d_main, n_classes),
        )
        
        self.aux_head = nn.Sequential(
            nn.LayerNorm(d_main),
            nn.ReLU(),
            nn.Linear(d_main, n_classes),
        )

        self.loss = getattr(nn, self.hparams.loss)()
        
        self.search_index = None
        self.pretrain_encoder = False
        self.reset_parameters()
        
    def make_block(self, prenorm) -> nn.Sequential:
        return nn.Sequential(
                    *([nn.LayerNorm(self.hparams.hidden_size)] if prenorm else []),
                    nn.Linear(self.hparams.hidden_size, self.d_block),
                    nn.ReLU(),
                    nn.Dropout(self.hparams.block_dropout0),
                    nn.Linear(self.d_block, self.hparams.hidden_size),
                    nn.Dropout(self.hparams.block_dropout1),
        )

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  
            nn.init.uniform_(self.label_encoder.bias, -bound, bound) 
            if self.hparams.output_dim>1:
                assert isinstance(self.label_encoder[0], nn.Embedding)
                nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  

    def encode(self, inputs):
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
        assert x
        
        x = torch.cat(x, dim=1)
        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    @torch.no_grad()
    def update_search_index(self, dl_candidates):
        candidate_k_list = []
        candidate_y_list = []
        candidate_id_list = []
        n = dl_candidates.size if hasattr(dl_candidates, 'size') else len(dl_candidates)
        
        loop = tqdm(enumerate(dl_candidates, start=1),total=n,
                    disable=True,ncols=100)
        
        for i,batch in loop:
            candidate_k_list.append(self.encode(batch)[1])
            candidate_y_list.append(batch['target'])
            candidate_id_list.append(batch['id'])

        self.candidate_k = torch.cat(candidate_k_list).cpu()
        self.candidate_y = torch.cat(candidate_y_list).cpu()
        self.candidate_id = torch.cat(candidate_id_list).cpu()
        self.dl_candidates = dl_candidates
        
        num_candidate, d_main = self.candidate_k.shape
        device = self.candidate_k.device
        
        if self.search_index is None:
            import faiss
            import faiss.contrib.torch_utils
            self.search_index = faiss.IndexHNSWFlat(d_main,16) 
                
        self.search_index.reset()
        self.search_index.train(self.candidate_k)
        self.search_index.add(self.candidate_k)
        
    @torch.no_grad()
    def find_context(self, inputs):
        x, k = self.encode(inputs)
        input_id = inputs['id']
        
        context_size = self.hparams.context_size + (1 if self.training else 0) 
        distances, context_idx = self.search_index.search(k.cpu(), context_size)
        context_id = self.candidate_id[context_idx].to(input_id.device)
        distances = distances.to(input_id.device)

        #delete sample with the same id in inputs, in case of a lable leakage.
        if self.training:
            context_id = torch.where(context_id!=input_id[:,None],context_id,-1)
            distances[context_id<0] = torch.inf
            context_id = context_id.gather(-1, distances.argsort()[:, :-1])

        ds_candidates = self.dl_candidates.dataset
        batch_ids = context_id.flatten().cpu()
        contexts = ds_candidates.get_batch(batch_ids)
        contexts = {key:value.to(input_id.device) for key,value in contexts.items()}
        return contexts 

    
    def forward(
        self,
        inputs,
        contexts = None
    ):
        x, k = self.encode(inputs)

        #pretrain_encoder
        if self.pretrain_encoder:
            aux_logits = self.aux_head(x)
            output = {'logits': aux_logits}
            return output 
        
        y = inputs.get('target',None)
        if contexts is None:
            contexts = self.find_context(inputs)

        batch_size, d_main = k.shape 
        context_size = self.hparams.context_size 
        
        context_k = self.encode(contexts)[1].reshape(batch_size,context_size,d_main)
        context_y = contexts['target'].reshape(batch_size,context_size)

        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        context_y_emb = self.label_encoder(context_y[..., None])
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x = x + block(x)
        logits = self.major_head(x)
        output = {'logits': logits}
        return output 
        
    def compute_loss(self,output, y):
        y_hat = output["logits"]
        computed_loss = self.loss(y_hat.squeeze(), y.squeeze())
        return computed_loss 

