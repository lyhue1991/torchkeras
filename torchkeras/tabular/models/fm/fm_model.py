import torch 
from torch import nn
from torch import nn,Tensor 
import torch.nn.functional as F 

from typing import Dict,Any
from ..base_model import BaseModel

class NumEmbedding(nn.Module):
    """
    input_shape: [batch_size,features_num(n), d_in], # d_in=1
    output_shape: [batch_size,features_num(n), d_out]
    """
    
    def __init__(self, n: int, d_in: int, d_out: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(n, d_in, d_out))
        self.bias = nn.Parameter(Tensor(n, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x_num):
        # x_num: batch_size, features_num, d_in
        assert x_num.ndim == 3
        x = x_num[..., None] * self.weight[None]
        x = x.sum(-2)
        #x = torch.einsum("bfi,fij->bfj",x_num,self.weight)
        if self.bias is not None:
            x = x + self.bias[None]
        return x
    
class CatEmbedding(nn.Module):
    """
    input_shape: [batch_size,features_num], 
    output_shape: [batch_size,features_num, d_embed]
    """
    def __init__(self, categories, d_embed):
        super().__init__()
        self.embedding = nn.Embedding(sum(categories), d_embed)
        self.offsets = nn.Parameter(
                torch.tensor([0] + categories[:-1]).cumsum(0),requires_grad=False)
        
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x_cat):
        """
        :param x_cat: Long tensor of size ``(batch_size, features_num)``
        """
        x = x_cat + self.offsets[None]
        return self.embedding(x) 
    
class CatLinear(nn.Module):
    """
    input_shape: [batch_size,features_num], 
    output_shape: [batch_size,d_out]
    """
    def __init__(self, categories, d_out=1):
        super().__init__()
        self.fc = nn.Embedding(sum(categories), d_out)
        self.bias = nn.Parameter(torch.zeros((d_out,)))
        self.offsets = nn.Parameter(
                torch.tensor([0] + categories[:-1]).cumsum(0),requires_grad=False)

    def forward(self, x_cat):
        """
        :param x: Long tensor of size ``(batch_size, features_num)``
        """
        x = x_cat + self.offsets[None]
        return torch.sum(self.fc(x), dim=1) + self.bias 
    
    
class FMLayer(nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x): #注意：这里的x是公式中的 <v_i> * xi
        """
        :param x: Float tensor of size ``(batch_size, num_features, k)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix



class FMBackbone(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.hparams = config
        d_numerical = self.hparams.continuous_dim
        d_embed = self.hparams.input_embed_dim
        
        categories = self.hparams.categorical_cardinality
        n_classes = self.hparams.output_dim 
        
        self.num_linear = nn.Linear(d_numerical,n_classes) if d_numerical else None
        self.cat_linear = CatLinear(categories,n_classes) if categories else None
        
        self.num_embedding = NumEmbedding(d_numerical,1, d_embed) if d_numerical else None
        self.cat_embedding = CatEmbedding(categories, d_embed) if categories else None

        self.fm = FMLayer(reduce_sum=False)
        self.fm_linear = nn.Linear(d_embed,n_classes)
        

    def forward(self, x: Dict):
        
        x_cat,x_num = x["categorical"], x["continuous"]
        
    
        #linear part
        x = 0.0
        if self.num_linear:
            x = x + self.num_linear(x_num) 
        if self.cat_linear:
            x = x + self.cat_linear(x_cat)
        
        #interaction part 
        x_embedding = []
        if self.num_embedding:
            x_embedding.append(self.num_embedding(x_num[...,None]))
        if self.cat_embedding:
            x_embedding.append(self.cat_embedding(x_cat))
        x_embedding = torch.cat(x_embedding,dim=1)

        x = x + self.fm_linear(self.fm(x_embedding)) 
        
        return x


class FMModel(BaseModel):
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
        self._embedding_layer = nn.Identity()
        self._backbone = FMBackbone(self.hparams)
        setattr(self.backbone, "output_dim", self.hparams.output_dim)
        self._head = nn.Identity()
        
    def forward(self, x: Dict) -> Dict[str, Any]:
        x = self.embed_input(x)
        x = self.compute_backbone(x)
        return self.compute_head(x)

    def extract_embedding(self):
        raise ValueError("Extracting Embeddings is not supported by FMModel.")


