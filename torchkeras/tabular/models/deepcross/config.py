from dataclasses import dataclass, field
from typing import Optional,List 

from torchkeras.tabular.config import ModelConfig

@dataclass
class DeepCrossConfig(ModelConfig):
    input_embed_max: int = field(
        default=64,
        metadata={"help": 
        "The max embedding dimension for the input categorical features. Defaults to 64."
        },
    )

    cross_type: str = field(
        default = 'matrix',
        metadata={
            "help": "cross type( vector for DCN-vector, matrix for DCN-matrix, mix for DCN-mix)",
            "choices": ['vector','matrix','mix'],
        }
    )
    
    cross_order: int = field(
        default = 2,
        metadata={
            "help": "cross order level",
            "choices": [2, 3,1],
        }
    )

    low_rank: int = field(
        default = 32,
        metadata={"help": "low rank for MOE, only used for DCN-mix"}
    )
    

    experts_num: int = field(
        default = 4,
        metadata={"help": "experts number for MOE, only used for DCN-mix"}
    )

    mlp_layers: str = field(
        default = "128-64-32",
        metadata={"help": "Hyphen-separated number of layers and units in the MLP. Defaults to 128-64-32"},
    )

    mlp_dropout: float = field(
        default = 0.25,
        metadata={"help": "The dropout ratio for  MLP. Defaults to 0.25"},
    )
    

    stacked: bool = field(
        default = False,
        metadata={"help": "stack or parallel the deep part and cross part"},
    )

    embedding_initialization: Optional[str] = field(
        default="kaiming_uniform",
        metadata={
            "help": "Initialization scheme for the embedding layers. Defaults to `kaiming`",
            "choices": ["kaiming_uniform", "kaiming_normal"],
        }
    )
    
    _module_src: str = field(default="models.deepcross")
    _model_name: str = field(default="DeepCrossModel")
    _backbone_name: str = field(default="DeepCrossBackbone")
    _config_name: str = field(default="DeepCrossConfig")
    
    