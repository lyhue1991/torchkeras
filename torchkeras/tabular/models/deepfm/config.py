from dataclasses import dataclass, field
from typing import Optional,List 

from torchkeras.tabular.config import ModelConfig

@dataclass
class DeepFMConfig(ModelConfig):
    
    input_embed_dim: int = field(
        default=32,
        metadata={"help": "The embedding dimension for the input categorical features. Defaults to 32"},
    )


    deep_layers: str = field(
        default = "128-64-32",
        metadata={"help": "Hyphen-separated number of layers and units in the deep MLP. Defaults to 128-64-32"},
    )

    deep_dropout: float = field(
        default = 0.1,
        metadata={"help": "The dropout ratio for deep MLP. Defaults to 0.1"},
    )
    
    embedding_initialization: Optional[str] = field(
        default="kaiming_uniform",
        metadata={
            "help": "Initialization scheme for the embedding layers. Defaults to `kaiming`",
            "choices": ["kaiming_uniform", "kaiming_normal"],
        }
    )
    
    _module_src: str = field(default="models.deepfm")
    _model_name: str = field(default="DeepFMModel")
    _backbone_name: str = field(default="DeepFMBackbone")
    _config_name: str = field(default="DeepFMConfig")
    