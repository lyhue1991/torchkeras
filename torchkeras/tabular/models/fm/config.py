from dataclasses import dataclass, field
from typing import Optional

from torchkeras.tabular.config import ModelConfig

@dataclass
class FMConfig(ModelConfig):
    input_embed_dim: int = field(
        default=32,
        metadata={"help": "The embedding dimension for the input categorical features. Defaults to 32"},
    )
    
    embedding_initialization: Optional[str] = field(
        default="kaiming_uniform",
        metadata={
            "help": "Initialization scheme for the embedding layers. Defaults to `kaiming`",
            "choices": ["kaiming_uniform", "kaiming_normal"],
        }
    )
    _module_src: str = field(default="models.fm")
    _model_name: str = field(default="FMModel")
    _backbone_name: str = field(default="FMBackbone")
    _config_name: str = field(default="FMConfig")
    