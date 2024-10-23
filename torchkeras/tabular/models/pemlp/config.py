from dataclasses import dataclass, field
from typing import Optional,List 
from torchkeras.tabular.config import ModelConfig

@dataclass
class PeMLPConfig(ModelConfig):
    input_embed_dim: int = field(
        default=24,
        metadata={"help": "The embedding dimension for the input num/cat features. Defaults to 24"},
    )

    mlp_layers: str = field(
        default = "384-384",
        metadata={"help": "Hyphen-separated number of layers and units in the MLP. Defaults to 384-384"},
    )

    mlp_dropout: float = field(
        default= 0.4,
        metadata={"help": "mlp dropout ration"},
    )

    _module_src: str = field(default="models.pemlp")
    _model_name: str = field(default="PeMLPModel")
    _backbone_name: str = field(default="PeMLPBackbone")
    _config_name: str = field(default="PeMLPConfig")
