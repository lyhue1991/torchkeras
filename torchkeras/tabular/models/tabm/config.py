from dataclasses import dataclass, field
from typing import Any, List, Literal, NamedTuple, Optional, Tuple, Union
from torchkeras.tabular.config import ModelConfig

@dataclass
class TabMConfig(ModelConfig):
    k: int = field(
        default=32,
        metadata={"help": "Number of parallel predictions to make (default: 32)"}
    )
    n_blocks: int = field(
        default=3,
        metadata={"help": "Number of Linear blocks in Tabm (default: 3)"}
    )
    d_block: int = field(
        default=512,
        metadata={"help": "Dimension of block in Tabm (default: 512)"}
    )
    
    plr_embed_dim: int = field(
            default=64,
            metadata={"help": "The embedding dimension of the PLREmbedding. Defaults to 64"},
    )
    
    plr_frequencies_num: int = field(
            default=96,
            metadata={"help": "The n_frequencies of the PLREmbedding. Defaults to 96"},
    )
    
    plr_frequency_scale: float = field(
            default=0.013675285379929491,
            metadata={"help": "The frequency_scale of the PLREmbedding. Defaults to 0.013675285379929491"},
    )
    dropout: float = field(default=0.1)
    scaling_init: str = field(default="normal")
    _module_src: str = field(default="models.tabm")
    _model_name: str = field(default="TabMModel")
    _config_name: str = field(default="TabMConfig")