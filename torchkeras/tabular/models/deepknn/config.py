from dataclasses import dataclass, field
from typing import Optional,List 
from torchkeras.tabular.config import ModelConfig

@dataclass
class DeepKNNConfig(ModelConfig):
    hidden_size: int = field(
        default= 128,
        metadata={"help": "The basic hidden size of the model"},
    )
    
    context_size: int = field(
        default= 96,
        metadata={"help": "The number of similar neighours to be considered."},
    )

    context_dropout: float = field(
        default = 0.16611582749898343,
        metadata={"help": "Hyphen-separated number of layers and units in the MLP. Defaults to 384-384"},
    )

    encoder_block_num: int = field(
        default= 0,
        metadata={"help": "The number of encoder block num"},
    )

    predictor_block_num: int = field(
        default= 2,
        metadata={"help": "The number of predictor block num"},
    )

    block_dropout0: float = field(
        default= 0.09432953364047161,
        metadata={"help": "the first block dropout ration "},
    )

    block_dropout1: float = field(
        default= 0.0,
        metadata={"help": "the secend block dropout ration"},
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
    
    