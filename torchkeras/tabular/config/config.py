import os
import re
import numpy as np 
from dataclasses import dataclass,MISSING, make_dataclass, fields, field, asdict
from typing import Any, Dict, Iterable, List, Optional
from torchkeras.tabular.models.common import heads

def get_inferred_config(ds):
    categorical_dim = len(ds.categorical_cols)
    continuous_dim = len(ds.continuous_cols)
    if ds.task == "regression":
        output_dim = len(ds.target) if ds.target else None
    elif ds.task == "binary":
        output_dim = 1
        if ds.target:
            target_values = set(np.unique(ds.data[ds.target[0]].astype(int)))
            for value in target_values:
                assert  value in {0,1}, f'not valid targets: {target_values}'
                
    elif ds.task in ("multiclass","classification"):
        output_dim = len(np.unique(ds.data[ds.target[0]])) if ds.target else None
    else:
        raise ValueError(f"{ds.task} is an unsupported task.")
    
    categorical_cardinality = [
        int(x) + 1 for x in ds.data[ds.categorical_cols].max().values
    ]
    embedding_dims = [(x, min(50, (x + 1) // 2)) for x in categorical_cardinality]
        
    return InferredConfig(
        categorical_cols = ds.categorical_cols,
        continuous_cols = ds.continuous_cols,
        categorical_dim=categorical_dim,
        continuous_dim=continuous_dim,
        output_dim=output_dim,
        categorical_cardinality=categorical_cardinality,
        embedding_dims=embedding_dims
    )
    
def safe_merge_config(config, inferred_config):
    """Merge two configurations.

    Args:
        config: The base configuration.
        inferred_config: The custom configuration.
    Returns:
        The merged configuration.
    """
    # using base config values if exist
    inferred_config.embedding_dims = config.embedding_dims or inferred_config.embedding_dims

    fields_list1 = fields(config.__class__)
    fields_list2 = fields(inferred_config.__class__)
    
    fields_info1 = [(f.name,f.type) for f in fields_list1]
    name1_list = [x[0] for x in fields_info1]
    fields_info2 = [(f.name,f.type)
                    for f in fields_list2 if f.name not in name1_list]
    fields_info = fields_info1 + fields_info2
    MergedConfig = make_dataclass('MergedConfig', fields_info)
    
    merged_dict = asdict(config)
    merged_dict.update(asdict(inferred_config))
    
    merged_config = MergedConfig(**merged_dict)
    return merged_config

@dataclass
class InferredConfig:
    """Configuration inferred from the dataset.

    Args:
        categorical_cols (List[str]): categorical feature names
        
        continuous_cols (List[str]): continuous feature names
        
        categorical_dim (int): The number of categorical features

        continuous_dim (int): The number of continuous features

        output_dim (Optional[int]): The number of output targets

        categorical_cardinality (Optional[List[int]]): The number of unique values in categorical features

        embedding_dims (Optional[List]): The dimensions of the embedding for each categorical column as a
                list of tuples (cardinality, embedding_dim).

        embedded_cat_dim (int): The number of features or dimensions of the embedded categorical features

    """
    categorical_cols: List[str] = field(
        metadata={"help": "The names of categorical features"},
    )

    continuous_cols: List[str] = field(
        metadata={"help": "The names of continuous features"},
    )

    categorical_dim: int = field(
        metadata={"help": "The number of categorical features"},
    )
    
    continuous_dim: int = field(
        metadata={"help": "The number of continuous features"},
    )
    output_dim: Optional[int] = field(
        default=None,
        metadata={"help": "The number of output targets"},
    )
    categorical_cardinality: Optional[List[int]] = field(
        default=None,
        metadata={"help": "The number of unique values in categorical features"},
    )
    embedding_dims: Optional[List] = field(
        default=None,
        metadata={
            "help": "The dimensions of the embedding for each categorical column as a list of tuples "
            "(cardinality, embedding_dim)."
        },
    )
    embedded_cat_dim: int = field(
        init=False,
        metadata={"help": "The number of features or dimensions of the embedded categorical features"},
    )

    def __post_init__(self):
        if self.embedding_dims is not None:
            assert all(
                (isinstance(t, Iterable) and len(t) == 2) for t in self.embedding_dims
            ), "embedding_dims must be a list of tuples (cardinality, embedding_dim)"
            self.embedded_cat_dim = sum([t[1] for t in self.embedding_dims])
        else:
            self.embedded_cat_dim = 0
            

@dataclass
class ModelConfig:
    """Base Model configuration.

    Args:
        task (str): Specify whether the problem is regression or classification. `backbone` is a task which
                considers the model as a backbone to generate features. Mostly used internally for SSL and related
                tasks.. Choices are: [`regression`,`classification`,`backbone`].

        head (Optional[str]): The head to be used for the model. Should be one of the heads defined in
                `pytorch_tabular.models.common.heads`. Defaults to  LinearHead. Choices are:
                [`None`,`LinearHead`,`MixtureDensityHead`].

        head_config (Optional[Dict]): The config as a dict which defines the head. If left empty, will be
                initialized as default linear head.

        embedding_dims (Optional[List]): The dimensions of the embedding for each categorical column as a
                list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of
                the categorical column using the rule min(50, (x + 1) // 2)

        embedding_dropout (float): Dropout to be applied to the Categorical Embedding. Defaults to 0.0

        batch_norm_continuous_input (bool): If True, we will normalize the continuous layer by passing it
                through a BatchNorm layer.

        virtual_batch_size (Optional[int]): If not None, all BatchNorms will be converted to GhostBatchNorm's
                with the specified virtual batch size. Defaults to None

        loss (Optional[str]): The loss function to be applied. By Default, it is MSELoss for regression and
                CrossEntropyLoss for classification. Unless you are sure what you are doing, leave it at MSELoss
                or L1Loss for regression and CrossEntropyLoss for classification

    """

    task: str = field(
        metadata={
            "help": "Specify whether the problem is regression or classification."
            " `backbone` is a task which considers the model as a backbone to generate features."
            " Mostly used internally for SSL and related tasks.",
            "choices": ["regression", "classification", "backbone"],
        }
    )

    head: Optional[str] = field(
        default="LinearHead",
        metadata={
            "help": "The head to be used for the model. ",
            "choices": [None, "LinearHead", "MixtureDensityHead"],
        },
    )

    head_config: Optional[Dict] = field(
        default_factory=lambda: {"layers": ""},
        metadata={
            "help": "The config as a dict which defines the head."
            " If left empty, will be initialized as default linear head."
        },
    )
    embedding_dims: Optional[List] = field(
        default=None,
        metadata={
            "help": "The dimensions of the embedding for each categorical column as a list of tuples "
            "(cardinality, embedding_dim). If left empty, will infer using the cardinality of the "
            "categorical column using the rule min(50, (x + 1) // 2)"
        },
    )
    embedding_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout to be applied to the Categorical Embedding. Defaults to 0.0"},
    )
    batch_norm_continuous_input: bool = field(
        default=True,
        metadata={"help": "If True, we will normalize the continuous layer by passing it through a BatchNorm layer."},
    )
    loss: Optional[str] = field(
        default=None,
        metadata={
            "help": "The loss function to be applied. By Default it is MSELoss for regression "
            "and CrossEntropyLoss for classification. Unless you are sure what you are doing, "
            "leave it at MSELoss or L1Loss for regression and CrossEntropyLoss for classification"
        },
    )

    virtual_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "If not None, all BatchNorms will be converted to GhostBatchNorm's "
            " with this virtual batch size. Defaults to None"
        },
    )

    _module_src: str = field(default="models")
    _model_name: str = field(default="Model")
    _backbone_name: str = field(default="Backbone")
    _config_name: str = field(default="Config")

    def __post_init__(self):
        if self.task == "regression":
            self.loss = self.loss or "MSELoss"
        elif self.task =='binary':
            self.loss = self.loss or 'BCEWithLogitsLoss'
        elif self.task in ("multiclass","classification"):
            self.loss = self.loss or "CrossEntropyLoss"
        else:
            raise NotImplementedError(
                f"{self.task} is not a valid task. Should be one of "
                f"{self.__dataclass_fields__['task'].metadata['choices']}"
            )
        if self.task != "backbone":
            assert self.head in dir(heads.blocks), f"{self.head} is not a valid head"
            _head_callable = getattr(heads.blocks, self.head)
            ideal_head_config = _head_callable._config_template
            invalid_keys = set(self.head_config.keys()) - set(ideal_head_config.__dict__.keys())
            assert len(invalid_keys) == 0, f"`head_config` has some invalid keys: {invalid_keys}"

        # For Custom models, setting these values for compatibility
        if not hasattr(self, "_config_name"):
            self._config_name = type(self).__name__
        if not hasattr(self, "_model_name"):
            self._model_name = re.sub("[Cc]onfig", "Model", self._config_name)
        if not hasattr(self, "_backbone_name"):
            self._backbone_name = re.sub("[Cc]onfig", "Backbone", self._config_name)
            
    def merge_dataset_config(self,ds):
        inferred_config = get_inferred_config(ds)
        merged_config = safe_merge_config(self,inferred_config)
        return merged_config
        

