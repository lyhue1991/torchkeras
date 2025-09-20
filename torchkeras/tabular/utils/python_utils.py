import math
import textwrap
from pathlib import Path
from typing import IO, Any, Callable, Dict, Optional, Union

import numpy as np
import torch


import torchkeras.tabular as root_module


_PATH = Union[str, Path]
_DEVICE = Union[torch.device, str, int]
_MAP_LOCATION_TYPE = Optional[Union[_DEVICE, Callable[[_DEVICE], _DEVICE], Dict[_DEVICE, _DEVICE]]]


def getattr_nested(_module_src, _model_name):
    module = root_module
    for m in _module_src.split("."):
        module = getattr(module, m)
    return getattr(module, _model_name)


def ifnone(arg, default_arg):
    return default_arg if arg is None else arg


def generate_doc_dataclass(dataclass, desc=None, width=100):
    if desc is not None:
        doc_str = f"{desc}\nArgs:"
    else:
        doc_str = "Args:"
    for key in dataclass.__dataclass_fields__.keys():
        if key.startswith("_"):  # Skipping private fields
            continue
        atr = dataclass.__dataclass_fields__[key]
        if atr.init:
            type = str(atr.type).replace("<class '", "").replace("'>", "").replace("typing.", "")
            help_str = atr.metadata.get("help", "")
            if "choices" in atr.metadata.keys():
                help_str += ". Choices are:" f" [{','.join(['`'+str(ch)+'`' for ch in atr.metadata['choices']])}]."
            # help_str += f'. Defaults to {atr.default}'
            h_str = textwrap.fill(
                f"{key} ({type}): {help_str}",
                width=width,
                subsequent_indent="\t\t",
                initial_indent="\t",
            )
            h_str = f"\n{h_str}\n"
            doc_str += h_str
    return doc_str



def check_numpy(x):
    """Makes sure x is a numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


def int_to_human_readable(number: int, round_number=True) -> str:
    millnames = ["", " T", " M", " B", " T"]
    n = float(number)
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )
    if round_number:
        return f"{int(n / 10 ** (3 * millidx))}{millnames[millidx]}"
    else:
        return f"{n / 10 ** (3 * millidx):.2f}{millnames[millidx]}"


def available_models():
    from torchkeras.tabular import models
    return [cl for cl in dir(models) if "config" in cl.lower()]

