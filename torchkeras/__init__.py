__version__="3.8.1"

import sys
from .kerasmodel import KerasModel
from .summary import summary
from .kerasmodel import colorful
from .utils import seed_everything
try:
    from .lightmodel import LightModel 
except Exception:
    pass

