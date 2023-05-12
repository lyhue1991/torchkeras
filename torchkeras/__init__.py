__version__="3.8.4"

import sys
from .kerasmodel import KerasModel
from .summary import summary
from .utils import seed_everything,printlog,colorful
try:
    from .lightmodel import LightModel 
except Exception:
    pass

