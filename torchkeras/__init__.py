__version__="3.8.0"

import sys
from .kerasmodel import KerasModel
from .summary import summary
from .kerasmodel import colorful
from .utils import seed_everything
try:
    from .lightmodel import LightModel 
except Exception:
    print("torchkeras.LightModel can't be used!",file = sys.stderr)

