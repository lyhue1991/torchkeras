__version__="3.7.2"

import sys
from .kerasmodel import KerasModel
from .summary import summary
from .kerasmodel import colorful
try:
    from .lightmodel import LightModel 
except Exception:
    print("torchkeras.LightModel can't be used!",file = sys.stderr)

