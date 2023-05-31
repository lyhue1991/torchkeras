__version__="3.8.7"

import sys
from .kerasmodel import KerasModel
from .summary import summary, flop_summary
from .utils import seed_everything,printlog,colorful

try:
    from .hugmodel import HugModel
except Exception:
    pass

