# Copyright 2022 The TorchKeras Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the TorchKeras API, the high-level API of Pytorch.

Detailed documentation and user guides are available at
https://github.com/lyhue1991/torchkeras.
"""

__version__="4.0.2"

from torchkeras.vlog import VLog

try:
    from torchkeras.kerasmodel import KerasModel
    from torchkeras.summary import summary, flop_summary
    from torchkeras.utils import seed_everything,printlog,colorful,delete_object
except Exception as err:
    print(err)

try:
    from torchkeras import tabular
except Exception as err:
    print(err)
    
try:
    from .hugmodel import HugModel
except Exception:
    pass

