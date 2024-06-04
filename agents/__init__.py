# Copyright 2024 Garena Online Private Limited.
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

from .BaseAgent import BaseAgent
from .SLCollect import SLCollect

from .A2C import A2C
from .A2Cstar import A2Cstar
from .A2Ccollect import A2Ccollect
from .MetaA2C import MetaA2C
from .MetapA2C import MetapA2C
from .MetaA2Cstar import MetaA2Cstar

from .PPO import PPO
from .PPOstar import PPOstar
from .MetaPPO import MetaPPO
from .MetapPPO import MetapPPO
from .MetaPPOstar import MetaPPOstar