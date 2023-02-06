# Copyright 2022 Garena Online Private Limited.
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

from .PPO import PPO
from .A2C import A2C
from .A2C2 import A2C2
from .RNNIndentity import RNNIndentity

from .CollectPPO import CollectPPO
from .CollectA2C import CollectA2C

from .MetaPPO import MetaPPO
from .MetaA2C import MetaA2C
from .StarA2C import StarA2C