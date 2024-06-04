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

import jax.numpy as jnp

from agents.MetaPPO import MetaPPO


class MetapPPO(MetaPPO):
  '''
  PPO for Brax with meta learned optimizer without Pipeline Training.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Reset all reset_indexes to -1
    self.agent_reset_interval = -1
    reset_indexes = [1]*self.local_devices_to_use
    self.reset_indexes = self.core_reshape(jnp.array(reset_indexes))