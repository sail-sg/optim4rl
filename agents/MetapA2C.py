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

from agents.MetaA2C import MetaA2C


class MetapA2C(MetaA2C):
  '''
  Implementation of Meta A2C *without* Pipeline Training:
    We set reset_interval = 1 and reset_index = -1 such that
    iter_num % reset_interval != reset_index, thus no reset.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Reset all reset_indexes to -1
    del self.reset_intervals, self.reset_indexes
    self.reset_intervals = [1] * len(self.env_names)
    self.reset_indexes = [None] * self.task_num
    for i in range(self.task_num):
      reset_indexes = [-1] * self.num_envs
      self.reset_indexes[i] = self.reshape(jnp.array(reset_indexes))