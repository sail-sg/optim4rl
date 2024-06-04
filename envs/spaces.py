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

import jax
import jax.numpy as jnp


class Discrete(object):
  '''
  Minimal jittable class for discrete gymnax spaces.
  '''
  def __init__(self, n):
    assert n >= 0
    self.n = n
    self.shape = ()
    self.dtype = jnp.int32

  def sample(self, key):
    """Sample random action uniformly from set of categorical choices."""
    return jax.random.randint(
      key, shape=self.shape, minval=0, maxval=self.n
    ).astype(self.dtype)


class Box(object):
  """
  Minimal jittable class for array-shaped gymnax spaces.
  """
  def __init__(
    self, low, high, shape, dtype=jnp.float32,
  ):
    self.low = low
    self.high = high
    self.shape = shape
    self.dtype = dtype

  def sample(self, key):
    """Sample random action uniformly from 1D continuous range."""
    return jax.random.uniform(
      key, shape=self.shape, minval=self.low, maxval=self.high
    ).astype(self.dtype)