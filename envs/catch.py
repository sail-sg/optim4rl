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

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, random

from envs.spaces import Discrete, Box


class Catch(object):
  """A JAX implementation of the Catch gridworld."""
  def __init__(self, rows=10, columns=5):
    self._rows = rows
    self._columns = columns
    self.num_as = 3
    self.action_space = Discrete(3)
    self.obs_space = Box(0.0, 1.0, shape=(self._rows, self._columns, 1), dtype=bool)

  @partial(jit, static_argnames=['self'])
  def reset(self, seed):
    ball_y = 0
    ball_x = random.randint(seed, (), 0, self._columns)
    paddle_y = self._rows - 1
    paddle_x = self._columns // 2
    s = jnp.array([ball_y, ball_x, paddle_y, paddle_x], dtype=jnp.int32)
    return s

  @partial(jit, static_argnames=['self'])
  def step(self, seed, s, a):
    # Generate the next env s: next_s
    paddle_x = jnp.clip(s[3]+a-1, 0, self._columns-1)
    next_s = jnp.array([s[0]+1, s[1], s[2], paddle_x])
    # Check if next_s is a teriminal s
    done = lax.select(next_s[0] == self._rows-1, 1, 0)
    # Compute the reward
    r = lax.select(done, lax.select(next_s[1] == next_s[3], 1., -1.), 0.)
    # Reset the next_s if done
    next_s = lax.select(done, self.reset(seed), next_s)
    return next_s, r, done

  @partial(jit, static_argnames=['self'])
  def render_obs(self, s):
    def f(y, x):
      fn = lax.select(
        jnp.bitwise_or(
          jnp.bitwise_and(y == s[0], x == s[1]),
          jnp.bitwise_and(y == s[2], x == s[3])
        ),
        1.,
        0.
      )
      return fn
    y_board = jnp.repeat(jnp.arange(self._rows), self._columns)
    x_board = jnp.tile(jnp.arange(self._columns), self._rows)
    return jax.vmap(f)(y_board, x_board).reshape((self._rows, self._columns, 1))