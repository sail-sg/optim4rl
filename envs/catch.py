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

import jax
import jax.numpy as jnp
from jax import lax, random

from envs.spaces import Box, Discrete


class Catch(object):
  """A JAX implementation of Catch."""
  def __init__(self, rows=10, columns=5):
    self._rows = rows
    self._columns = columns
    self.num_actions = 3
    self.action_space = Discrete(3)
    self.observation_space = Box(0.0, 1.0, shape=(self._rows, self._columns, 1), dtype=bool)

  def reset(self, seed):
    ball_y = 0
    ball_x = random.randint(seed, (), 0, self._columns)
    paddle_y = self._rows - 1
    paddle_x = self._columns // 2
    state = jnp.array([ball_y, ball_x, paddle_y, paddle_x], dtype=jnp.int32)
    return lax.stop_gradient(state)

  def step(self, seed, state, action):
    # Generate the next env state: next_state
    paddle_x = jnp.clip(state[3] + action - 1, 0, self._columns - 1)
    next_state = jnp.array([state[0] + 1, state[1], state[2], paddle_x])
    # Check if next_state is a teriminal state
    done = self.is_terminal(next_state)
    # Compute the reward
    reward = self.reward(state, action, next_state, done)
    # Reset the next_state if done
    next_state = lax.select(done, self.reset(seed), next_state)
    next_state = lax.stop_gradient(next_state)
    return next_state, reward, done

  def render_obs(self, state):
    def f(y, x):
      return lax.select(
        jnp.bitwise_or(
          jnp.bitwise_and(y == state[0], x == state[1]),
          jnp.bitwise_and(y == state[2], x == state[3])
        ), 1., 0.)
    y_board = jnp.repeat(jnp.arange(self._rows), self._columns)
    x_board = jnp.tile(jnp.arange(self._columns), self._rows)
    return lax.stop_gradient(jax.vmap(f)(y_board, x_board).reshape((self._rows, self._columns, 1)))

  def reward(self, state, action, next_state, done):
    r = lax.select(done, lax.select(next_state[1] == next_state[3], 1., -1.), 0.)
    return r

  def is_terminal(self, state):
    done = lax.select(state[0] == self._rows-1, 1, 0)
    return done