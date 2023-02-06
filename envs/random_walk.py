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
from jax import lax

from envs.spaces import Box, Discrete


class RandomWalk(object):
  """
  A JAX implementation of Random Walk (Example 6.2 in Rich Sutton's RL introduction book).
    T <--- A(0) <---> B(1) <---> C(2) <---> D(3) <---> E(4) ---> T
    True values without discount: A(1/6), B(2/6), C(3/6), D(4/6), E(5/6)
  """
  def __init__(self):
    self.num_actions = 2
    self.num_states = 5
    self.action_space = Discrete(2)
    self.observation_space = Box(0, 1, shape=(5,), dtype=bool)

  def reset(self, seed):
    state = jnp.array(2)
    return lax.stop_gradient(state)

  def step(self, seed, state, action):
    # Generate the next env state: next_state
    next_state = jnp.array(state+2*action-1)
    # Check if next_state is a teriminal state
    done = self.is_terminal(next_state)
    # Compute the reward
    reward = lax.select(next_state==self.num_states, 1.0, 0.0)
    # Reset the next_state if done
    next_state = lax.select(done, self.reset(seed), next_state)
    next_state = lax.stop_gradient(next_state)
    return next_state, reward, done

  def render_obs(self, state):
    obs = jax.nn.one_hot(state, self.num_states)
    return lax.stop_gradient(obs)

  def is_terminal(self, state):
    done = jnp.logical_or(state==-1, state==self.num_states)
    return done