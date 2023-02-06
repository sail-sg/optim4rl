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

import chex
from typing import List
import jax.numpy as jnp
from jax import lax, random

from envs.spaces import Box, Discrete


@chex.dataclass(frozen=True)
class GridworldConfig:
  env_map: chex.Array
  empty_pos_list: chex.Array
  objects: chex.Array
  max_steps: int

@chex.dataclass
class EnvState:
  agent_pos: chex.Array
  objects_pos: chex.Array
  time: int


def string_to_bool_map(str_map: List[str]) -> chex.Array:
  """Convert string map into boolean walking map."""
  bool_map = []
  for row in str_map:
    bool_map.append([r=='#' for r in row])
  return jnp.array(bool_map)

def get_all_empty_pos(env_map: chex.Array) -> chex.Array:
  """Get all empty positions, i.e. {(x,y): env_map[x,y]==0}"""
  pos_list = []
  for x in range(env_map.shape[0]):
    for y in range(env_map.shape[1]):
      if env_map[x,y]==0:
        pos_list.append([x,y])
  return jnp.array(pos_list)


"""Grid worlds with different properties.
Descriptions:
  - State space: big/small 
  - Reward: dense/sparse
  - Horizon: long/short
  - object: [reward, terminate_prob, respawn_prob]
Note:
  Due to a feature/bug in Jax, please make sure all envs have different shapes for (str_map, objects).
"""
GridworldConfigDict = dict(
  small_sparse_short = dict(
    str_map=[
      "##########",
      "#        #",
      "#        #",
      "#        #",
      "##########",
    ],
    objects=jnp.array([
      [1.0, 0.0, 0.05],
      [-1.0, 0.5, 0.05],
    ]),
    max_steps=50
  ),
  small_sparse_long = dict(
    str_map=[
      "#####",
      "#   #",
      "#   #",
      "#   #",
      "#   #",
      "#   #",
      "#   #",
      "#   #",
      "#   #",
      "#####",
    ],
    objects=jnp.array([
      [1.0, 0.0, 0.05],
      [-1.0, 0.5, 0.05],
    ]),
    max_steps=500
  ),
  small_dense_short = dict(
    str_map=[
      "########",
      "#      #",
      "#      #",
      "#      #",
      "#      #",
      "########",
    ],
    objects=jnp.array([
      [1.0, 0.0, 0.5],
      [-1.0, 0.5, 0.5],
    ]),
    max_steps=50
  ),
  small_dense_long = dict(
    str_map=[
      "######",
      "#    #",
      "#    #",
      "#    #",
      "#    #",
      "#    #",
      "#    #",
      "######",
    ],
    objects=jnp.array([
      [1.0, 0.0, 0.5],
      [-1.0, 0.5, 0.5],
    ]),
    max_steps=500
  ),
  big_sparse_short = dict(
    str_map=[
      "##############",
      "#            #",
      "#            #",
      "#            #",
      "#            #",
      "#            #",
      "#            #",
      "#            #",
      "#            #",
      "#            #",
      "#            #",
      "##############",
    ],
    objects=jnp.array([
      [1.0, 0.0, 0.05],
      [1.0, 0.0, 0.05],
      [-1.0, 0.5, 0.05],
      [-1.0, 0.5, 0.05],
    ]),
    max_steps=50
  ),
  big_sparse_long = dict(
    str_map=[
      "############",
      "#          #",
      "#          #",
      "#          #",
      "#          #",
      "#          #",
      "#          #",
      "#          #",
      "#          #",
      "#          #",
      "#          #",
      "#          #",
      "#          #",
      "############",
    ],
    objects=jnp.array([
      [1.0, 0.0, 0.05],
      [1.0, 0.0, 0.05],
      [-1.0, 0.5, 0.05],
      [-1.0, 0.5, 0.05],
    ]),
    max_steps=500
  ),
  big_dense_short = dict(
    str_map=[
      "###############",
      "#             #",
      "#             #",
      "#             #",
      "#             #",
      "#             #",
      "#             #",
      "#             #",
      "#             #",
      "#             #",
      "###############",
    ],
    objects=jnp.array([
      [1.0, 0.0, 0.5], 
      [1.0, 0.0, 0.5], 
      [-1.0, 0.5, 0.5], 
      [-1.0, 0.5, 0.5],
    ]),
    max_steps=50
  ),
  big_dense_long = dict(
    str_map=[
      "###########",
      "#         #",
      "#         #",
      "#         #",
      "#         #",
      "#         #",
      "#         #",
      "#         #",
      "#         #",
      "#         #",
      "#         #",
      "#         #",
      "#         #",
      "#         #",
      "###########",
    ],
    objects=jnp.array([
      [1.0, 0.0, 0.5], 
      [1.0, 0.0, 0.5],
      [-1.0, 0.5, 0.5], 
      [-1.0, 0.5, 0.5],
    ]),
    max_steps=500
  )
)

class Gridworld(object):
  """
  A JAX implementation of Gridworlds.
  """
  def __init__(self, env_name, env_cfg):
    env_dict = GridworldConfigDict[env_name]
    self.env_name = env_name
    self.env_map = string_to_bool_map(env_dict['str_map'])
    self.empty_pos_list = get_all_empty_pos(self.env_map)
    self.max_steps = env_dict['max_steps']
    self.objects = env_dict['objects']
    self.num_objects = len(self.objects)
    env_cfg.setdefault('reward_scaling', 1.0)
    self.reward_scaling = env_cfg['reward_scaling']
    self.num_actions = 9
    self.action_space = Discrete(self.num_actions)
    self.move_delta = jnp.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    self.observation_space = Box(
      low = 0,
      high = 1,
      shape = (self.num_objects+1, *self.env_map.shape), 
      dtype = bool
    )

  def reset(self, seed):
    pos = random.choice(seed, self.empty_pos_list, shape=(1+self.num_objects,), replace=False)
    agent_pos = jnp.array(pos[0])
    objects_pos = jnp.array(pos[1:])
    state = EnvState(
      agent_pos=agent_pos, 
      objects_pos=objects_pos, 
      time=0
    )
    return lax.stop_gradient(state)
  
  def step(self, seed, state, action):
    # Agent move one step: if hit a wall, go back
    agent_pos = state.agent_pos + self.move_delta[action]
    agent_pos = jnp.maximum(jnp.minimum(agent_pos, jnp.array(self.env_map.shape)-1), jnp.array([0,0]))
    agent_pos = lax.select(self.env_map[agent_pos[0], agent_pos[1]]==0, agent_pos, state.agent_pos)
    # Collect objects and compute reward
    def body_func(i, carry_in):
      seed, reward, done_flag, state = carry_in
      seed_terminate, seed_respawn1, seed_respawn2, seed = random.split(seed, 4)
      # Collect the object
      is_collected = jnp.logical_and(agent_pos[0]==state.objects_pos[i][0], agent_pos[1]==state.objects_pos[i][1])
      # Compute the reward
      reward = lax.select(is_collected, reward+self.objects[i][0], reward)
      # Remove the object by changing its position to [-1,-1]
      obj_pos = lax.select(is_collected, jnp.array([-1,-1]), state.objects_pos[i])
      # Terminate with probability
      done = jnp.logical_and(is_collected, random.uniform(seed_terminate) <= self.objects[i][1])
      done_flag = lax.select(done, 1, done_flag)
      # Respawn with probability
      not_present = jnp.logical_and(obj_pos[0]<0, obj_pos[1]<0)
      respawn = jnp.logical_and(not_present, random.uniform(seed_respawn1) <= self.objects[i][2])
      empty_pos = random.choice(seed_respawn2, self.empty_pos_list, shape=(), replace=False)
      obj_pos = lax.select(respawn, empty_pos, obj_pos)
      # Set object position
      state = state.replace(objects_pos=state.objects_pos.at[i].set(obj_pos))
      carry_out = (seed, reward, done_flag, state)
      return carry_out
    reward, done_flag = 0., 0
    carry_in = (seed, reward, done_flag, state)
    seed, reward, done_flag, state = lax.fori_loop(0, self.num_objects, body_func, carry_in)
    # Generate the next env state
    next_state = EnvState(agent_pos=agent_pos, objects_pos=state.objects_pos, time=state.time+1)
    # Check if next_state is a teriminal state
    done = lax.select(done_flag, 1, self.is_terminal(next_state))
    # Reset the next_state if done
    reset_state = self.reset(seed)
    next_state.agent_pos = lax.select(done, reset_state.agent_pos, next_state.agent_pos)
    next_state.objects_pos = lax.select(done, reset_state.objects_pos, next_state.objects_pos)
    next_state.time = lax.select(done, reset_state.time, next_state.time)
    # Generate the next env state
    next_state = lax.stop_gradient(next_state)
    return next_state, self.reward_scaling*reward, done

  def render_obs(self, state):
    obs_map = jnp.zeros(self.observation_space.shape)
    # Render objects
    def body_func(i, maps):
      obj_is_present = jnp.logical_and(state.objects_pos[i][0]>=0, state.objects_pos[i][1]>=0)
      maps = maps.at[i, state.objects_pos[i][0], state.objects_pos[i][1]].set(obj_is_present)
      return maps
    obs_map = lax.fori_loop(0, self.num_objects, body_func, obs_map)
    # Render agent
    obs_map = obs_map.at[-1, state.agent_pos[0], state.agent_pos[1]].set(1)
    return lax.stop_gradient(obs_map)

  def is_terminal(self, state):
    done = lax.select(state.time >= self.max_steps, 1, 0)
    return done