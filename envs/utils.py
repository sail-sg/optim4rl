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

from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_map

import brax.envs
from envs.spaces import Box
from envs.catch import Catch
from envs.random_walk import RandomWalk
from envs.gridworld import Gridworld, GridworldConfigDict


def make_env(env_name, env_cfg):
  if env_name == 'catch':
    return Catch()
  elif env_name in GridworldConfigDict.keys():
    return Gridworld(env_name, env_cfg)
  elif env_name == 'random_walk':
    return RandomWalk()
  elif is_gymnax_env(env_name):
    import gymnax
    env, env_param = gymnax.make(env_name)
    if env_name == 'MountainCar-v0':
      object.__setattr__(env_param, 'max_steps_in_episode', 1000)
    return GymnaxWrapper(env, env_param)
  else:
    raise NameError('Please choose a valid environment name!')


def is_gymnax_env(env_name):
  if env_name in ['Pendulum-v1', 'CartPole-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Acrobot-v1']:
    return True
  if 'bsuite' in env_name:
    return True
  if 'MinAtar' in env_name:
    return True
  if 'misc' in env_name:
    return True
  return False


class GymnaxWrapper(object):
  """A wrapper for gymnax games"""
  def __init__(self, env, env_param):
    self.env = env
    self.env_param = env_param
    self.action_space = env.action_space(env_param)
    self.observation_space = env.observation_space(env_param)

  def reset(self, seed):
    obs, state = self.env.reset(seed, self.env_param)
    return state

  def step(self, seed, state, action):
    next_obs, next_state, reward, done, info = self.env.step(seed, state, action, self.env_param)
    return next_state, reward, done

  def render_obs(self, state):
    try:
      return self.env.get_obs(state)
    except Exception:
      return self.env.get_obs(state, self.env_param)

  def is_terminal(self, state):
    return self.env.is_terminal(state, self.env_param)


brax_envs = ['acrobot', 'ant', 'fast', 'fetch', 'grasp', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'reacherangle', 'swimmer', 'ur5e', 'walker2d']

def make_brax_env(env_name: str, episode_length: int=1000, action_repeat: int=1):
  assert env_name in brax_envs, f'{env_name} is not a Brax environment!'
  env = brax.envs.create(
    env_name = env_name,
    episode_length = episode_length,
    action_repeat = action_repeat,
    auto_reset = False
  )
  return BraxWrapper(env)

class BraxWrapper(object):
  """A wrapper for gymnax games"""
  def __init__(self, env):
    self.env = env
    self.action_space = Box(low=-1.0, high=1.0, shape=(self.env.action_size,))
    self.observation_space = Box(low=-jnp.inf, high=jnp.inf, shape=(self.env.observation_size,))

  def reset(self, seed):
    state = self.env.reset(seed)
    return lax.stop_gradient(state)

  def step(self, seed, state, action):
    next_state = lax.stop_gradient(self.env.step(state, action))
    reward, done = next_state.reward, next_state.done
    reset_state = self.reset(seed)
    next_state = tree_map(lambda reset_s, next_s: lax.select(done>0, reset_s, next_s), reset_state, next_state)
    return next_state, reward, done

  def render_obs(self, state):
    return state.obs