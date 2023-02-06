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
import pickle
import numpy as np
import jax.numpy as jnp
from copy import deepcopy
from jax import lax, random, tree_util

from utils.logger import Logger
from envs.utils import make_env
from envs.spaces import Box, Discrete

from gymnax.environments.spaces import Box as gymnax_Box
from gymnax.environments.spaces import Discrete as gymnax_Discrete


class BaseAgent(object):
  def __init__(self, cfg):
    self.cfg = cfg
    self.logger = Logger(cfg['logs_dir'])
    self.seed = random.PRNGKey(self.cfg['seed'])
    # Create all environments
    if isinstance(cfg['env']['name'], str):
      self.env_names = [cfg['env']['name']]
    elif isinstance(cfg['env']['name'], list):
      self.env_names = cfg['env']['name']
    else:
      raise TypeError('Only List[str] or str is allowed')
    del cfg['env']['name']
    # Make envs
    self.envs = []
    for i in range(len(self.env_names)):
      env_name = self.env_names[i]
      env_cfg = deepcopy(cfg['env'])
      if 'reward_scaling' in env_cfg.keys() and isinstance(env_cfg['reward_scaling'], list):
        env_cfg['reward_scaling'] = env_cfg['reward_scaling'][i]
      self.envs.append(make_env(env_name, env_cfg))
    # Get action_types, action_sizes, and state_sizes
    self.get_env_info()
    # Create agent networks
    self.agent_nets = self.create_agent_nets()
    self.agent_name = cfg['agent']['name']
    self.config_idx = cfg['config_idx']
    self.discount = cfg['discount']
    self.train_steps = int(cfg['env']['train_steps'])
    self.log_path = lambda env: self.cfg['logs_dir'] + f'result_{env}.feather'
    # Get available cores
    self.core_count = jax.device_count()
    self.logger.info(f'Number of cores: {self.core_count}')
    # Set batch size
    self.num_envs = self.cfg['env']['num_envs']
    assert self.num_envs % self.core_count == 0
    self.batch_size = self.num_envs // self.core_count
    # Compute the number of training iterations
    self.inner_updates = int(cfg['agent']['inner_updates'])
    self.rollout_steps = int(cfg['agent']['rollout_steps'])
    self.macro_step = int(self.num_envs * self.rollout_steps * (self.inner_updates+1))
    self.iterations = int(self.train_steps // self.macro_step)
    self.train_steps = self.iterations * self.macro_step

  def create_agent_nets(self):
    pass

  def get_env_info(self):
    self.action_types, self.action_sizes, self.state_sizes = [], [], []
    for env in self.envs:
      # Get state info
      if isinstance(env.observation_space, Discrete) or isinstance(env.observation_space, gymnax_Discrete):
        self.state_sizes.append(env.observation_space.n)
      else: # Box, MultiBinary
        self.state_sizes.append(int(np.prod(env.observation_space.shape)))
      # Get action info
      if isinstance(env.action_space, Discrete) or isinstance(env.action_space, gymnax_Discrete):
        self.action_types.append('DISCRETE')
        self.action_sizes.append(env.action_space.n)
      elif isinstance(env.action_space, Box) or isinstance(env.action_space, gymnax_Box):
        self.action_types.append('CONTINUOUS')
        self.action_sizes.append(env.action_space.shape[0])
      else:
        raise ValueError('Unknown action type.')

  def pytree2array(self, values):
    leaves = tree_util.tree_leaves(lax.stop_gradient(values))
    a = jnp.concatenate(leaves, axis=None)
    return a
    
  def save_model_param(self, model_param, filepath):
    f = open(filepath, 'wb')
    pickle.dump(model_param, f)
    f.close()

  def load_model_param(self, filepath):
    f = open(filepath, 'rb')
    model_param = pickle.load(f)
    model_param = tree_util.tree_map(jnp.array, model_param)
    f.close()
    return model_param