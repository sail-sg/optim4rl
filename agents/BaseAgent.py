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

import numpy as np
from copy import deepcopy

import jax
from jax import random

from utils.logger import Logger
from envs.utils import make_env
from envs.spaces import Box, Discrete


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
    self.task_num = len(self.env_names)
    for i in range(self.task_num):
      env_cfg = deepcopy(cfg['env'])
      if 'reward_scaling' in env_cfg.keys() and isinstance(env_cfg['reward_scaling'], list):
        env_cfg['reward_scaling'] = env_cfg['reward_scaling'][i]
      self.envs.append(make_env(self.env_names[i], env_cfg))
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
      if isinstance(env.obs_space, Discrete):
        self.state_sizes.append(env.obs_space.n)
      else: # Box, MultiBinary
        self.state_sizes.append(int(np.prod(env.obs_space.shape)))
      # Get action info
      if isinstance(env.action_space, Discrete):
        self.action_types.append('DISCRETE')
        self.action_sizes.append(env.action_space.n)
      elif isinstance(env.action_space, Box):
        self.action_types.append('CONTINUOUS')
        self.action_sizes.append(env.action_space.shape[0])
      else:
        raise ValueError('Unknown action type.')