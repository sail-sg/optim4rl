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

import os
import sys
import copy
import time
import json
import numpy as np

import agents
from utils.helper import set_random_seed, rss_memory_usage


class Experiment(object):
  '''
  Train the agent to play the game.
  '''
  def __init__(self, cfg):
    self.cfg = copy.deepcopy(cfg)
    self.config_idx = cfg['config_idx']
    self.agent_name = cfg['agent']['name']
    if self.cfg['generate_random_seed']:
      self.cfg['seed'] = np.random.randint(int(1e6))
    self.model_path = self.cfg['model_path']
    self.cfg_path = self.cfg['cfg_path']
    self.save_config()

  def run(self):
    '''
    Run the game for multiple steps
    '''
    self.start_time = time.time()
    set_random_seed(self.cfg['seed'])
    self.agent = getattr(agents, self.agent_name)(self.cfg)
    # Train && Test
    self.agent.train()
    self.end_time = time.time()
    self.agent.logger.info(f'Memory usage: {rss_memory_usage():.2f} MB')
    self.agent.logger.info(f'Time elapsed: {(self.end_time-self.start_time)/60:.4f} minutes')

  def save_config(self):
    cfg_json = json.dumps(self.cfg, indent=2)
    f = open(self.cfg_path, 'w')
    f.write(cfg_json)
    f.close()