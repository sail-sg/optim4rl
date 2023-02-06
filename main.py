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

import sys
import argparse

from experiment import Experiment
from utils.helper import make_dir
from utils.sweeper import Sweeper

# from jax.config import config
# config.update('jax_disable_jit', True)
# config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)
# config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
# Set a specific platform
# config.update('jax_platform_name', 'cpu')

# Fake devices
# import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'


def main(argv):
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--config_file', type=str, default='./configs/sds_a2c.json', help='Configuration file')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  args = parser.parse_args()
  
  sweeper = Sweeper(args.config_file)
  cfg = sweeper.generate_config_for_idx(args.config_idx)
  # Set config dict default value
  cfg.setdefault('agent', dict())
  cfg['agent'].setdefault('inner_updates', 0)
  cfg.setdefault('save_param', 0)
  
  # Set experiment name and log paths
  cfg['exp'] = args.config_file.split('/')[-1].split('.')[0]
  cfg['logs_dir'] = f"./logs/{cfg['exp']}/{cfg['config_idx']}/"
  make_dir(f"./logs/{cfg['exp']}/{cfg['config_idx']}/")
  cfg['cfg_path'] = cfg['logs_dir'] + 'config.json'

  exp = Experiment(cfg)
  exp.run()

if __name__=='__main__':
  main(sys.argv)