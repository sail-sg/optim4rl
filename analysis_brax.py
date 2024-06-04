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
import math
import numpy as np
from scipy.stats import bootstrap
from collections import namedtuple

from utils.plotter import Plotter
from utils.sweeper import unfinished_index, time_info, memory_info


def get_process_result_dict(result, config_idx, mode='Test'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return'][-100:].mean(skipna=False) if mode=='Train' else result['Return'][-2:].mean(skipna=False)
  }
  return result_dict

def get_csv_result_dict(result, config_idx, mode='Train', ci=90, method='percentile'):
  return_mean = result['Return (mean)'].values.tolist()
  if len(return_mean) > 1:
    CI = bootstrap(
      (result['Return (mean)'].values.tolist(),),
      np.mean, confidence_level=ci/100,
      method=method
    ).confidence_interval
  else:
    CI = namedtuple('ConfidenceInterval', ['low', 'high'])(low=return_mean[0], high=return_mean[0])
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return (mean)'].mean(skipna=False),
    'Return (se)': result['Return (mean)'].sem(ddof=0),
    'Return (bootstrap_mean)': (CI.high + CI.low) / 2,
    f'Return (ci={ci})': (CI.high - CI.low) / 2,
  }
  return result_dict


cfg = {
  'exp': 'exp_name',
  'merged': True,
  'x_label': 'Step',
  'y_label': 'Return',
  'rolling_score_window': -1,
  'hue_label': 'Agent',
  'show': False,
  'imgType': 'png',
  'estimator': 'mean',
  'ci': ('ci', 90),
  'x_format': None,
  'y_format': None,
  'xlim': {'min': None, 'max': None},
  'ylim': {'min': None, 'max': None},
  'EMA': True,
  'loc': 'upper left',
  'sweep_keys': ['optim/name', 'optim/kwargs/learning_rate'],
  'sort_by': ['Return (mean)', 'Return (se)'],
  'ascending': [False, True],
  'runs': 1
}

def analyze(exp, runs=1):
  cfg['exp'] = exp
  cfg['runs'] = runs
  plotter = Plotter(cfg)

  sweep_keys_dict = dict(
    ppo = ['optim/name', 'optim/kwargs/learning_rate'],
    lopt = ['optim/name', 'optim/kwargs/learning_rate', 'optim/kwargs/param_load_path'],
    meta = ['agent/reset_interval', 'agent_optim/name', 'agent_optim/kwargs/learning_rate', 'meta_optim/kwargs/learning_rate', 'meta_optim/kwargs/grad_clip', 'meta_optim/kwargs/grad_norm']
  )
  algo = exp.split('_')[0].rstrip('0123456789')
  plotter.sweep_keys = sweep_keys_dict[algo]
  mode = 'Test'
  plotter.csv_merged_results(mode, get_csv_result_dict, get_process_result_dict)
  plotter.plot_results(mode, indexes='all')


if __name__ == "__main__":
  meta_ant_list = ['meta_rl_ant', 'meta_rlp_ant', 'meta_lin_ant', 'meta_l2l_ant', 'meta_star_ant']
  meta_humanoid_list = ['meta_rl_humanoid', 'meta_rlp_humanoid', 'meta_lin_humanoid', 'meta_l2l_humanoid', 'meta_star_humanoid']

  ppo_list = ['ppo_ant', 'ppo_humanoid', 'ppo_pendulum', 'ppo_walker2d']
  lopt_ant_list = ['lopt_rl_ant', 'lopt_rlp_ant', 'lopt_lin_ant', 'lopt_l2l_ant', 'lopt_star_ant']
  lopt_humanoid_list = ['lopt_rl_humanoid', 'lopt_rlp_humanoid', 'lopt_lin_humanoid', 'lopt_l2l_humanoid', 'lopt_star_humanoid']
  lopt_rl_grid_brax = ['lopt_rl_grid_ant', 'lopt_rl_grid_humanoid', 'lopt_rl_grid_pendulum', 'lopt_rl_grid_walker2d']

  exp_list, runs = meta_ant_list, 1
  exp_list, runs = lopt_ant_list, 10
  for exp in exp_list:
    unfinished_index(exp, runs=runs)
    memory_info(exp, runs=runs)
    time_info(exp, runs=runs)
    analyze(exp, runs=runs)