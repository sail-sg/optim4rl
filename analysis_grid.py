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


def get_process_result_dict(result, config_idx, mode='Train'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return'][-20:].mean(skipna=False)
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
  'EMA': True,
  'loc': 'upper left',
  'sweep_keys': ['meta_optim/kwargs/learning_rate', 'inner_updates', 'meta_net/hidden_size', 'meta_net/inner_scale', 'meta_net/input_scale', 'grad_clip', 'meta_param_path'],
  'sort_by': ['Return (mean)', 'Return (se)'],
  'ascending': [False, True],
  'runs': 1
}

def analyze(exp, runs=1):
  cfg['exp'] = exp
  cfg['runs'] = runs
  plotter = Plotter(cfg)

  modes = []
  if 'catch' in exp:
    modes.append('catch')
  if 'sds' in exp:
    modes.append("small_dense_short")
  if 'sdl' in exp:
    modes.append("small_dense_long")
  if 'bss' in exp:
    modes.append("big_sparse_short")
  if 'bds' in exp:
    modes.append("big_dense_short")
  if 'bsl' in exp:
    modes.append("big_sparse_long")
  if 'bdl' in exp:
    modes.append("big_dense_long")
  if 'grid' in exp:
    modes = ["small_dense_short", "small_dense_long", "big_sparse_short", "big_sparse_long", "big_dense_short", "big_dense_long"]

  sweep_keys_dict = dict(
    a2c = ['agent_optim/name', 'agent_optim/kwargs/learning_rate'],
    collect = ['agent_optim/name', 'agent_optim/kwargs/learning_rate', 'agent_optim/kwargs/grid_clip'],
    lopt = ['agent_optim/name', 'agent_optim/kwargs/learning_rate', 'agent_optim/kwargs/param_load_path'],
    meta = ['agent/reset_interval', 'agent_optim/name', 'meta_optim/kwargs/learning_rate', 'meta_optim/kwargs/grad_clip', 'meta_optim/kwargs/grad_norm', 'meta_optim/kwargs/max_norm']
  )
  algo = exp.split('_')[0].rstrip('0123456789')
  plotter.sweep_keys = sweep_keys_dict[algo]

  for mode in modes:
    plotter.csv_merged_results(mode, get_csv_result_dict, get_process_result_dict)
    plotter.plot_results(mode=mode, indexes='all')


if __name__ == "__main__":
  meta_catch_list = ['meta_rl_catch', 'meta_lin_catch', 'meta_l2l_catch', 'meta_star_catch']
  meta_sdl_list = ['meta_rl_sdl', 'meta_rlp_sdl']
  meta_bdl_list = ['meta_rl_bdl', 'meta_rlp_bdl', 'meta_lin_bdl', 'meta_l2l_bdl', 'meta_star_bdl']
  meta_grid_list = ['meta_rl_grid']
  
  a2c_list = ['a2c_grid', 'a2c_catch']
  lopt_catch_list = ['lopt_rl_catch', 'lopt_star_catch', 'lopt_l2l_catch', 'lopt_lin_catch']
  lopt_sdl_list = ['lopt_rl_sdl', 'lopt_rlp_sdl']
  lopt_bdl_list = ['lopt_rl_bdl', 'lopt_rlp_bdl', 'lopt_lin_bdl', 'lopt_l2l_bdl', 'lopt_star_bdl']
  
  exp_list, runs = meta_catch_list, 1
  exp_list, runs = lopt_catch_list, 10
  for exp in exp_list:
    unfinished_index(exp, runs=runs)
    memory_info(exp, runs=runs)
    time_info(exp, runs=runs)
    analyze(exp, runs=runs)