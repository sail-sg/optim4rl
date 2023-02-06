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

from utils.plotter import Plotter
from utils.sweeper import memory_info, time_info, unfinished_index


def get_process_result_dict(result, config_idx, mode='Test'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return'][-100:].mean(skipna=False) if mode=='Train' else result['Return'][-5:].mean(skipna=False)
  }
  return result_dict

def get_csv_result_dict(result, config_idx, mode='Test'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return (mean)'].mean(skipna=False),
    'Return (se)': result['Return (mean)'].sem(ddof=0)
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
  'ci': 'se',
  'x_format': None,
  'y_format': None,
  'xlim': {'min': None, 'max': None},
  'ylim': {'min': None, 'max': None},
  'EMA': True,
  'loc': 'upper left',
  'sweep_keys': ['optimizer/name', 'optimizer/kwargs/learning_rate'],
  'sort_by': ['Return (mean)', 'Return (se)'],
  'ascending': [False, True],
  'runs': 1
}

def analyze(exp, runs=1):
  cfg['exp'] = exp
  cfg['runs'] = runs
  plotter = Plotter(cfg)

  sweep_keys_dict = dict(
    ppo = ['optimizer/name', 'optimizer/kwargs/learning_rate', 'optimizer/kwargs/gradient_clip'],
    collect = ['agent_optimizer/name', 'agent_optimizer/kwargs/learning_rate'],
    lopt = ['optimizer/name', 'optimizer/kwargs/learning_rate', 'optimizer/kwargs/param_load_path', 'optimizer/kwargs/gradient_clip'],
    meta = ['agent_optimizer/name', 'agent_optimizer/kwargs/learning_rate', 'meta_optimizer/kwargs/learning_rate', 'meta_optimizer/kwargs/max_norm'],
    online = ['agent_optimizer/name', 'agent_optimizer/kwargs/learning_rate', 'meta_optimizer/kwargs/learning_rate', 'meta_optimizer/kwargs/max_norm'],
  )
  algo = exp.split('_')[-1].rstrip('0123456789')
  plotter.sweep_keys = sweep_keys_dict[algo]
  mode = 'Test'
  plotter.csv_results(mode, get_csv_result_dict, get_process_result_dict)
  plotter.plot_results(mode, indexes='all')


if __name__ == "__main__":
  """Collect"""
  # exp, runs = 'ant_collect', 1
  # exp, runs = 'fetch_collect', 1
  # exp, runs = 'grasp_collect', 1
  # exp, runs = 'halfcheetah_collect', 1
  # exp, runs = 'humanoid_collect', 1
  # exp, runs = 'humanoidstandup_collect', 1
  # exp, runs = 'pusher_collect', 1
  # exp, runs = 'reacher_collect', 1
  # exp, runs = 'ur5e_collect', 1
  """PPO"""
  # exp, runs = 'ant_ppo', 10
  # exp, runs = 'fetch_ppo', 10
  # exp, runs = 'grasp_ppo', 10
  # exp, runs = 'halfcheetah_ppo', 10
  # exp, runs = 'humanoid_ppo', 10
  # exp, runs = 'humanoidstandup_ppo', 10
  # exp, runs = 'pusher_ppo', 10
  # exp, runs = 'reacher_ppo', 10
  # exp, runs = 'ur5e_ppo', 10
  """Lopt"""
  exp, runs = 'ant_lopt', 1
  unfinished_index(exp, runs=runs)
  memory_info(exp, runs=runs)
  time_info(exp, runs=runs)
  analyze(exp, runs=runs)