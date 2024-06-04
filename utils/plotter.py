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
import copy
import json
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks"); sns.set_context("notebook") # paper, talk, notebook
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
# Set font family, bold, and font size
font = {'size': 16} # font = {'family':'normal', 'weight':'normal', 'size': 12}
matplotlib.rc('font', **font)
# Avoid Type 3 fonts: http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


from utils.helper import make_dir
from utils.sweeper import Sweeper


class Plotter(object):
  def __init__(self, cfg):
    # Set default value for symmetric EMA (exponential moving average)
    # Note that EMA only works when merged is True
    cfg.setdefault('EMA', False)
    cfg.setdefault('ci', ('ci', 95))
    cfg.setdefault('rolling_score_window', -1)
    self.cfg = copy.deepcopy(cfg)
    # Copy parameters
    self.exp = cfg['exp']
    self.merged = cfg['merged']
    self.x_label = cfg['x_label']
    self.y_label = cfg['y_label']
    self.rolling_score_window = cfg['rolling_score_window']
    self.hue_label = cfg['hue_label']
    self.show = cfg['show']
    self.imgType = cfg['imgType']
    if type(cfg['ci']) == int:
      self.ci = ('ci', cfg['ci'])
    else:
      self.ci = cfg['ci']
    self.EMA = cfg['EMA']
    ''' Set sweep_keys:
    For a hierarchical dict, all keys along the path are cancatenated with '/' into one key.
    For example, for config_dict = {"env": {"names": ["Catch-bsuite"]}},
      key = 'env': return the whole dict {"names": ["Catch-bsuite"]},
      key = 'env/names': return ["Catch-bsuite"].
    '''
    self.sweep_keys = cfg['sweep_keys']
    self.sort_by = cfg['sort_by']
    self.ascending = cfg['ascending']
    self.loc = cfg['loc']
    self.runs = cfg['runs']
    # Get total combination of configurations
    self.total_combination = get_total_combination(self.exp)

  def merge_index(self, config_idx, mode, processed, exp=None):
    '''
    Given exp and config index, merge the results of multiple runs
    '''
    if exp is None:
      exp = self.exp
    result_list = []
    for _ in range(self.runs):
      result_file = f'./logs/{exp}/{config_idx}/result_{mode}.feather'
      # If result file exist, read and merge
      result = read_file(result_file)
      if result is not None:
        # Add config index as a column
        result['Config Index'] = config_idx
        result_list.append(result)
      config_idx += get_total_combination(exp)
    
    if len(result_list) == 0:
      return None
    
    # Do symmetric EMA (exponential moving average) only
    # when we want the original data (i.e. no processed)
    if (self.EMA) and (processed == False):
      # Get x's and y's in form of numpy arries
      xs, ys = [], []
      for result in result_list:
        xs.append(result[self.x_label].to_numpy())
        ys.append(result[self.y_label].to_numpy())
      # Moving average
      if self.rolling_score_window > 0:
        for i in range(len(xs)):
          ys[i] = moving_average(ys[i], self.rolling_score_window)
      # Do symetric EMA to get new x's and y's
      low  = max(x[0] for x in xs)
      high = min(x[-1] for x in xs)
      n = min(len(x) for x in xs)
      for i in range(len(xs)):
        new_x, new_y, _ = symmetric_ema(xs[i], ys[i], low, high, n)
        result_list[i] = result_list[i][:n]
        result_list[i].loc[:, self.x_label] = new_x
        result_list[i].loc[:, self.y_label] = new_y
    elif processed == False:
      # Moving average
      if self.rolling_score_window > 0:
        for i in range(len(result_list)):
          x, y = result_list[i][self.x_label].to_numpy(), result_list[i][self.y_label].to_numpy()
          y = moving_average(y, self.rolling_score_window)
          result_list[i].loc[:, self.x_label] = new_x
          result_list[i].loc[:, self.y_label] = new_y
      # Cut off redundant results
      n = min(len(result) for result in result_list)
      for i in range(len(result_list)):
        result_list[i] = result_list[i][:n]

    return result_list

  def get_result(self, exp, config_idx, mode, get_process_result_dict=None):
    '''
    Return: (merged, processed) result 
    - if (merged == True) or (get_process_result_dict is not None):
        Return a list of (processed) result for all runs.
    - if (merged == False):
        Return unmerged result of one single run in a list.
    '''
    if get_process_result_dict is not None:
      processed = True
    else:
      processed = False
    
    if self.merged == True or processed == True:
      # Merge results
      print(f'[{exp}]: Merge {mode} results: {config_idx}/{get_total_combination(exp)}')
      result_list = self.merge_index(config_idx, mode, processed, exp)
      if result_list is None:
        print(f'[{exp}]: No {mode} results for {config_idx}')
        return None
      # Process result
      if processed:
        print(f'[{exp}]: Process {mode} results: {config_idx}/{get_total_combination(exp)}')
        for i in range(len(result_list)):
          new_result = get_process_result_dict(result_list[i], config_idx, mode)
          result_list[i] = new_result
      return result_list
    else:
      result_file = f'./logs/{exp}/{config_idx}/result_{mode}.feather'
      result = read_file(result_file)
      if result is None:
        return None
      else:
        return [result]

  def plot_vanilla(self, data, image_path):
    '''
    Plot results for data:
      data = [result_1_list, result_2_list, ...]
      result_i_list = [result_run_1, result_run_2, ...]
      result_run_i is a Dataframe
    '''
    fig, ax = plt.subplots()
    for i in range(len(data)):
      # Convert to numpy array
      ys = []
      for result in data[i]:
        ys.append(result[self.y_label].to_numpy())
      # Put all results in a dataframe
      ys = np.array(ys)
      x_mean = data[i][0][self.x_label].to_numpy()
      runs = len(data[i])
      x = np.tile(x_mean, runs)
      y = ys.reshape((-1))
      result_df = pd.DataFrame(list(zip(x, y)), columns=['x', 'y'])
      sns.lineplot(
        data=result_df, x='x', y='y',
        estimator=self.cfg['estimator'],
        errorbar=self.ci, err_kws={'alpha':0.5},
        linewidth=1.0, label=data[i][0][self.hue_label][0]
      )
    plt.legend(loc=self.loc)
    plt.xlabel(self.x_label)
    plt.ylabel(self.y_label)
    plt.tight_layout()
    plt.savefig(image_path)
    if self.show:
      plt.show()
    plt.clf()   # clear figure
    plt.cla()   # clear axis
    plt.close() # close window

  def plot_indexList(self, indexList, mode, image_name):
    '''
    Func: Given (config index) list and mode
    - merged == True: plot merged result for all runs.
    - merged == False: plot unmerged result of one single run. 
    '''
    expIndexModeList = []
    for x in indexList:
      expIndexModeList.append([self.exp, x ,mode])
    self.plot_expIndexModeList(expIndexModeList, image_name)

  def plot_indexModeList(self, indexModeList, image_name):
    '''
    Func: Given (config index, mode) list
    - merged == True: plot merged result for all runs.
    - merged == False: plot unmerged result of one single run. 
    '''
    expIndexModeList = []
    for x in indexModeList:
      expIndexModeList.append([self.exp] + x)
    self.plot_expIndexModeList(expIndexModeList, image_name)

  def plot_expIndexModeList(self, expIndexModeList, image_name):
    '''
    Func: Given (exp, config index, mode) list
    - merged == True: plot merged result for all runs.
    - merged == False: plot unmerged result of one single run.
    '''
    # Get results
    results = []
    for exp, config_idx, mode in expIndexModeList:
      print(f'[{exp}]: Plot {mode} results: {config_idx}')
      result_list = self.get_result(exp, config_idx, mode)
      if result_list is None:
        continue
      # Modify `hue_label` value in result_list for better visualization
      for i in range(len(result_list)):
        result_list[i][self.hue_label] = result_list[i][self.hue_label].map(lambda x: f'[{exp}] {mode} {x} {config_idx}')
      results.append(result_list)
    
    make_dir(f'./logs/{self.exp}/0/')
    # Plot
    if self.merged:
      image_path = f'./logs/{self.exp}/0/{image_name}_merged.{self.imgType}'
    else:
      image_path = f'./logs/{self.exp}/0/{image_name}.{self.imgType}'
    self.plot_vanilla(results, image_path)

  def plot_results(self, mode, indexes='all'):
    '''
    Plot merged result for all config indexes
    '''
    if indexes == 'all':
      if self.merged:
        indexes = range(1, self.total_combination+1)
      else:
        indexes = range(1, self.total_combination*self.runs+1)

    for config_idx in indexes:
      print(f'[{self.exp}]: Plot {mode} results: {config_idx}/{self.total_combination}')
      # Get result
      result_list = self.get_result(self.exp, config_idx, mode)
      if result_list is None:
        continue
      # Plot
      if self.merged:
        image_path = f'./logs/{self.exp}/{config_idx}/{self.y_label}_{mode}_merged.{self.imgType}'
      else:
        image_path = f'./logs/{self.exp}/{config_idx}/{self.y_label}_{mode}.{self.imgType}'
      self.plot_vanilla([result_list], image_path)

  def csv_merged_results(self, mode, get_csv_result_dict, get_process_result_dict):
    '''
    Show results: generate a *.csv file that store all **merged** results
    '''
    new_result_list = []
    for config_idx in range(1, self.total_combination+1):
      print(f'[{self.exp}]: CSV {mode} results: {config_idx}/{self.total_combination}')
      result_list = self.get_result(self.exp, config_idx, mode, get_process_result_dict)
      if result_list is None:
        continue
      result = pd.DataFrame(result_list)
      # Get test results dict
      result_dict = get_csv_result_dict(result, config_idx, mode)
      # Expand test result dict from config dict
      for i in range(self.runs):
        config_file = f'./logs/{self.exp}/{config_idx+i*self.total_combination}/config.json'
        if os.path.exists(config_file):
          break

      with open(config_file, 'r') as f:
        config_dict = json.load(f)
        for key in self.sweep_keys:
          result_dict[key] = find_key_value(config_dict, key.split('/'))
      new_result_list.append(result_dict)

    if len(new_result_list) == 0:
      print(f'[{self.exp}]: No {mode} results')
      return
    make_dir(f'./logs/{self.exp}/0/')
    results = pd.DataFrame(new_result_list)
    # Sort by mean and se of result label value
    sorted_results = results.sort_values(by=self.sort_by, ascending=self.ascending)
    # Save sorted results into a .feather file
    sorted_results_file = f'./logs/{self.exp}/0/results_{mode}_merged.csv'
    sorted_results.to_csv(sorted_results_file, index=False)

  def csv_unmerged_results(self, mode, get_process_result_dict):
    '''
    Show results: generate a *.csv file that store all **unmerged** results
    '''
    new_result_list = []
    for config_idx in range(1, self.runs*self.total_combination+1):
      print(f'[{self.exp}]: CSV {mode} results: {config_idx}/{self.runs*self.total_combination}')
      result_file = f'./logs/{self.exp}/{config_idx}/result_{mode}.feather'
      config_file = f'./logs/{self.exp}/{config_idx}/config.json'
      result = read_file(result_file)
      if result is None or not os.path.exists(config_file):
        continue
      result = get_process_result_dict(result, config_idx, mode)
      with open(config_file, 'r') as f:
        config_dict = json.load(f)
        for key in self.sweep_keys:
          result[key] = find_key_value(config_dict, key.split('/'))
      new_result_list.append(result)
    if len(new_result_list) == 0:
      print(f'[{self.exp}]: No {mode} results')
      return
    make_dir(f'./logs/{self.exp}/0/')
    results = pd.DataFrame(new_result_list)
    # Save results into a .feather file
    results_file = f'./logs/{self.exp}/0/results_{mode}_unmerged.csv'
    results.to_csv(results_file, index=False)

  def compare_parameter(self, param_name, perf_name=None, image_name=None, constraints=[], mode='Train', stat='count', kde=False):
    '''
    Plot histograms for hyper-parameter selection.
      perf_name: the performance metric from results_{mode}.csv, such as Return (mean).
      param_name: the name of considered hyper-parameter, such lr.
      image_name: the name of the plotted image.
      constraints: a list of tuple (k, [x,y,...]). We only consider index with config_dict[k] in [x,y,...].
      mode: Train or Test.
      stat: for seaborn plot function
      kde: if True, plot all kdes (kernel density estimations) in one figure; o.w. plot histograms in different subfigures
    '''
    param_name_short = param_name.split('/')[-1]
    if image_name is None:
      image_name = param_name_short
    if perf_name is None:
      perf_name = f'{self.y_label} (mean)'
    config_file = f'./configs/{self.exp}.json'
    results_file = f'./logs/{self.exp}/0/results_{mode}_unmerged.csv'
    if kde: 
      image_path = f'./logs/{self.exp}/0/{image_name}_{mode}_kde.{self.imgType}'
    else:
      image_path = f'./logs/{self.exp}/0/{image_name}_{mode}.{self.imgType}'
    assert os.path.exists(results_file), f'{results_file} does not exist. Please generate it first with csv_unmerged_results.'
    assert os.path.exists(config_file), f'{config_file} does not exist.'
    # Load all results
    results = pd.read_csv(results_file)
    # Select results based on the constraints and param_name
    for k, vs in constraints:
      results = results.loc[lambda df: df[k].isin(vs), :]    
    results = results.loc[:, [perf_name, param_name]]
    results.rename(columns={param_name: param_name_short}, inplace=True)
    # Plot
    param_values = sorted(list(set(results[param_name_short])))
    if len(param_values) == 1 and param_values[0] == '/':
      return
    if kde: # Plot all kdes in one figure
      fig, ax = plt.subplots()
      # sns.histplot(data=results, x=perf_name, hue=param_name_short, kde=True, stat=stat, palette='bright', discrete=True)
      sns.kdeplot(data=results, x=perf_name, hue=param_name_short, palette='bright')
      ax.grid(axis='y')
    else: # Plot histograms in different subfigures
      fig, axs = plt.subplots(len(param_values), 1, sharex=True, sharey=True, figsize=(7, 3*len(param_values)))
      if len(param_values) == 1:
        axs = [axs]
      for i, param_v in enumerate(param_values):
        sns.histplot(data=results[results[param_name_short]==param_v], x=perf_name, hue=param_name_short, kde=False, stat=stat, palette='bright', ax=axs[i], discrete=True)
        axs[i].grid(axis='y')
    plt.xlabel(perf_name)
    plt.tight_layout()
    plt.savefig(image_path)
    if self.show:
      plt.show()
    plt.clf()   # clear figure
    plt.cla()   # clear axis
    plt.close() # close window


def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1.0, low_counts_threshold=0.0):
  ''' Copy from baselines.common.plot_util
  Functionality:
    perform one-sided (causal) EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]
  Arguments:
    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds
    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]
    n: int                - number of points in new x grid
    decay_steps: float    - EMA decay factor, expressed in new x grid steps.
    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN
  Returns:
    tuple sum_ys, count_ys where
      xs                  - array with new x grid
      ys                  - array of EMA of y at each point of the new x grid
      count_ys            - array of EMA of y counts at each point of the new x grid
  '''

  low = xolds[0] if low is None else low
  high = xolds[-1] if high is None else high

  assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
  assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
  assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))

  xolds, yolds = xolds.astype('float64'), yolds.astype('float64')
  luoi = 0 # last unused old index
  sum_y = 0.
  count_y = 0.
  xnews = np.linspace(low, high, n)
  decay_period = (high - low) / (n - 1) * decay_steps
  interstep_decay = np.exp(- 1. / decay_steps)
  sum_ys = np.zeros_like(xnews)
  count_ys = np.zeros_like(xnews)
  for i in range(n):
    xnew = xnews[i]
    sum_y *= interstep_decay
    count_y *= interstep_decay
    while True:
      if luoi >= len(xolds): break
      xold = xolds[luoi]
      if xold <= xnew:
        decay = np.exp(- (xnew - xold) / decay_period)
        sum_y += decay * yolds[luoi]
        count_y += decay
        luoi += 1
      else: break
    sum_ys[i] = sum_y
    count_ys[i] = count_y

  ys = sum_ys / count_ys
  ys[count_ys < low_counts_threshold] = np.nan
  return xnews, ys, count_ys

def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1.0, low_counts_threshold=0.0):
  ''' Copy from baselines.common.plot_util
  Functionality:
    Perform symmetric EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]
  Arguments:
    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds
    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]
    n: int                - number of points in new x grid
    decay_steps: float    - EMA decay factor, expressed in new x grid steps.
    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN
  Returns:
    tuple sum_ys, count_ys where
      xs        - array with new x grid
      ys        - array of EMA of y at each point of the new x grid
      count_ys  - array of EMA of y counts at each point of the new x grid

  '''
  xs, ys1, count_ys1 = one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold)
  _,  ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold)
  ys2 = ys2[::-1]
  count_ys2 = count_ys2[::-1]
  count_ys = count_ys1 + count_ys2
  ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
  ys[count_ys < low_counts_threshold] = np.nan
  xs = [int(x) for x in xs]
  return xs, ys, count_ys

def moving_average(values, window):
  # Copied from https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo_plots.ipynb
  numerator = np.nancumsum(values)
  numerator[window:] = numerator[window:] - numerator[:-window]
  denominator = np.ones(len(values)) * window
  denominator[:window] = np.arange(1, window + 1)
  smoothed = numerator / denominator
  assert values.shape == smoothed.shape
  return smoothed
  
def get_total_combination(exp):
  '''
  Get total combination of experiment configuration
  '''
  config_file = f'./configs/{exp}.json'
  assert os.path.isfile(config_file), f'[{exp}]: No config file <{config_file}>!'
  sweeper = Sweeper(config_file)
  return sweeper.config_dicts['num_combinations']

def find_key_value(config_dict, key_list):
  '''
  Find key value in config dict recursively given a key_list which represents the keys in path.
  '''
  for k in key_list:
    try:
      config_dict = config_dict[k]
    except:
      return '/'
  return config_dict

def read_file(result_file):
  if not os.path.isfile(result_file):
    print(f'[No such file <{result_file}>')
    return None
  result = pd.read_feather(result_file)
  if result is None:
    print(f'No result in file <{result_file}>')
    return None
  else:
    result = result.replace(np.nan, 0)
    return result