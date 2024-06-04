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

import jax
from jax import random
import jax.numpy as jnp

import time
import optax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from flax.training.train_state import TrainState

from components import network
from utils.logger import Logger
from utils.helper import pytree2array
from components.optim import set_optim
from utils.dataloader import load_data


class SLCollect(object):
  """
  Classification task.
  """
  def __init__(self, cfg):
    self.cfg = cfg
    self.config_idx = cfg['config_idx']
    self.logger = Logger(cfg['logs_dir'])
    self.task = cfg['task']
    self.model_name = self.cfg['model']['name']
    self.seed = random.PRNGKey(self.cfg['seed'])
    self.log_path = {
      'Train': cfg['logs_dir'] + 'result_Train.feather',
      'Test': cfg['logs_dir'] + 'result_Test.feather'
    }
    self.results = {'Train': [], 'Test': []}
    try:
      self.output_dim = cfg['model']['kwargs']['output_dim']
    except:
      self.output_dim = 10

  def createNN(self, model, model_cfg):
    NN = getattr(network, model)(**model_cfg)
    return NN
  
  def train(self):
    self.logger.info(f'Load dataset: {self.task}')
    self.seed, data_seed = random.split(self.seed)
    self.data = load_data(dataset=self.task, seed=data_seed, batch_size=self.cfg['batch_size'])
    for mode in ['Train', 'Test']:
      self.logger.info(f'Datasize [{mode}]: {len(self.data[mode]["y"])}')
    self.logger.info('Create train state ...')
    self.logger.info('Create train state: build neural network model')
    model = self.createNN(self.model_name, self.cfg['model']['kwargs'])
    self.seed, nn_seed, optim_seed = random.split(self.seed, 3)
    params = model.init(nn_seed, self.data['dummy_input'])
    self.logger.info('Create train state: set optimzer')
    optim = set_optim(self.cfg['optimizer']['name'], self.cfg['optimizer']['kwargs'], optim_seed)
    self.state = TrainState.create(
      apply_fn = jax.jit(model.apply),
      params = params,
      tx = optim
    )
    self.loss_fn = jax.jit(self.compute_loss)
    
    mode='Train'
    nan_error = False
    data_size = len(self.data[mode]['x'])
    batch_num = data_size // self.cfg['batch_size']

    self.logger.info('Start training ...')
    all_grad = []
    for epoch in range(1, self.cfg['epochs']+1):
      epoch_start_time = time.time()
      """Train for a single epoch."""
      self.seed, seed = random.split(self.seed)
      perms = random.permutation(seed, data_size)
      perms = perms[:batch_num * self.cfg['batch_size']]  # Skip incomplete batch
      perms = perms.reshape((batch_num, self.cfg['batch_size']))
      epoch_loss, epoch_perf = [], []
      for perm in perms:
        batch = {
          'x': self.data[mode]['x'][perm, ...],
          'y': self.data[mode]['y'][perm, ...]
        }
        # Forward: compute loss, performance, and gradient
        (loss, perf), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(
          self.state.params,
          self.state,
          batch
        )
        # Backward: update train state
        self.state = self.update_state(self.state, grads)
        # Log
        loss = float(jax.device_get(loss))
        perf = float(jax.device_get(perf))
        grad = pytree2array(grads)
        idxs = jnp.array(range(0, len(grad), self.cfg['agent']['data_reduce']))
        grad = jax.device_get(grad[idxs])
        # Check NaN error
        if np.isnan(loss) or np.isnan(perf):
          nan_error = True
          self.logger.info("NaN error detected!")
          break
        epoch_loss.append(loss)
        epoch_perf.append(perf)
        all_grad.append(grad)
      if nan_error:
        break
      epoch_loss = np.mean(epoch_loss)
      epoch_perf = np.mean(epoch_perf)
      # Save training result
      self.save_results(mode, epoch, epoch_loss, epoch_perf)
      # Display speed
      if (epoch % self.cfg['display_interval'] == 0) or (epoch == self.cfg['epochs']):
        speed = time.time() - epoch_start_time
        eta = (self.cfg['epochs'] - epoch) * speed / 60 if speed > 0 else -1
        self.logger.info(f'Speed={speed:.2f} (s/epoch), ETA={eta:.2f} (mins)')
        self.logger.info(f'<{self.config_idx}> {self.task} {self.model_name} [{mode}]: Epoch={epoch}, Loss={loss:.4f}, Perf={perf:.4f}')
    self.process_logs(all_grad)

  @partial(jax.jit, static_argnums=0)
  def compute_loss(self, params, state, batch):
    logits = state.apply_fn(params, batch['x'])
    one_hot = jax.nn.one_hot(batch['y'], self.output_dim)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    perf = jnp.mean(jnp.argmax(logits, -1) == batch['y'])
    return loss, perf

  @partial(jax.jit, static_argnums=0)
  def update_state(self, state, grads):
    return state.apply_gradients(grads=grads)
  
  def save_results(self, mode, epoch, loss, perf):
    """Save and display result."""
    result_dict = {
      'Task': self.task,
      'Model': self.model_name,
      'Epoch': epoch,
      'Loss': loss,
      'Perf': perf
    }
    self.results[mode].append(result_dict)
    results = pd.DataFrame(self.results[mode])
    results['Task'] = results['Task'].astype('category')
    results['Model'] = results['Model'].astype('category')
    results.to_feather(self.log_path[mode])

  def process_logs(self, agent_grad):
    # Shape to: (num_param, optimization_steps)
    agent_grad = np.array(agent_grad)
    x = np.stack(agent_grad, axis=1)
    # Save grad
    self.logger.info(f"# of param collected: {x.shape[0]}")
    np.savez(self.cfg['logs_dir']+'data.npz', x=x)
    # Plot log(abs(grad))
    grad = x.reshape(-1)
    log_abs_grad = np.log10(np.abs(grad)+1e-8)
    self.logger.info(f'g: min = {grad.min():.4f}, max = {grad.max():.4f}, mean = {grad.mean():.4f}')    
    self.logger.info(f'log(|g|+1e-8): min = {log_abs_grad.min():.4f}, max = {log_abs_grad.max():.4f}, mean = {log_abs_grad.mean():.4f}')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    ax1.hist(grad, bins=40, density=False)
    ax1.set_yscale('log')
    ax1.set_xlabel('$g$', fontsize=18)
    ax1.set_ylabel('log(counts)', fontsize=18)
    ax1.grid(True)
    # Plot log(|grad|)
    ax2.hist(log_abs_grad, bins=list(np.arange(-9, 5, 0.5)), density=True)
    ax2.set_xlim(-9, 5)
    ax2.set_xticks(list(np.arange(-9, 5, 1)))
    ax2.set_xlabel('$\log(|g|+10^{-8})$', fontsize=18)
    ax2.set_ylabel('Probability density', fontsize=18)
    ax2.grid(True)
    # Adjust figure layout
    plt.tick_params(axis='both', which='major', labelsize=14)
    fig.tight_layout()
    # Save figure
    plt.savefig(self.cfg['logs_dir']+'grad.png')
    plt.clf()
    plt.cla()
    plt.close()