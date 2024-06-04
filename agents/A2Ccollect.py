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

import optax
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit, lax, tree_util

from utils.helper import jitted_split, pytree2array
from agents.A2C import A2C, MyTrainState



class A2Ccollect(A2C):
  '''
  Collect agent gradient in A2C.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)

  @partial(jit, static_argnames=['self', 'i'])
  def learn(self, carry_in, i):
    training_state, env_state, seed = carry_in
    seed, step_seed = jitted_split(seed)
    # Generate one rollout and compute the gradient
    agent_grad, (env_state, rollout) = jax.grad(self.compute_loss, has_aux=True)(training_state.agent_param, env_state, step_seed, i)
    # Reduce mean gradients across batch an cores
    agent_grad = lax.pmean(agent_grad, axis_name='batch')
    agent_grad = lax.pmean(agent_grad, axis_name='core')
    # Compute the updates of model parameters
    param_update, new_optim_state = self.agent_optim.update(agent_grad, training_state.agent_optim_state)
    # Update model parameters
    new_param = optax.apply_updates(training_state.agent_param, param_update)
    training_state = training_state.replace(
      agent_param = new_param,
      agent_optim_state = new_optim_state
    )
    carry_out = (training_state, env_state, seed)
    # Choose part of agent_grad due to memory limit
    agent_grad = pytree2array(agent_grad)
    idxs = jnp.array(range(0, len(agent_grad), self.cfg['agent']['data_reduce']))
    agent_grad = agent_grad[idxs]
    logs = dict(done=rollout.done, reward=rollout.reward)
    return carry_out, (logs, agent_grad)

  @partial(jit, static_argnames=['self', 'i'])
  def train_iterations(self, carry_in, i):
    # Vectorize the learn function across batch
    batched_learn = jax.vmap(
      self.learn,
      in_axes=((None, 0, 0), None),
      out_axes=((None, 0, 0), (0, None)),
      axis_name='batch'
    )
    # Repeat the training for many iterations
    train_one_iteration = lambda carry, _: batched_learn(carry, i)
    carry_out, logs = lax.scan(f=train_one_iteration, init=carry_in, length=self.iterations, xs=None)
    return carry_out, logs

  def train(self):
    seed = self.seed
    for i in range(self.task_num):
      self.logger.info(f'<{self.config_idx}> Task {i+1}/{self.task_num}: {self.env_names[i]}')
      # Generate random seeds for env and agent
      seed, env_seed, agent_seed = jitted_split(seed, 3)
      # Initialize agent parameter and optimizer state
      dummy_obs = self.envs[i].render_obs(self.envs[i].reset(env_seed))[None,]
      agent_param = self.agent_nets[i].init(agent_seed, dummy_obs)
      training_state = MyTrainState(
        agent_param = agent_param,
        agent_optim_state = self.agent_optim.init(agent_param)
      )
      # Intialize env_states over cores and batch
      seed, *env_seeds = jitted_split(seed, self.core_count * self.batch_size + 1)
      env_states = jax.vmap(self.envs[i].reset)(jnp.stack(env_seeds))
      env_states = tree_util.tree_map(self.reshape, env_states)
      seed, *step_seeds = jitted_split(seed, self.core_count * self.batch_size + 1)
      step_seeds = self.reshape(jnp.stack(step_seeds))
      # Replicate the training process over multiple cores
      pmap_train_iterations = jax.pmap(
        self.train_iterations,
        in_axes = ((None, 0, 0), None),
        out_axes = ((None, 0, 0), 0), 
        axis_name = 'core',
        static_broadcasted_argnums = (1)
      )
      carry_in = (training_state, env_states, step_seeds)
      carry_out, logs = pmap_train_iterations(carry_in, i)
      # Process and save logs
      return_logs, agent_grad = logs
      return_logs['agent_grad'] = agent_grad
      self.process_logs(self.env_names[i], return_logs)

  def process_logs(self, env_name, logs):
    # Move logs to CPU, with shape {[core_count, iterations, batch_size, *]}
    logs = jax.device_get(logs)
    # Reshape to {[iterations, core_count, batch_size, *]}
    for k in logs.keys():
      logs[k] = logs[k].swapaxes(0, 1)
    # Compute episode return
    episode_return, step_list = self.get_episode_return(logs['done'], logs['reward'])
    result = {
      'Env': env_name,
      'Agent': self.agent_name,
      'Step': step_list*self.macro_step,
      'Return': episode_return
    }
    # Save logs
    self.save_logs(env_name, result)
    # Save agent_grad: (num_param, optimization_steps)
    self.logger.info(f"# of agent_param collected for {env_name}: {logs['agent_grad'].shape[0]}")
    np.savez(self.cfg['logs_dir']+'data.npz', x=logs['agent_grad'])
    # Print some grad statistics
    grad = logs['agent_grad'].reshape(-1)
    log_abs_grad = np.log10(np.abs(grad)+1e-8)
    self.logger.info(f'g: min = {grad.min():.4f}, max = {grad.max():.4f}, mean = {grad.mean():.4f}')    
    self.logger.info(f'log(|g|+1e-8): min = {log_abs_grad.min():.4f}, max = {log_abs_grad.max():.4f}, mean = {log_abs_grad.mean():.4f}')
    # Plot grad
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