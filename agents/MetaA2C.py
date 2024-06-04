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

import time
import optax
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, tree_util

from components.optim import set_optim
from utils.helper import jitted_split, tree_transpose, save_model_param
from agents.A2C import A2C, MyTrainState


class MetaA2C(A2C):
  '''
  Implementation of Meta A2C
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Set meta optimizer
    self.seed, optim_seed = jitted_split(self.seed)
    self.cfg['meta_optim']['kwargs'].setdefault('max_norm', -1)
    self.max_norm = self.cfg['meta_optim']['kwargs']['max_norm']
    del self.cfg['meta_optim']['kwargs']['max_norm']
    self.meta_optim = set_optim(self.cfg['meta_optim']['name'], cfg['meta_optim']['kwargs'], optim_seed)
    # Set reset_indexes
    if isinstance(cfg['agent']['reset_interval'], int):
      self.reset_intervals = [cfg['agent']['reset_interval']] * self.task_num
    elif isinstance(cfg['agent']['reset_interval'], list):
      self.reset_intervals = cfg['agent']['reset_interval'].copy()
    else:
      raise TypeError('Only List[int] or int is allowed')
    self.reset_indexes = [None]*self.task_num
    for i in range(self.task_num):
      reset_indexes = [int(x) for x in jnp.linspace(0, self.reset_intervals[i]-1, num=self.num_envs)]
      self.reset_indexes[i] = self.reshape(jnp.array(reset_indexes))

  def abs_sq(self, x: jax.Array) -> jax.Array:
    """Returns the squared norm of a (maybe complex) array.
    Copy from https://github.com/deepmind/optax/blob/master/optax/_src/numerics.py
    """
    if not isinstance(x, (np.ndarray, jnp.ndarray)):
      raise ValueError(f"`abs_sq` accepts only NDarrays, got: {x}.")
    return (x.conj() * x).real

  def global_norm(self, updates):
    """
    Compute the global norm across a nested structure of tensors.
    Copy from https://github.com/deepmind/optax/blob/master/optax/_src/linear_algebra.py
    """
    return jnp.sqrt(sum(jnp.sum(self.abs_sq(x)) for x in tree_util.tree_leaves(updates)))
  
  @partial(jit, static_argnames=['self', 'i'])
  def agent_update(self, carry_in, i):
    '''Perform a step of inner update to the agent.'''
    meta_param, training_state, env_state, seed, lr = carry_in
    seed, step_seed = jitted_split(seed)
    # Generate one rollout and compute agent gradient
    agent_grad, (env_state, rollout) = jax.grad(self.compute_loss, has_aux=True)(training_state.agent_param, env_state, step_seed, i)
    # Update agent parameters
    param_update, new_optim_state = self.agent_optim.update_with_param(
      meta_param, agent_grad, training_state.agent_optim_state, lr
    )
    new_param = optax.apply_updates(training_state.agent_param, param_update)
    # Set new training_state
    training_state = training_state.replace(
      agent_param = new_param,
      agent_optim_state = new_optim_state
    )
    carry_out = (meta_param, training_state, env_state, seed, lr)
    return carry_out, i

  @partial(jit, static_argnames=['self', 'i'])
  def agent_update_and_meta_loss(self, meta_param, carry_in, i):
    '''Update agent param and compute meta loss with the last rollout.'''
    # Perform inner updates
    carry = (meta_param,) + carry_in
    carry, _ = lax.scan(
      f = lambda carry, _: self.agent_update(carry, i),
      init = carry,
      length = self.inner_updates,
      xs = None
    )
    meta_param, training_state, env_state, step_seed, lr = carry
    # Use the last rollout as the validation data to compute meta loss
    meta_loss, (env_state, rollout) = self.compute_loss(training_state.agent_param, env_state, step_seed, i)
    return meta_loss, (training_state, env_state)

  @partial(jit, static_argnames=['self', 'i'])
  def learn(self, carry_in, i):
    '''Two level updates for meta_param (outer update) and agent_param (inner update).'''
    meta_param, training_state, env_state, seed, lr = carry_in
    # Perform inner updates and compute meta gradient.
    seed, step_seed = jitted_split(seed)
    carry_in = (training_state, env_state, step_seed, lr)
    meta_grad, (training_state, env_state) = jax.grad(self.agent_update_and_meta_loss, has_aux=True)(meta_param, carry_in, i)
    # Reduce mean gradient across batch an cores
    meta_grad = lax.pmean(meta_grad, axis_name='batch')
    meta_grad = lax.pmean(meta_grad, axis_name='core')
    carry_out = (meta_grad, training_state, env_state, seed, lr)
    return carry_out

  @partial(jit, static_argnames=['self', 'i'])
  def get_training_state(self, seed, dummy_obs, i):
    agent_param = self.agent_nets[i].init(seed, dummy_obs)
    training_state = MyTrainState(
      agent_param = agent_param,
      agent_optim_state = self.agent_optim.init(agent_param)
    )
    return training_state

  @partial(jit, static_argnames=['self', 'i'])
  def reset_agent_training(self, training_state, env_state, reset_index, seed, optim_param, iter_num, dummy_obs, i):
    # Select the new one if iter_num % agent_reset_interval == reset_index
    f_select = lambda n_s, o_s: lax.select(iter_num % self.reset_intervals[i] == reset_index, n_s, o_s)
    # Generate a new training_state and env_state
    new_training_state = self.get_training_state(seed, dummy_obs, i)
    new_env_state = self.envs[i].reset(seed)
    # Select the new training_state 
    training_state = tree_util.tree_map(f_select, new_training_state, training_state)
    env_state = tree_util.tree_map(f_select, new_env_state, env_state)
    # Update optim_param
    agent_optim_state = training_state.agent_optim_state
    agent_optim_state = agent_optim_state.replace(optim_param=optim_param)
    training_state = training_state.replace(agent_optim_state=agent_optim_state)
    return training_state, env_state

  def train(self):
    seed = self.seed
    # Initialize pmap_train_one_iteration and carries (hidden_state, agent_param, agent_optim_state, env_states, step_seeds)
    carries = [None] * self.task_num
    dummy_obs = [None] * self.task_num
    pmap_train_one_iterations = [None] * self.task_num
    pvmap_reset_agent_training = [None] * self.task_num
    for i in range(self.task_num):
      # Generate random seeds for env and agent
      seed, env_seed, agent_seed = jitted_split(seed, num=3)
      # Initialize agent parameter and optimizer
      dummy_obs[i] = self.envs[i].render_obs(self.envs[i].reset(env_seed))[None,]
      pvmap_reset_agent_training[i] = jax.pmap(
        jax.vmap(
          self.reset_agent_training,
          in_axes = (0, 0, 0, 0, None, None, None, None),
          out_axes= (0, 0),
          axis_name = 'batch'
        ),
        in_axes = (0, 0, 0, 0, None, None, None, None),
        out_axes= (0, 0),
        axis_name = 'core',
        static_broadcasted_argnums = (7)
      )
      # We initialize core_count*batch_size different agent parameters and optimizer states.
      pvmap_get_training_state = jax.pmap(
        jax.vmap(
          self.get_training_state,
          in_axes = (0, None, None),
          out_axes = (0),
          axis_name = 'batch'
        ),
        in_axes = (0, None, None),
        out_axes = (0),
        axis_name = 'core',
        static_broadcasted_argnums = (2)
      )
      agent_seed, *agent_seeds = jitted_split(agent_seed, self.core_count * self.batch_size + 1)
      agent_seeds = self.reshape(jnp.stack(agent_seeds))
      training_states = pvmap_get_training_state(agent_seeds, dummy_obs[i], i)
      # Intialize env_states over cores and batch
      seed, *env_seeds = jitted_split(seed, self.core_count * self.batch_size + 1)
      env_states = jax.vmap(self.envs[i].reset)(jnp.stack(env_seeds))
      env_states = tree_util.tree_map(self.reshape, env_states)
      seed, *step_seeds = jitted_split(seed, self.core_count * self.batch_size + 1)
      step_seeds = self.reshape(jnp.stack(step_seeds))
      # Save in carries dict
      carries[i] = (training_states, env_states, step_seeds, self.learning_rates[i])
      # Replicate the training process over multiple cores
      pmap_train_one_iterations[i] = jax.pmap(
        jax.vmap(
          self.learn,
          in_axes = ((None, 0, 0, 0, None), None),
          out_axes = (None, 0, 0, 0, None),
          axis_name = 'batch'
        ),
        in_axes = ((None, 0, 0, 0, None), None),
        out_axes = (None, 0, 0, 0, None),
        axis_name = 'core',
        static_broadcasted_argnums = (1)
      )

    self.meta_param = self.agent_optim.optim_param
    self.meta_optim_state = self.meta_optim.init(self.meta_param)
    # Train for self.iterations for each env
    for t in range(1, self.iterations+1):
      start_time = time.time()
      meta_grads = []
      for i in range(self.task_num):
        # Reset agent training: agent_param, hidden_state, env_state
        # and update meta parameter (i.e. optim_param)
        training_states, env_states = carries[i][0], carries[i][1]
        seed, *reset_seeds = jitted_split(seed, self.core_count * self.batch_size + 1)
        reset_seeds = self.reshape(jnp.stack(reset_seeds))
        training_states, env_states = pvmap_reset_agent_training[i](training_states, env_states, self.reset_indexes[i], reset_seeds, self.meta_param, t-1, dummy_obs[i], i)
        carries[i] = list(carries[i])
        carries[i][0], carries[i][1] = training_states, env_states
        carries[i] = tuple(carries[i])
        # Train for one iteration
        carry_out = pmap_train_one_iterations[i]((self.meta_param,)+carries[i], i)
        #  Gather meta grad and update carries
        meta_grad, carries[i] = carry_out[0], carry_out[1:]
        if self.max_norm > 0:
          g_norm = self.global_norm(meta_grad)
          meta_grad = tree_util.tree_map(lambda x: (x / g_norm.astype(x.dtype)) * self.max_norm, meta_grad)
        meta_grads.append(meta_grad)
      # Update meta paramter
      meta_grad = tree_transpose(meta_grads)
      meta_grad = tree_util.tree_map(lambda x: jnp.mean(x, axis=0), meta_grad)
      # Update meta parameter
      meta_param_update, self.meta_optim_state = self.meta_optim.update(meta_grad, self.meta_optim_state)
      self.meta_param = optax.apply_updates(self.meta_param, meta_param_update)
      # Show log
      if t % self.cfg['display_interval'] == 0:
        step_count = t * self.macro_step
        speed = self.macro_step / (time.time() - start_time)
        eta = (self.train_steps - step_count) / speed / 60 if speed>0 else -1
        self.logger.info(f'<{self.config_idx}> Step {step_count}/{self.train_steps} Iteration {t}/{self.iterations}: Speed={speed:.2f} (steps/s), ETA={eta:.2f} (mins)')
      # Save meta param
      if t == self.iterations:
        save_model_param(self.meta_param, self.cfg['logs_dir']+'param.pickle')