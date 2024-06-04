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
import jax.numpy as jnp
from jax import jit, lax, tree_util

import time
import flax
import optax
from typing import Any
from functools import partial

from utils.helper import jitted_split, tree_transpose, save_model_param
from agents.MetaA2C import MetaA2C


@flax.struct.dataclass
class StarTrainingState:
  agent_param: Any
  agent_optim_param: Any
  agent_optim_state: optax.OptState


class MetaA2Cstar(MetaA2C):
  '''
  Implementation of Meta A2C with STAR optmizer only
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    assert self.task_num == 1, 'Only single task training is supported in MetaA2Cstar for now.'

  @partial(jit, static_argnames=['self', 'i'])
  def agent_update(self, carry_in, i):
    '''Perform a step of inner update to the agent.'''
    meta_param, training_state, env_state, seed, lr = carry_in
    seed, step_seed = jitted_split(seed)
    # Generate one rollout and compute agent gradient
    (agent_loss, (env_state, rollout)), agent_grad = jax.value_and_grad(self.compute_loss, has_aux=True)(training_state.agent_param, env_state, step_seed, i)
    # Update agent parameters
    agent_optim_state = self.agent_optim.update_with_param(
      meta_param, agent_grad, training_state.agent_optim_state, agent_loss
    )
    # Set new training_state
    training_state = training_state.replace(
      agent_param = agent_optim_state.params,
      agent_optim_state = agent_optim_state
    )
    carry_out = (meta_param, training_state, env_state, seed, lr)
    return carry_out, i

  @partial(jit, static_argnames=['self', 'i'])
  def get_training_state(self, seed, dummy_obs, i):
    agent_param = self.agent_nets[i].init(seed, dummy_obs)
    training_state = StarTrainingState(
      agent_param = agent_param,
      agent_optim_param = self.agent_optim.get_optim_param(),
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
    training_state = training_state.replace(agent_optim_param=optim_param)
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
      carries[i] = (training_states, env_states, step_seeds, -1)
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

    self.meta_param = self.agent_optim.get_optim_param()
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
      # Reset agent_optim with new meta_param
      self.agent_optim.reset_optimizer(self.meta_param)
      # Show log
      if t % self.cfg['display_interval'] == 0:
        step_count = t * self.macro_step
        speed = self.macro_step / (time.time() - start_time)
        eta = (self.train_steps - step_count) / speed / 60 if speed>0 else -1
        self.logger.info(f'<{self.config_idx}> Step {step_count}/{self.train_steps} Iteration {t}/{self.iterations}: Speed={speed:.2f} (steps/s), ETA={eta:.2f} (mins)')
      # Save meta param
      if t == self.iterations:
        save_model_param(self.meta_param, self.cfg['logs_dir']+'param.pickle')