# Copyright 2023 The Brax Authors.
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

"""Proximal policy optimization training.
See: https://arxiv.org/pdf/1707.06347.pdf
"""

import time
import flax
import optax
import functools
from typing import Tuple, Any

import jax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_map

from brax import envs
from brax.training import acting, types
from brax.training.types import PRNGKey
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import specs

from components import running_statistics
from components.optim import set_optim
from components import gradients
from utils.helper import jitted_split, save_model_param, pytree2array
from agents.PPO import PPO


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  agent_optim_state: Any
  agent_param: ppo_losses.PPONetworkParams
  normalizer_param: running_statistics.RunningStatisticsState


class MetaPPO(PPO):
  '''
  PPO for Brax with meta learned optimizer.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Agent reset interval
    self.agent_reset_interval = self.cfg['agent']['reset_interval']
    reset_indexes = [int(x) for x in jnp.linspace(0, self.agent_reset_interval-1, num=self.local_devices_to_use)]
    self.reset_indexes = self.core_reshape(jnp.array(reset_indexes))

  def train(self):
    # Env
    environment = self.env
    num_timesteps = self.train_steps
    episode_length = self.cfg['env']['episode_length']
    action_repeat = self.cfg['env']['action_repeat']
    reward_scaling = self.cfg['env']['reward_scaling']
    num_envs = self.cfg['env']['num_envs']
    normalize_observations = self.cfg['env']['normalize_obs']
    # Agent
    network_factory = ppo_networks.make_ppo_networks
    gae_lambda = self.cfg['agent']['gae_lambda']
    unroll_length = self.cfg['agent']['rollout_steps']
    num_minibatches = self.cfg['agent']['num_minibatches']
    clipping_epsilon = self.cfg['agent']['clipping_epsilon']
    update_epochs = self.cfg['agent']['update_epochs']
    entropy_cost = self.cfg['agent']['entropy_weight']
    normalize_advantage = True
    # Meta learning
    inner_updates = self.cfg['agent']['inner_updates']
    # Others
    batch_size = self.cfg['batch_size']
    discounting = self.cfg['discount']
    seed = self.cfg['seed']
    
    """PPO training."""
    device_count = self.local_devices_to_use * self.process_count
    assert num_envs % device_count == 0
    assert batch_size * num_minibatches % num_envs == 0
    # The number of environment steps executed for every training step.
    env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat
    meta_env_step_per_training_step = max(batch_size // num_envs, 1) * num_envs * unroll_length * 1 * action_repeat
    total_env_step_per_training_step = env_step_per_training_step * inner_updates
    
    # The number of training_step calls per training_epoch call.
    self.iterations = num_timesteps // total_env_step_per_training_step
    self.logger.info(f'meta_env_step_per_training_step = {meta_env_step_per_training_step}')
    self.logger.info(f'total_env_step_per_training_step = {total_env_step_per_training_step}')
    self.logger.info(f'total iterations = {self.iterations}')
    
    # Prepare keys
    # key_networks should be global so that
    # the initialized networks are the same for different processes.
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jitted_split(key)
    local_key = jax.random.fold_in(local_key, self.process_id)
    local_key, key_env, key_reset = jitted_split(local_key, 3)
    key_agent_param, key_agent_optim, key_meta_optim = jitted_split(global_key, 3)
    del key, global_key
    key_envs = jitted_split(key_env, num_envs // self.process_count)
    # Reshape to (local_devices_to_use, num_envs // process_count, 2)
    key_envs = jnp.reshape(key_envs, (self.local_devices_to_use, -1) + key_envs.shape[1:])

    # Set training and evaluation env
    env = envs.training.wrap(
      environment,
      episode_length = episode_length,
      action_repeat = action_repeat,
      randomization_fn = None
    )
    reset_fn = jax.pmap(
      env.reset,
      axis_name = self._PMAP_AXIS_NAME
    )
    env_states = reset_fn(key_envs)
    obs_shape = env_states.obs.shape
    
    # Set agent and meta optimizer
    agent_optim = set_optim(self.cfg['agent_optim']['name'], self.cfg['agent_optim']['kwargs'], key_agent_optim)
    meta_optim = set_optim(self.cfg['meta_optim']['name'], self.cfg['meta_optim']['kwargs'], key_meta_optim)
    
    # Set PPO network
    if normalize_observations:
      normalize = running_statistics.normalize
    else:
      normalize = lambda x, y: x
    ppo_network = network_factory(
      obs_shape[-1],
      env.action_size,
      preprocess_observations_fn = normalize,
      policy_hidden_layer_sizes = (32,) * 4,
      value_hidden_layer_sizes = (64,) * 5,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    
    # Set training states
    def get_training_state(key):
      key_policy, key_value = jitted_split(key)
      agent_param = ppo_losses.PPONetworkParams(
        policy = ppo_network.policy_network.init(key_policy),
        value = ppo_network.value_network.init(key_value)
      )
      training_state = TrainingState(
        agent_optim_state = agent_optim.init(agent_param),
        agent_param = agent_param,
        normalizer_param = running_statistics.init_state(specs.Array(obs_shape[-1:], jnp.dtype('float32')))
      )
      return training_state
    key_agents = jitted_split(key_agent_param, self.local_devices_to_use)
    training_states = jax.pmap(
      get_training_state,
      axis_name = self._PMAP_AXIS_NAME
    )(key_agents)
    
    # Set meta param and meta optim state
    meta_param = agent_optim.optim_param
    meta_optim_state = meta_optim.init(meta_param)
    
    # Set loss function
    agent_loss_fn = functools.partial(
      ppo_losses.compute_ppo_loss,
      ppo_network=ppo_network,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling,
      gae_lambda=gae_lambda,
      clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage
    )
    meta_loss_fn = agent_loss_fn

    # Set pmap_axis_name to None so we don't average agent grad over cores
    agent_grad_update_fn = gradients.gradient_update_fn_with_optim_param(agent_loss_fn, agent_optim, pmap_axis_name=None, has_aux=True)
    
    def convert_data(x: jnp.ndarray, key: PRNGKey):
      x = jax.random.permutation(key, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    def minibatch_step(
        carry, data: types.Transition,
        normalizer_param: running_statistics.RunningStatisticsState
      ):
      meta_param, optim_state, agent_param, key = carry
      key, key_loss = jitted_split(key)
      (loss, _), agent_param, optim_state = agent_grad_update_fn(
        agent_param,
        normalizer_param,
        data,
        key_loss,
        optim_param = meta_param,
        optimizer_state = optim_state
      )
      return (meta_param, optim_state, agent_param, key), None

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_param: running_statistics.RunningStatisticsState
      ):
      meta_param, optim_state, agent_param, key = carry
      key, key_perm, key_grad = jitted_split(key, 3)
      shuffled_data = tree_map(functools.partial(convert_data, key=key_perm), data)
      (meta_param, optim_state, agent_param, key_grad), _ = lax.scan(
        f = functools.partial(minibatch_step, normalizer_param=normalizer_param),
        init = (meta_param, optim_state, agent_param, key_grad),
        xs = shuffled_data,
        length = num_minibatches
      )
      return (meta_param, optim_state, agent_param, key), None

    def training_step(
        carry: Tuple[flax.core.FrozenDict, TrainingState, envs.State, PRNGKey],
        unused_t
      ) -> Tuple[Tuple[flax.core.FrozenDict, TrainingState, envs.State, PRNGKey], Any]:
      meta_param, training_state, env_state, key = carry
      key_sgd, key_generate_unroll, new_key = jitted_split(key, 3)
      policy = make_policy((training_state.normalizer_param, training_state.agent_param.policy))
      # Set rollout function
      def rollout(carry, unused_t):
        current_state, current_key = carry
        current_key, next_key = jitted_split(current_key)
        next_state, data = acting.generate_unroll(
          env,
          current_state,
          policy,
          current_key,
          unroll_length,
          extra_fields = ('truncation',)
        )
        return (next_state, next_key), data
      # Rollout for `batch_size * num_minibatches * unroll_length` steps
      (env_state, _), data = lax.scan(
        f = rollout,
        init = (env_state, key_generate_unroll),
        length = batch_size * num_minibatches // num_envs,
        xs = None
      )
      # shape = (batch_size * num_minibatches // num_envs, unroll_length, num_envs, ...)
      data = tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
      # shape = (batch_size * num_minibatches // num_envs, num_envs, unroll_length, ...)
      data = tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
      # shape = (batch_size * num_minibatches, unroll_length, ...)
      assert data.discount.shape[1:] == (unroll_length,)
      # Update agent_param normalization
      normalizer_param = running_statistics.update(
        training_state.normalizer_param,
        data.observation,
        pmap_axis_name = self._PMAP_AXIS_NAME
      )
      # SGD steps
      (meta_param, agent_optim_state, agent_param, key_sgd), _ = lax.scan(
        f = functools.partial(sgd_step, data=data, normalizer_param=normalizer_param),
        init = (meta_param, training_state.agent_optim_state, training_state.agent_param, key_sgd),
        length = update_epochs,
        xs = None
      )
      # Set the new training_state
      new_training_state = TrainingState(
        agent_optim_state = agent_optim_state,
        agent_param = agent_param,
        normalizer_param = normalizer_param
      )
      return (meta_param, new_training_state, env_state, new_key), None

    def agent_update_and_meta_loss(
        meta_param: flax.core.FrozenDict,
        training_state: TrainingState,
        env_state: envs.State,
        key: PRNGKey
      ) -> Tuple[jnp.ndarray, Tuple[flax.core.FrozenDict, TrainingState, envs.State, PRNGKey]]:
      """Agent learning: update agent params"""
      (meta_param, training_state, env_state, key), _ = lax.scan(
        f = training_step,
        init = (meta_param, training_state, env_state, key),
        length = inner_updates,
        xs = None
      )
      """Meta learning: update meta params"""
      # Gather data for meta learning
      key_meta, key_generate_unroll, new_key = jitted_split(key, 3)
      policy = make_policy((training_state.normalizer_param, training_state.agent_param.policy))
      # Set rollout function
      def rollout(carry, unused_t):
        current_state, current_key = carry
        current_key, next_key = jitted_split(current_key)
        next_state, data = acting.generate_unroll(
          env,
          current_state,
          policy,
          current_key,
          unroll_length,
          extra_fields = ('truncation',)
        )
        return (next_state, next_key), data
      # Rollout for `batch_size * unroll_length` steps
      (env_state, _), data = lax.scan(
        f = rollout,
        init = (env_state, key_generate_unroll),
        length = max(batch_size // num_envs, 1),
        xs = None
      )
      data = tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
      data = tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
      assert data.discount.shape[1:] == (unroll_length,)
      # Compute meta loss
      meta_loss, _ = meta_loss_fn(
        params = training_state.agent_param,
        normalizer_params = training_state.normalizer_param,
        data = data,
        rng = key_meta
      )
      return meta_loss, (meta_param, training_state, env_state, new_key)

    def meta_training_step(meta_param, meta_optim_state, training_state, env_state, key):
      # Compute meta_grad
      meta_grad, (meta_param, training_state, env_state, key) = jax.grad(agent_update_and_meta_loss, has_aux=True)(
        meta_param, training_state, env_state, key
      )
      meta_grad = lax.pmean(meta_grad, axis_name=self._PMAP_AXIS_NAME)
      # Update meta_param
      meta_param_update, meta_optim_state = meta_optim.update(meta_grad, meta_optim_state)
      meta_param = optax.apply_updates(meta_param, meta_param_update)
      # Update training_state: optim_param
      agent_optim_state = training_state.agent_optim_state
      agent_optim_state = agent_optim_state.replace(optim_param=meta_param)
      training_state = training_state.replace(agent_optim_state=agent_optim_state)
      return meta_param, meta_optim_state, training_state, env_state, key
    
    pmap_meta_training_iteration = jax.pmap(
      meta_training_step,
      in_axes = (None, None, 0, 0, 0),
      out_axes = (None, None, 0, 0, 0),
      devices = jax.local_devices()[:self.local_devices_to_use],
      axis_name = self._PMAP_AXIS_NAME
    )

    # Setup agent training reset
    def reset_agent_training(training_state, env_state, reset_index, key, iter_num):
      # Select the new one if iter_num % agent_reset_interval == reset_index
      f_select = lambda n_s, o_s: lax.select(iter_num % self.agent_reset_interval == reset_index, n_s, o_s)
      # Generate a new training_state and env_state
      key_env, key_agent = jitted_split(key, 2)
      new_training_state = get_training_state(key_agent)
      key_envs = jitted_split(key_env, num_envs // self.process_count // self.local_devices_to_use)
      new_env_state = jax.jit(env.reset)(key_envs)
      # Select the new training_state 
      training_state = tree_map(f_select, new_training_state, training_state)
      env_state = tree_map(f_select, new_env_state, env_state)
      return training_state, env_state
    
    pmap_reset_training_state = jax.pmap(
      reset_agent_training,
      in_axes = (0, 0, 0, 0, None),
      out_axes = (0, 0),
      axis_name = self._PMAP_AXIS_NAME
    )

    # Start training
    step_key, local_key = jitted_split(local_key)
    step_keys = jitted_split(step_key, self.local_devices_to_use)
    step_keys = self.core_reshape(jnp.stack(step_keys))
    for t in range(1, self.iterations+1):
      self.start_time = time.time()
      # Reset agent training: agent_param, hidden_state, env_state
      key_reset, *reset_keys = jitted_split(key_reset, self.local_devices_to_use+1)
      reset_keys = self.core_reshape(jnp.stack(reset_keys))
      training_states, env_states = pmap_reset_training_state(training_states, env_states, self.reset_indexes, reset_keys, t-1)
      # Train for one iteration
      meta_param, meta_optim_state, training_states, env_states, step_keys = pmap_meta_training_iteration(
        meta_param, meta_optim_state, training_states, env_states, step_keys
      )
      # Check NaN error
      if jnp.any(jnp.isnan(pytree2array(meta_param))):
        self.logger.info('NaN error detected!')
        break
      # Show log
      if (t % self.cfg['display_interval'] == 0 or t == self.iterations) and self.process_id == 0:
        speed = total_env_step_per_training_step / (time.time() - self.start_time)
        eta = (self.iterations - t) * total_env_step_per_training_step / speed / 60 if speed>0 else -1
        self.logger.info(f'<{self.config_idx}> Iteration {t}/{self.iterations}, Speed={speed:.2f} (steps/s), ETA={eta:.2f} (mins)')
      # Save meta param
      if t == self.iterations:
        save_model_param(meta_param, self.cfg['logs_dir']+'param.pickle')