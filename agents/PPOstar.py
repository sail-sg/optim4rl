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
import functools
import numpy as np
from typing import Tuple

import jax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_map

from brax import envs
from brax.training import acting, types
from brax.training.types import PRNGKey, Metrics
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import specs

from components import running_statistics
from components import star_gradients
from components.optim import set_optim
from utils.helper import jitted_split

from agents.PPO import PPO, TrainingState


class PPOstar(PPO):
  '''
  PPO for Brax for STAR optimizer only.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)

  def train(self):
    # Env
    environment = self.env
    num_timesteps = self.train_steps
    episode_length = self.cfg['env']['episode_length']
    action_repeat = self.cfg['env']['action_repeat']
    reward_scaling = self.cfg['env']['reward_scaling']
    num_envs = self.cfg['env']['num_envs']
    num_evals = self.cfg['env']['num_evals']
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
    # Others
    batch_size = self.cfg['batch_size']
    discounting = self.cfg['discount']
    seed = self.cfg['seed']
    deterministic_eval = False
    progress_fn = self.save_progress
    
    """PPO training."""
    device_count = self.local_devices_to_use * self.process_count
    assert num_envs % device_count == 0
    assert batch_size * num_minibatches % num_envs == 0
    # The number of environment steps executed for every training step.
    env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat
    num_evals = max(num_evals, 1)
    # The number of training_step calls per training_epoch call.
    num_training_steps_per_epoch = num_timesteps // (num_evals * env_step_per_training_step)
    self.macro_step = num_training_steps_per_epoch * env_step_per_training_step

    # Prepare keys
    # key_networks should be global so that
    # the initialized networks are the same for different processes.
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jitted_split(key)
    local_key = jax.random.fold_in(local_key, self.process_id)
    local_key, key_env, eval_key = jitted_split(local_key, 3)
    key_policy, key_value, key_optim = jitted_split(global_key, 3)
    del key, global_key
    key_envs = jitted_split(key_env, num_envs // self.process_count)
    # Reshape to (local_devices_to_use, num_envs//process_count, 2)
    key_envs = jnp.reshape(key_envs, (self.local_devices_to_use, -1) + key_envs.shape[1:])

    # Set training and evaluation env
    env = envs.training.wrap(
      environment,
      episode_length = episode_length,
      action_repeat = action_repeat,
      randomization_fn = None
    )
    eval_env = envs.training.wrap(
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
    obs_shape = env_states.obs.shape # (local_devices_to_use, num_envs//process_count, ...)
    
    # Set optimizer
    optimizer = set_optim(self.cfg['optim']['name'], self.cfg['optim']['kwargs'], key_optim)

    # Set PPO network
    if normalize_observations:
      normalize = running_statistics.normalize
    else:
      normalize = lambda x, y: x
    ppo_network = network_factory(
      obs_shape[-1],
      env.action_size,
      preprocess_observations_fn = normalize,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    agent_param = ppo_losses.PPONetworkParams(
      policy = ppo_network.policy_network.init(key_policy),
      value = ppo_network.value_network.init(key_value)
    )
    training_state = TrainingState(
      agent_optim_state = optimizer.init(agent_param),
      agent_param = agent_param,
      normalizer_param = running_statistics.init_state(specs.Array(obs_shape[-1:], jnp.dtype('float32'))),
      env_step = 0
    )
    
    # Set loss function
    loss_fn = functools.partial(
      ppo_losses.compute_ppo_loss,
      ppo_network=ppo_network,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling,
      gae_lambda=gae_lambda,
      clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage
    )
    gradient_update_fn = star_gradients.gradient_update_fn(loss_fn, optimizer, pmap_axis_name=self._PMAP_AXIS_NAME, has_aux=True)

    def convert_data(x: jnp.ndarray, key: PRNGKey):
      x = jax.random.permutation(key, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    def minibatch_step(
        carry, data: types.Transition,
        normalizer_param: running_statistics.RunningStatisticsState
      ):
      agent_optim_state, agent_param, key = carry
      key, key_loss = jitted_split(key)
      (loss, _), agent_param, agent_optim_state = gradient_update_fn(
        agent_param,
        normalizer_param,
        data,
        key_loss,
        optimizer_state = agent_optim_state
      )
      return (agent_optim_state, agent_param, key), None

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_param: running_statistics.RunningStatisticsState
      ):
      agent_optim_state, agent_param, key = carry
      key, key_perm, key_grad = jitted_split(key, 3)
      shuffled_data = tree_map(functools.partial(convert_data, key=key_perm), data)
      (agent_optim_state, agent_param, key_grad), _ = lax.scan(
        f = functools.partial(minibatch_step, normalizer_param=normalizer_param),
        init = (agent_optim_state, agent_param, key_grad),
        xs = shuffled_data,
        length = num_minibatches
      )
      return (agent_optim_state, agent_param, key), None

    def training_step(
        carry: Tuple[TrainingState, envs.State, PRNGKey],
        unused_t
      ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
      training_state, state, key = carry
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
      (state, _), data = lax.scan(
        f = rollout,
        init = (state, key_generate_unroll),
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
      (agent_optim_state, agent_param, key_sgd), _ = lax.scan(
        f = functools.partial(sgd_step, data=data, normalizer_param=normalizer_param),
        init = (training_state.agent_optim_state, training_state.agent_param, key_sgd),
        length = update_epochs,
        xs = None
      )
      # Set the new training_state
      new_training_state = TrainingState(
        agent_optim_state = agent_optim_state,
        agent_param = agent_param,
        normalizer_param = normalizer_param,
        env_step = training_state.env_step + env_step_per_training_step
      )
      return (new_training_state, state, new_key), None

    def training_epoch(
        training_state: TrainingState,
        state: envs.State,
        key: PRNGKey
      ) -> Tuple[TrainingState, envs.State, Metrics]:
      (training_state, state, key), _ = lax.scan(
        f = training_step,
        init = (training_state, state, key),
        length = num_training_steps_per_epoch,
        xs = None
      )
      return training_state, state
    
    pmap_training_epoch = jax.pmap(
      training_epoch,
      in_axes = (None, 0, 0),
      out_axes = (None, 0),
      devices = jax.local_devices()[:self.local_devices_to_use],
      axis_name = self._PMAP_AXIS_NAME
    )

    # Set evaluator
    evaluator = acting.Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs = 128,
      episode_length = episode_length,
      action_repeat = action_repeat,
      key = eval_key
    )

    # Run an initial evaluation
    i, current_step = 0, 0
    if self.process_id == 0 and num_evals > 1:
      metrics = evaluator.run_evaluation(
        (training_state.normalizer_param, training_state.agent_param.policy),
        training_metrics={}
      )
      episode_return, _, _ = progress_fn(0, metrics)
      self.logger.info(f'<{self.config_idx}> Iteration {i}/{num_evals}, Step {current_step}, Return={episode_return:.2f}')
    
    # Start training
    for i in range(1, num_evals+1):
      self.start_time = time.time()
      epoch_key, local_key = jitted_split(local_key)
      epoch_keys = jitted_split(epoch_key, self.local_devices_to_use)
      # Train for one epoch
      training_state, env_states = pmap_training_epoch(training_state, env_states, epoch_keys)
      current_step = int(training_state.env_step)
      # Run evaluation
      if self.process_id == 0:
        metrics = evaluator.run_evaluation(
          (training_state.normalizer_param, training_state.agent_param.policy),
          training_metrics={}
        )
        episode_return, speed, eta = progress_fn(current_step, metrics)
        self.logger.info(f'<{self.config_idx}> Iteration {i}/{num_evals}, Step {current_step}, Return={episode_return:.2f}, Speed={speed:.2f} (steps/s), ETA={eta:.2f} (mins)')
        if np.isnan(episode_return):
          self.logger.info('NaN error detected!')
          break