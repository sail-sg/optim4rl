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

import flax
import rlax
import optax
import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import jit, lax, random, tree_util

from utils.helper import jitted_split
from agents.BaseAgent import BaseAgent
from components.optim import set_optim
from components.network import ActorVCriticNet


@flax.struct.dataclass
class TimeStep:
  obs: jax.Array
  action: jax.Array
  reward: jax.Array
  done: jax.Array


@flax.struct.dataclass
class MyTrainState:
  agent_param: flax.core.FrozenDict
  agent_optim_state: Any


class A2C(BaseAgent):
  '''
  Implementation of Actor Critic.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Set agent optimizer
    self.seed, optim_seed = jitted_split(self.seed)
    # Set learning_rates for tasks
    agent_optim_cfg = deepcopy(cfg['agent_optim']['kwargs'])
    if ('MetaA2C' in self.agent_name or 'MetapA2C' in self.agent_name) and 'star' not in self.agent_name:
      if isinstance(agent_optim_cfg['learning_rate'], list):
        self.learning_rates = agent_optim_cfg['learning_rate'].copy()
      elif isinstance(agent_optim_cfg['learning_rate'], float):
        self.learning_rates = [agent_optim_cfg['learning_rate']] * self.task_num
      else:
        raise TypeError('Only List[float] or float is allowed')
      agent_optim_cfg['learning_rate'] = 1.0
    self.agent_optim = set_optim(cfg['agent_optim']['name'], agent_optim_cfg, optim_seed)
    # Make some utility tools
    self.reshape = lambda x: x.reshape((self.core_count, self.batch_size) + x.shape[1:])

  def create_agent_nets(self):
    agent_nets = []
    for i, env_name in enumerate(self.env_names):   
      agent_net = ActorVCriticNet(
        action_size=self.action_sizes[i],
        env_name=env_name
      )
      agent_nets.append(agent_net)
    return agent_nets
    
  @partial(jit, static_argnames=['self', 'i'])
  def move_one_step(self, carry_in, step_seed, i):
    agent_param, env_state = carry_in
    step_seed, action_seed = jitted_split(step_seed)
    obs = self.envs[i].render_obs(env_state)
    logits, _ = self.agent_nets[i].apply(agent_param, obs[None,])
    # Select an action
    action = random.categorical(key=action_seed, logits=logits[0])
    # Move one step in env
    env_state, reward, done = self.envs[i].step(step_seed, env_state, action)
    carry_out = (agent_param, env_state)
    return carry_out, TimeStep(obs=obs, action=action, reward=reward, done=done)

  @partial(jit, static_argnames=['self', 'i'])
  def move_rollout_steps(self, agent_param, env_state, step_seed, i):
    carry_in = (agent_param, env_state)
    # Move for rollout_steps
    step_seeds = jitted_split(step_seed, self.rollout_steps)
    carry_out, rollout = lax.scan(f=partial(self.move_one_step, i=i), init=carry_in, xs=step_seeds)
    env_state = carry_out[1]
    return env_state, rollout

  @partial(jit, static_argnames=['self', 'i'])
  def compute_loss(self, agent_param, env_state, step_seed, i):
    # Move for rollout_steps
    env_state, rollout = self.move_rollout_steps(agent_param, env_state, step_seed, i)
    last_obs = self.envs[i].render_obs(env_state)
    all_obs = jnp.concatenate([rollout.obs, jnp.expand_dims(last_obs, 0)], axis=0)
    logits, v = self.agent_nets[i].apply(agent_param, all_obs)
    # Compute multi-step temporal difference error
    td_error = rlax.td_lambda(
      v_tm1 = v[:-1],
      r_t = rollout.reward,
      discount_t = self.discount * (1.0-rollout.done),
      v_t = v[1:],
      lambda_ = self.cfg['agent']['gae_lambda'],
      stop_target_gradients = True
    )
    # Compute critic loss
    critic_loss = self.cfg['agent']['critic_loss_weight'] * jnp.mean(td_error**2)
    # Compute actor loss 
    actor_loss = rlax.policy_gradient_loss(
      logits_t = logits[:-1],
      a_t = rollout.action,
      adv_t = td_error,
      w_t = jnp.ones_like(td_error),
      use_stop_gradient = True
    )
    entropy_loss = self.cfg['agent']['entropy_weight'] * rlax.entropy_loss(
      logits_t = logits[:-1],
      w_t=jnp.ones_like(td_error)
    )
    total_loss = actor_loss + critic_loss + entropy_loss
    return total_loss, (env_state, rollout)

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
    logs = dict(done=rollout.done, reward=rollout.reward)
    return carry_out, logs
    
  @partial(jit, static_argnames=['self', 'i'])
  def train_iterations(self, carry_in, i):
    # Vectorize the learn function across batch
    batched_learn = jax.vmap(
      self.learn,
      in_axes=((None, 0, 0), None),
      out_axes=((None, 0, 0), 0),
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
      self.process_logs(self.env_names[i], logs)

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

  def get_episode_return(self, done_list, reward_list):
    # Input shape: [iterations, core_count, batch_size, rollout_steps*(inner_updates+1)]
    # Reshape to: [batch_size, core_count, iterations*rollout_steps*(inner_updates+1)]
    done_list = done_list.swapaxes(0, 2)
    done_list = done_list.reshape(done_list.shape[:2]+ (-1,))
    reward_list = reward_list.swapaxes(0, 2)
    reward_list = reward_list.reshape(reward_list.shape[:2]+ (-1,))
    # Compute return
    for j in range(1, reward_list.shape[-1]):
      reward_list[:,:,j] = reward_list[:,:,j] + reward_list[:,:,j-1] * (1-done_list[:,:,j-1])
    return_list = reward_list * done_list
    # Shape: [batch_size, core_count, iterations, rollout_steps*(inner_updates+1)]
    return_list = return_list.reshape(return_list.shape[:2]+ (self.iterations, -1))
    done_list = done_list.reshape(done_list.shape[:2]+ (self.iterations, -1))
    # Average over batch, core, and rollout, to shape [iterations]
    return_list = return_list.sum(axis=(0,1,3))
    done_list = done_list.sum(axis=(0,1,3))
    # Get return logs
    step_list, episode_return = [], []
    for i in range(self.iterations):
      if done_list[i] != 0:
        episode_return.append(return_list[i]/done_list[i])
        step_list.append(i)
    return np.array(episode_return), np.array(step_list)

  def save_logs(self, env_name, result):
    result = pd.DataFrame(result)
    result['Env'] = result['Env'].astype('category')
    result['Agent'] = result['Agent'].astype('category')
    result.to_feather(self.log_path(env_name))