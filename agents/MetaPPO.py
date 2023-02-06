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

# Copyright 2022 The Brax Authors.
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

import functools
import pickle
import time
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from brax import envs
from brax import jumpy as jp
from brax.envs import wrappers
from brax.training import acting, types
from brax.training.acme import specs
from jax import lax
from jax.tree_util import tree_map

import components.losses as ppo_losses
from components import gradients, ppo_networks, running_statistics
from components.optim import OptimState, set_meta_optimizer, set_optimizer
from utils.logger import Logger

InferenceParams = Tuple[running_statistics.NestedMeanStd, types.Params]


@chex.dataclass
class TrainingState:
    """Contains training_state for the learner."""

    agent_optim_state: OptimState
    agent_param: ppo_losses.PPONetworkParams


class MetaPPO(object):
    """
    Meta-train a learned optimizer during traing PPO in Brax, compatible with LinearOptim, Optim4RL, and L2LGD2.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.config_idx = cfg["config_idx"]
        self.logger = Logger(cfg["logs_dir"])
        self.log_path = cfg["logs_dir"] + "result_Test.feather"
        self.result = []
        # Set environment
        self.env_name = cfg["env"]["name"]
        self.agent_name = cfg["agent"]["name"]
        self.train_steps = int(cfg["env"]["train_steps"])
        self.env = envs.get_environment(env_name=self.env_name)
        self.env_state = self.env.reset(rng=jp.random_prngkey(seed=self.cfg["seed"]))
        self._PMAP_AXIS_NAME = "i"
        # Agent reset interval
        self.agent_reset_interval = self.cfg["agent"]["reset_interval"]

    def save_model_param(self, model_param, filepath):
        f = open(filepath, "wb")
        pickle.dump(model_param, f)
        f.close()

    def train(self):
        # Env
        env = self.env
        num_timesteps = self.train_steps
        episode_length = self.cfg["env"]["episode_length"]
        action_repeat = self.cfg["env"]["action_repeat"]
        reward_scaling = self.cfg["env"]["reward_scaling"]
        num_envs = self.cfg["env"]["num_envs"]
        normalize_observations = self.cfg["env"]["normalize_obs"]
        # Agent
        network_factory = ppo_networks.make_ppo_networks
        gae_lambda = self.cfg["agent"]["gae_lambda"]
        unroll_length = self.cfg["agent"]["rollout_steps"]
        num_minibatches = self.cfg["agent"]["num_minibatches"]
        clip_ratio = self.cfg["agent"]["clip_ratio"]
        update_epochs = self.cfg["agent"]["update_epochs"]
        entropy_cost = self.cfg["agent"]["entropy_weight"]
        normalize_advantage = True
        # Meta learning
        inner_updates = self.cfg["agent"]["inner_updates"]
        # Others
        batch_size = self.cfg["batch_size"]
        discounting = self.cfg["discount"]
        max_devices_per_host = self.cfg["max_devices_per_host"]
        seed = self.cfg["seed"]

        """PPO training."""
        process_id = jax.process_index()
        process_count = jax.process_count()
        total_device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        if max_devices_per_host is not None and max_devices_per_host > 0:
            local_devices_to_use = min(local_device_count, max_devices_per_host)
        else:
            local_devices_to_use = local_device_count
        self.logger.info(
            f"Total device: {total_device_count}, Process: {process_count} (ID {process_id})"
        )
        self.logger.info(
            f"Local device: {local_device_count}, Devices to be used: {local_devices_to_use}"
        )
        device_count = local_devices_to_use * process_count
        assert num_envs % device_count == 0
        assert batch_size * num_minibatches % num_envs == 0
        self.core_reshape = lambda x: x.reshape((local_devices_to_use,) + x.shape[1:])

        # The number of environment steps executed for every training step.
        env_step_per_training_step = (
            batch_size * unroll_length * num_minibatches * action_repeat
        )
        meta_env_step_per_training_step = (
            max(batch_size // num_envs, 1)
            * num_envs
            * unroll_length
            * 1
            * action_repeat
        )
        total_env_step_per_training_step = (
            env_step_per_training_step * inner_updates + meta_env_step_per_training_step
        )
        # The number of training_step calls per training_epoch call.
        self.iterations = num_timesteps // total_env_step_per_training_step

        self.logger.info(f"env_step_per_training_step = {env_step_per_training_step}")
        self.logger.info(
            f"meta_env_step_per_training_step = {meta_env_step_per_training_step}"
        )
        self.logger.info(
            f"total_env_step_per_training_step = {total_env_step_per_training_step}"
        )
        self.logger.info(f"total iterations = {self.iterations}")

        # Prepare keys
        # key_networks should be global so that
        # the initialized networks are the same for different processes.
        key = jax.random.PRNGKey(seed)
        global_key, local_key = jax.random.split(key)
        local_key = jax.random.fold_in(local_key, process_id)
        local_key, key_env, key_reset = jax.random.split(local_key, 3)
        key_agent_param, key_agent_optim, key_meta_optim = jax.random.split(
            global_key, 3
        )
        del key, global_key
        key_envs = jax.random.split(key_env, num_envs // process_count)
        key_envs = jnp.reshape(
            key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
        )

        # Set training and evaluation env
        env = wrappers.wrap_for_training(
            env, episode_length=episode_length, action_repeat=action_repeat
        )
        reset_fn = jax.jit(jax.vmap(env.reset))
        env_states = reset_fn(key_envs)

        # Set agent and meta optimizer
        agent_optimizer = set_optimizer(
            self.cfg["agent_optimizer"]["name"],
            self.cfg["agent_optimizer"]["kwargs"],
            key_agent_optim,
        )
        meta_optimizer = set_meta_optimizer(
            self.cfg["meta_optimizer"]["name"],
            self.cfg["meta_optimizer"]["kwargs"],
            key_meta_optim,
        )

        # Set PPO network
        if normalize_observations:
            normalize = running_statistics.normalize
        else:
            normalize = lambda x, y: x
        normalizer_param = running_statistics.init_state(
            specs.Array((env.observation_size,), jnp.float32)
        )
        ppo_network = network_factory(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=normalize,
            policy_hidden_layer_sizes=(32,) * 4,
            value_hidden_layer_sizes=(64,) * 5,
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)

        # Set training states
        def get_training_state(key):
            key_policy, key_value = jax.random.split(key)
            agent_param = ppo_losses.PPONetworkParams(
                policy=ppo_network.policy_network.init(key_policy),
                value=ppo_network.value_network.init(key_value),
            )
            training_state = TrainingState(
                agent_param=agent_param,
                agent_optim_state=agent_optimizer.init(agent_param),
            )
            return training_state

        key_agents = jax.random.split(key_agent_param, local_devices_to_use)
        training_states = jax.pmap(get_training_state)(key_agents)

        # Set meta param and meta optim state
        meta_param = agent_optimizer.optim_param
        meta_optim_state = meta_optimizer.init(meta_param)

        # Set loss function
        agent_loss_fn = functools.partial(
            ppo_losses.compute_ppo_loss,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            normalize_advantage=normalize_advantage,
        )

        # Set pmap_axis_name to None so we don't average agent grad over cores
        agent_grad_update_fn = gradients.gradient_update_fn_with_optim_param(
            agent_loss_fn, agent_optimizer, pmap_axis_name=None, has_aux=True
        )

        meta_loss_fn = functools.partial(
            ppo_losses.compute_ppo_loss,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            normalize_advantage=normalize_advantage,
        )

        def convert_data(x: jnp.ndarray, key: types.PRNGKey):
            x = jax.random.permutation(key, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        def minibatch_step(
            carry,
            data: types.Transition,
            normalizer_param: running_statistics.RunningStatisticsState,
        ):
            meta_param, optim_state, agent_param, key = carry
            key, key_loss = jax.random.split(key)
            (loss, _), agent_param, optim_state = agent_grad_update_fn(
                agent_param,
                normalizer_param,
                data,
                key_loss,
                optim_param=meta_param,
                optimizer_state=optim_state,
            )
            return (meta_param, optim_state, agent_param, key), None

        def sgd_step(
            carry,
            unused_t,
            data: types.Transition,
            normalizer_param: running_statistics.RunningStatisticsState,
        ):
            meta_param, optim_state, agent_param, key = carry
            key, key_perm, key_grad = jax.random.split(key, 3)
            shuffled_data = tree_map(
                functools.partial(convert_data, key=key_perm), data
            )
            (meta_param, optim_state, agent_param, key_grad), _ = lax.scan(
                f=functools.partial(minibatch_step, normalizer_param=normalizer_param),
                init=(meta_param, optim_state, agent_param, key_grad),
                xs=shuffled_data,
                length=num_minibatches,
            )
            return (meta_param, optim_state, agent_param, key), None

        def training_step(
            carry: Tuple[
                chex.ArrayTree,
                running_statistics.RunningStatisticsState,
                TrainingState,
                envs.State,
                types.PRNGKey,
            ],
            unused_t,
        ) -> Tuple[
            Tuple[
                chex.ArrayTree,
                running_statistics.RunningStatisticsState,
                TrainingState,
                envs.State,
                types.PRNGKey,
            ],
            types.Metrics,
        ]:
            meta_param, normalizer_param, training_state, env_state, key = carry
            key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)
            policy = make_policy((normalizer_param, training_state.agent_param.policy))

            # Set rollout function
            def rollout(carry, unused_t):
                current_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                next_state, data = acting.generate_unroll(
                    env,
                    current_state,
                    policy,
                    current_key,
                    unroll_length,
                    extra_fields=("truncation",),
                )
                return (next_state, next_key), data

            # Rollout for `batch_size * num_minibatches * unroll_length` steps
            (env_state, _), data = lax.scan(
                f=rollout,
                init=(env_state, key_generate_unroll),
                xs=None,
                length=batch_size * num_minibatches // num_envs,
            )
            # shape = (batch_size * num_minibatches // num_envs, unroll_length, num_envs, ...)
            data = tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
            # shape = (batch_size * num_minibatches // num_envs, num_envs, unroll_length, ...)
            data = tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
            # shape = (batch_size * num_minibatches, unroll_length, ...)
            assert data.discount.shape[1:] == (unroll_length,)
            # Update normalization params and normalize observations.
            normalizer_param = running_statistics.update(
                normalizer_param, data.observation, pmap_axis_name=self._PMAP_AXIS_NAME
            )
            # SGD steps
            (meta_param, agent_optim_state, agent_param, key_sgd), _ = lax.scan(
                f=functools.partial(
                    sgd_step, data=data, normalizer_param=normalizer_param
                ),
                init=(
                    meta_param,
                    training_state.agent_optim_state,
                    training_state.agent_param,
                    key_sgd,
                ),
                xs=None,
                length=update_epochs,
            )
            # Set the new training_state
            training_state = training_state.replace(
                agent_optim_state=agent_optim_state, agent_param=agent_param
            )
            return (
                meta_param,
                normalizer_param,
                training_state,
                env_state,
                new_key,
            ), None

        def agent_update_and_meta_loss(
            meta_param, normalizer_param, training_state, env_state, key
        ):
            """Agent learning: update agent params"""
            (
                meta_param,
                normalizer_param,
                training_state,
                env_state,
                key,
            ), _ = lax.scan(
                f=training_step,
                init=(meta_param, normalizer_param, training_state, env_state, key),
                length=inner_updates,
                xs=None,
            )
            """Meta learning: update meta params"""
            # Gather data for meta learning
            key_meta, key_generate_unroll, new_key = jax.random.split(key, 3)
            policy = make_policy((normalizer_param, training_state.agent_param.policy))

            # Set rollout function
            def rollout(carry, unused_t):
                current_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                next_state, data = acting.generate_unroll(
                    env,
                    current_state,
                    policy,
                    current_key,
                    unroll_length,
                    extra_fields=("truncation",),
                )
                return (next_state, next_key), data

            # Rollout for `batch_size * unroll_length` steps
            (env_state, _), data = lax.scan(
                f=rollout,
                init=(env_state, key_generate_unroll),
                xs=None,
                length=max(batch_size // num_envs, 1),
            )
            data = tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
            data = tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
            assert data.discount.shape[1:] == (unroll_length,)
            # Update normalization params and normalize observations.
            normalizer_param = running_statistics.update(
                normalizer_param, data.observation, pmap_axis_name=self._PMAP_AXIS_NAME
            )
            # Compute meta loss
            meta_loss, _ = meta_loss_fn(
                params=training_state.agent_param,
                normalizer_params=normalizer_param,
                data=data,
                rng=key_meta,
            )
            return meta_loss, (
                meta_param,
                normalizer_param,
                training_state,
                env_state,
                key,
            )

        def meta_training_step(
            meta_param,
            meta_optim_state,
            normalizer_param,
            training_state,
            env_state,
            key,
        ):
            # Compute meta_grad
            meta_grad, (
                meta_param,
                normalizer_param,
                training_state,
                env_state,
                key,
            ) = jax.grad(agent_update_and_meta_loss, has_aux=True)(
                meta_param, normalizer_param, training_state, env_state, key
            )
            meta_grad = lax.pmean(meta_grad, axis_name=self._PMAP_AXIS_NAME)
            # Update meta_param
            meta_param_update, meta_optim_state = meta_optimizer.update(
                meta_grad, meta_optim_state
            )
            meta_param = optax.apply_updates(meta_param, meta_param_update)
            # Update training_state: optim_param
            agent_optim_state = training_state.agent_optim_state
            agent_optim_state = agent_optim_state.replace(optim_param=meta_param)
            training_state = training_state.replace(agent_optim_state=agent_optim_state)
            return (
                meta_param,
                meta_optim_state,
                normalizer_param,
                training_state,
                env_state,
                key,
            )

        pmap_meta_training_iteration = jax.pmap(
            meta_training_step,
            in_axes=(None, None, None, 0, 0, 0),
            out_axes=(None, None, None, 0, 0, 0),
            devices=jax.local_devices()[:local_devices_to_use],
            axis_name=self._PMAP_AXIS_NAME,
        )

        # Setup agent training reset
        reset_indexes = [
            int(x)
            for x in jnp.linspace(
                0, self.agent_reset_interval - 1, num=local_devices_to_use
            )
        ]
        self.reset_indexes = self.core_reshape(jnp.array(reset_indexes))

        def reset_agent_training(training_state, env_state, reset_index, key, iter_num):
            # Select the new one if iter_num % agent_reset_interval == reset_index
            def f_select(n_s, o_s):
                return lax.select(
                    iter_num % self.agent_reset_interval == reset_index, n_s, o_s
                )

            # Generate a new training_state and env_state
            key_env, key_agent = jax.random.split(key, 2)
            new_training_state = get_training_state(key_agent)
            key_envs = jax.random.split(
                key_env, num_envs // process_count // local_devices_to_use
            )
            new_env_state = jax.jit(jax.vmap(env.reset))(key_envs[None, :])
            new_env_state = tree_map(lambda x: x[0], new_env_state)
            # Select the new training_state
            training_state = tree_map(f_select, new_training_state, training_state)
            env_state = tree_map(f_select, new_env_state, env_state)
            return training_state, env_state

        pmap_reset_training_state = jax.pmap(
            reset_agent_training, in_axes=(0, 0, 0, 0, None)
        )

        # Start training
        step_key, local_key = jax.random.split(local_key)
        step_keys = jax.random.split(step_key, local_devices_to_use)
        for t in range(1, self.iterations + 1):
            self.start_time = time.time()
            # Reset agent training: agent_param, hidden_state, env_state
            key_reset, *reset_keys = jax.random.split(
                key_reset, local_devices_to_use + 1
            )
            reset_keys = jnp.stack(reset_keys)
            training_states, env_states = pmap_reset_training_state(
                training_states, env_states, self.reset_indexes, reset_keys, t - 1
            )
            # Train for one iteration
            (
                meta_param,
                meta_optim_state,
                normalizer_param,
                training_state,
                env_states,
                step_keys,
            ) = pmap_meta_training_iteration(
                meta_param,
                meta_optim_state,
                normalizer_param,
                training_states,
                env_states,
                step_keys,
            )
            # Show log
            if (
                t % self.cfg["display_interval"] == 0 or t == self.iterations
            ) and process_id == 0:
                speed = total_env_step_per_training_step / (
                    time.time() - self.start_time
                )
                eta = (
                    (self.iterations - t)
                    * total_env_step_per_training_step
                    / speed
                    / 60
                    if speed > 0
                    else -1
                )
                self.logger.info(
                    f"<{self.config_idx}> Iteration {t}/{self.iterations}, Speed={speed:.2f} (steps/s), ETA={eta:.2f} (mins)"
                )
            # Save meta param
            if (self.cfg["save_param"] > 0 and t % self.cfg["save_param"] == 0) or (
                t == self.iterations
            ):
                self.save_model_param(
                    meta_param, self.cfg["logs_dir"] + f"param{t}.pickle"
                )
