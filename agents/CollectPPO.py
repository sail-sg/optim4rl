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
import time
from typing import Tuple

import flax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from brax import envs
from brax import jumpy as jp
from brax.envs import wrappers
from brax.training import acting, types
from brax.training.acme import specs
from jax import lax
from jax.tree_util import tree_leaves, tree_map

import components.losses as ppo_losses
from components import gradients, ppo_networks, running_statistics
from components.optim import set_optimizer
from utils.logger import Logger

InferenceParams = Tuple[running_statistics.NestedMeanStd, types.Params]


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_param: running_statistics.RunningStatisticsState
    env_step: jnp.ndarray


def pytree2array(values):
    leaves = tree_leaves(lax.stop_gradient(values))
    a = jnp.concatenate(leaves, axis=None)
    return a


class CollectPPO(object):
    """
    Collect agent gradients and parameter updates during training PPO in Brax.
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
        self.state = self.env.reset(rng=jp.random_prngkey(seed=self.cfg["seed"]))
        # Timing
        self.start_time = time.time()
        self._PMAP_AXIS_NAME = "i"

    def save_progress(self, step_count, metrics):
        episode_return = float(jax.device_get(metrics["eval/episode_reward"]))
        result_dict = {
            "Env": self.env_name,
            "Agent": self.agent_name,
            "Step": step_count,
            "Return": episode_return,
        }
        self.result.append(result_dict)
        # Save result to files
        result = pd.DataFrame(self.result)
        result["Env"] = result["Env"].astype("category")
        result["Agent"] = result["Agent"].astype("category")
        result.to_feather(self.log_path)
        # Show log
        speed = step_count / (time.time() - self.start_time)
        eta = (self.train_steps - step_count) / speed / 60 if speed > 0 else -1
        return episode_return, speed, eta

    def train(self):
        # Env
        env = self.env
        num_timesteps = self.train_steps
        episode_length = self.cfg["env"]["episode_length"]
        action_repeat = self.cfg["env"]["action_repeat"]
        reward_scaling = self.cfg["env"]["reward_scaling"]
        num_envs = self.cfg["env"]["num_envs"]
        num_evals = self.cfg["env"]["num_evals"]
        num_eval_envs = 128
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
        # Optimization
        batch_size = self.cfg["batch_size"]
        discounting = self.cfg["discount"]
        max_devices_per_host = self.cfg["max_devices_per_host"]
        # Others
        seed = self.cfg["seed"]
        eval_env = None
        deterministic_eval = False
        progress_fn = self.save_progress
        logs = dict(agent_grad=[], agent_param_update=[])

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

        # The number of environment steps executed for every training step.
        env_step_per_training_step = (
            batch_size * unroll_length * num_minibatches * action_repeat
        )
        num_evals = max(num_evals, 1)
        # The number of training_step calls per training_epoch call.
        num_training_steps_per_epoch = num_timesteps // (
            num_evals * env_step_per_training_step
        )
        self.logger.info(
            f"num_minibatches={num_minibatches}, update_epochs={update_epochs}, num_training_steps_per_epoch={num_training_steps_per_epoch}, num_evals={num_evals}"
        )

        # Prepare keys
        # key_networks should be global so that
        # the initialized networks are the same for different processes.
        key = jax.random.PRNGKey(seed)
        global_key, local_key = jax.random.split(key)
        local_key = jax.random.fold_in(local_key, process_id)
        local_key, key_env, eval_key = jax.random.split(local_key, 3)
        key_policy, key_value, key_optim = jax.random.split(global_key, 3)
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
        if eval_env is None:
            eval_env = env
        else:
            eval_env = wrappers.wrap_for_training(
                eval_env, episode_length=episode_length, action_repeat=action_repeat
            )

        # Set optimizer
        optimizer = set_optimizer(
            self.cfg["optimizer"]["name"], self.cfg["optimizer"]["kwargs"], key_optim
        )

        # Set PPO network
        if normalize_observations:
            normalize = running_statistics.normalize
        else:
            normalize = lambda x, y: x
        ppo_network = network_factory(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=normalize,
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)
        init_params = ppo_losses.PPONetworkParams(
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value),
        )
        training_state = TrainingState(
            optimizer_state=optimizer.init(init_params),
            params=init_params,
            normalizer_param=running_statistics.init_state(
                specs.Array((env.observation_size,), jnp.float32)
            ),
            env_step=0,
        )

        # Set loss function
        loss_fn = functools.partial(
            ppo_losses.compute_ppo_loss,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            normalize_advantage=normalize_advantage,
        )
        gradient_update_fn = gradients.gradient_update_fn(
            loss_fn, optimizer, pmap_axis_name=self._PMAP_AXIS_NAME, has_aux=True
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
            optimizer_state, params, key = carry
            key, key_loss = jax.random.split(key)
            (
                (loss, _),
                params,
                optimizer_state,
                grads,
                params_update,
            ) = gradient_update_fn(
                params,
                normalizer_param,
                data,
                key_loss,
                optimizer_state=optimizer_state,
            )
            grads = pytree2array(grads)
            params_update = pytree2array(params_update)
            idxs = jnp.array(range(0, len(grads), self.cfg["agent"]["data_reduce"]))
            grads, params_update = grads[idxs], params_update[idxs]
            return (optimizer_state, params, key), (grads, params_update)

        def sgd_step(
            carry,
            unused_t,
            data: types.Transition,
            normalizer_param: running_statistics.RunningStatisticsState,
        ):
            optimizer_state, params, key = carry
            key, key_perm, key_grad = jax.random.split(key, 3)
            shuffled_data = tree_map(
                functools.partial(convert_data, key=key_perm), data
            )
            (optimizer_state, params, key_grad), (grads, params_update) = lax.scan(
                f=functools.partial(minibatch_step, normalizer_param=normalizer_param),
                init=(optimizer_state, params, key_grad),
                xs=shuffled_data,
                length=num_minibatches,
            )
            return (optimizer_state, params, key), (grads, params_update)

        def training_step(
            carry: Tuple[TrainingState, envs.State, types.PRNGKey], unused_t
        ) -> Tuple[Tuple[TrainingState, envs.State, types.PRNGKey], types.Metrics]:
            training_state, state, key = carry
            key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)
            policy = make_policy(
                (training_state.normalizer_param, training_state.params.policy)
            )

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
            (state, _), data = lax.scan(
                f=rollout,
                init=(state, key_generate_unroll),
                xs=None,
                length=batch_size * num_minibatches // num_envs,
            )
            # shape = (batch_size * num_minibatches // num_envs, unroll_length, num_envs)
            data = tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
            # shape = (batch_size * num_minibatches // num_envs, num_envs, unroll_length)
            data = tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
            # shape = (batch_size * num_minibatches, unroll_length)
            assert data.discount.shape[1:] == (unroll_length,)
            # Update normalization params and normalize observations.
            normalizer_param = running_statistics.update(
                training_state.normalizer_param,
                data.observation,
                pmap_axis_name=self._PMAP_AXIS_NAME,
            )
            # SGD steps
            (optimizer_state, params, key_sgd), (grads, params_update) = lax.scan(
                f=functools.partial(
                    sgd_step, data=data, normalizer_param=normalizer_param
                ),
                init=(training_state.optimizer_state, training_state.params, key_sgd),
                xs=None,
                length=update_epochs,
            )
            # Set the new training state
            new_training_state = TrainingState(
                optimizer_state=optimizer_state,
                params=params,
                normalizer_param=normalizer_param,
                env_step=training_state.env_step + env_step_per_training_step,
            )
            return (new_training_state, state, new_key), (grads, params_update)

        def training_epoch(
            training_state: TrainingState, state: envs.State, key: types.PRNGKey
        ) -> Tuple[TrainingState, envs.State, types.Metrics]:
            (training_state, state, _), (grads, params_update) = lax.scan(
                f=training_step,
                init=(training_state, state, key),
                xs=None,
                length=num_training_steps_per_epoch,
            )
            return training_state, state, (grads, params_update)

        pmap_training_epoch = jax.pmap(
            training_epoch,
            in_axes=(None, 0, 0),
            out_axes=(None, 0, None),
            devices=jax.local_devices()[:local_devices_to_use],
            axis_name=self._PMAP_AXIS_NAME,
        )

        # Set evaluator
        evaluator = acting.Evaluator(
            eval_env,
            functools.partial(make_policy, deterministic=deterministic_eval),
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=eval_key,
        )

        # Run an initial evaluation
        i, current_step = 0, 0
        if process_id == 0 and num_evals > 1:
            metrics = evaluator.run_evaluation(
                (training_state.normalizer_param, training_state.params.policy),
                training_metrics={},
            )
            episode_return, speed, eta = progress_fn(0, metrics)
            self.logger.info(
                f"<{self.config_idx}> Iteration {i}/{num_evals}, Step {current_step}, Return={episode_return:.2f}, Speed={speed:.2f} (steps/s), ETA={eta:.2f} (mins)"
            )

        # Start training
        for i in range(1, num_evals + 1):
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            # Train for one epoch
            training_state, env_states, (grads, params_update) = pmap_training_epoch(
                training_state, env_states, epoch_keys
            )
            current_step = int(training_state.env_step)
            # Run evaluation
            if process_id == 0:
                metrics = evaluator.run_evaluation(
                    (training_state.normalizer_param, training_state.params.policy),
                    training_metrics={},
                )
                episode_return, speed, eta = progress_fn(current_step, metrics)
                self.logger.info(
                    f"<{self.config_idx}> Iteration {i}/{num_evals}, Step {current_step}, Return={episode_return:.2f}, Speed={speed:.2f} (steps/s), ETA={eta:.2f} (mins)"
                )
                # Save grads, params_update
                logs["agent_grad"].append(grads)
                logs["agent_param_update"].append(params_update)
        self.process_logs(logs)

    def process_logs(self, logs):
        # Stack to shape {[iterations, num_training_steps_per_epoch, update_epochs, num_minibatches, *]}
        logs = {k: jnp.stack(v, axis=0) for k, v in logs.items()}
        # Reshape to {[iterations * num_training_steps_per_epoch * update_epochs * num_minibatches, *]}
        logs = tree_map(lambda x: jnp.reshape(x, (-1, x.shape[-1])), logs)
        # Move logs to CPU, with shape {[core_count, iterations, batch_size, *]}
        logs = jax.device_get(logs)
        # Shape to: (num_param, optimization_steps)
        # Save agent_grad and agent_param_update into a npz file
        x = logs["agent_grad"].swapaxes(0, 1)
        y = (
            logs["agent_param_update"].swapaxes(0, 1)
            / self.cfg["optimizer"]["kwargs"]["learning_rate"]
        )
        self.logger.info(f"x: {x.shape}, y: {y.shape}")
        np.savez(self.cfg["logs_dir"] + "data.npz", x=x, y=y)

        # Plot
        grad = x.reshape(-1)
        abs_update = np.abs(y).reshape(-1)
        # Plot log(|g|)
        abs_grad = np.abs(grad)
        self.logger.info(
            f"|g|: min = {abs_grad.min():.4f}, max = {abs_grad.max():.4f}, mean = {abs_grad.mean():.4f}"
        )
        log_abs_grad = np.log10(abs_grad + 1e-16)
        self.logger.info(
            f"log(|g|+1e-16): min = {log_abs_grad.min():.4f}, max = {log_abs_grad.max():.4f}, mean = {log_abs_grad.mean():.4f}"
        )
        num, bins, patches = plt.hist(log_abs_grad, bins=20)
        plt.xlabel(r"$\log(|g|+10^{-16})$")
        plt.ylabel("Counts in the bin")
        plt.grid(True)
        plt.savefig(self.cfg["logs_dir"] + "grad.png")
        plt.clf()
        plt.cla()
        plt.close()
        # Plot log(|update|)
        self.logger.info(
            f"|update|: min = {abs_update.min():.4f}, max = {abs_update.max():.4f}, mean = {abs_update.mean():.4f}"
        )
        log_abs_update = np.log10(abs_update + 1e-16)
        self.logger.info(
            f"log(|update|): min = {log_abs_update.min():.4f}, max = {log_abs_update.max():.4f}, mean = {log_abs_update.mean():.4f}"
        )
        num, bins, patches = plt.hist(log_abs_update, bins=20)
        plt.xlabel(r"$\log(|\Delta \theta|+10^{-16})$")
        plt.ylabel("Counts in the bin")
        plt.grid(True)
        plt.savefig(self.cfg["logs_dir"] + "update.png")
        plt.clf()
        plt.cla()
        plt.close()
