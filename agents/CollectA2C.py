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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import lax, random, tree_util

from agents.A2C import A2C, TrainingState


class CollectA2C(A2C):
    """
    Collect agent gradients and parameter updates during training A2C in gridworlds.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def learn(self, carry_in):
        training_state, env_state, seed = carry_in
        seed, step_seed = random.split(seed)
        # Generate one rollout and compute the gradient
        agent_grad, (env_state, rollout) = jax.grad(
            self.compute_agent_loss, has_aux=True
        )(training_state.agent_param, env_state, step_seed)
        # Reduce mean gradients across batch an cores
        agent_grad = lax.pmean(agent_grad, axis_name="batch")
        agent_grad = lax.pmean(agent_grad, axis_name="core")
        # Compute the updates of model parameters
        agent_param_update, agent_optim_state = self.agent_optimizer.update(
            agent_grad, training_state.agent_optim_state
        )
        # Update model parameters
        agent_param = optax.apply_updates(
            training_state.agent_param, agent_param_update
        )
        training_state = training_state.replace(
            agent_param=agent_param, agent_optim_state=agent_optim_state
        )
        carry_out = [training_state, env_state, seed]
        # Pick agent_grad and agent_param_update for a few parameter
        agent_grad = self.pytree2array(agent_grad)
        agent_param_update = (
            self.pytree2array(agent_param_update)
            / self.cfg["agent_optimizer"]["kwargs"]["learning_rate"]
        )
        idxs = jnp.array(range(0, len(agent_grad), self.cfg["agent"]["data_reduce"]))
        agent_grad, agent_param_update = agent_grad[idxs], agent_param_update[idxs]
        return_logs = dict(done=rollout.done, reward=rollout.reward)
        grad_logs = dict(agent_grad=agent_grad, agent_param_update=agent_param_update)
        return carry_out, (return_logs, grad_logs)

    def train_iterations(self, carry_in):
        # Vectorize the learn function across batch
        batched_learn = jax.vmap(
            self.learn,
            in_axes=([None, 0, 0],),
            out_axes=([None, 0, 0], (0, None)),
            axis_name="batch",
        )

        # Repeat the training for many iterations
        def train_one_iteration(carry, _):
            return batched_learn(carry)

        carry_out, logs = lax.scan(
            f=train_one_iteration, init=carry_in, length=self.iterations, xs=None
        )
        return carry_out, logs

    def train(self):
        seed = self.seed
        for i, env_name in enumerate(self.env_names):
            self.logger.info(
                f"<{self.config_idx}> Environment {i+1}/{len(self.env_names)}: {env_name}"
            )
            # Generate random seeds for env and agent
            seed, env_seed, agent_seed = random.split(seed, 3)
            # Set environment and agent network
            self.env, self.agent_net = self.envs[i], self.agent_nets[i]
            # Initialize agent parameter and optimizer state
            dummy_obs = self.env.render_obs(self.env.reset(env_seed))[None, :]
            agent_param = self.agent_net.init(agent_seed, dummy_obs)
            self.logger.info(
                f"# of agent_param for {env_name}: {self.pytree2array(agent_param).size}"
            )
            training_state = TrainingState(
                agent_param=agent_param,
                agent_optim_state=self.agent_optimizer.init(agent_param),
            )
            # Intialize env_states over cores and batch
            seed, *env_seeds = random.split(seed, self.core_count * self.batch_size + 1)
            env_states = jax.vmap(self.env.reset)(jnp.stack(env_seeds))
            env_states = tree_util.tree_map(self.reshape, env_states)
            seed, *step_seeds = random.split(
                seed, self.core_count * self.batch_size + 1
            )
            step_seeds = self.reshape(jnp.stack(step_seeds))
            # Replicate the training process over multiple cores
            pmap_train_iterations = jax.pmap(
                self.train_iterations,
                in_axes=([None, 0, 0],),
                out_axes=([None, 0, 0], (0, None)),
                axis_name="core",
            )
            carry_in = [training_state, env_states, step_seeds]
            carry_out, logs = pmap_train_iterations(carry_in)
            # Process and save logs
            return_logs, grad_logs = logs
            return_logs["agent_grad"] = grad_logs["agent_grad"]
            return_logs["agent_param_update"] = grad_logs["agent_param_update"]
            self.process_logs(env_name, return_logs)

    def process_logs(self, env_name, logs):
        # Move logs to CPU, with shape {[core_count, iterations, batch_size, *]}
        logs = jax.device_get(logs)
        # Reshape to {[iterations, core_count, batch_size, *]}
        for k in logs.keys():
            logs[k] = logs[k].swapaxes(0, 1)
        # Compute episode return
        episode_return, step_list = self.get_episode_return(
            logs["done"], logs["reward"]
        )
        result = {
            "Env": env_name,
            "Agent": self.agent_name,
            "Step": step_list * self.macro_step,
            "Return": episode_return,
        }
        # Save logs
        self.save_logs(env_name, result)
        # Save agent_grad and agent_param_update into a npz file: (num_param, optimization_steps)
        self.logger.info(
            f"# of agent_param collected for {env_name}: {logs['agent_grad'].shape[0]}"
        )
        x = logs["agent_grad"]
        y = logs["agent_param_update"]
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
