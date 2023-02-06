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

import functools
import time
from typing import Any

import chex
import jax
import jax.numpy as jnp
import optax
from jax import lax, random, tree_util

from agents.MetaA2C import MetaA2C
from utils.helper import tree_transpose


@chex.dataclass
class TrainingState:
    agent_param: Any
    agent_optim_param: Any
    agent_optim_state: optax.OptState


class StarA2C(MetaA2C):
    """
    Meta-train STAR during traing A2C in gridworlds.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        assert (
            len(self.env_names) == 1
        ), "Only single task training is supported in StarA2C."

    def agent_update(self, carry_in, _):
        """Perform a step of inner update to the agent."""
        meta_param, training_state, env_state, seed, lr = carry_in
        seed, step_seed = random.split(seed)
        # Generate one rollout and compute agent gradient
        (agent_loss, (env_state, rollout)), agent_grad = jax.value_and_grad(
            self.compute_agent_loss, has_aux=True
        )(training_state.agent_param, env_state, step_seed)
        # Update agent parameters
        agent_optim_state = self.agent_optimizer.update_with_param(
            meta_param, agent_grad, training_state.agent_optim_state, agent_loss
        )
        # Set new training_state
        training_state = training_state.replace(
            agent_param=training_state.agent_optim_state.params,
            agent_optim_state=agent_optim_state,
        )
        carry_out = [meta_param, training_state, env_state, seed, lr]
        return carry_out, None

    def learn(self, carry_in):
        """Two level updates for meta_param (outer update) and agent_param (inner update)."""
        training_state, env_state, seed, lr = carry_in
        # Perform inner updates and compute meta gradient.
        seed, step_seed = random.split(seed)
        carry_in = [training_state, env_state, step_seed, lr]
        meta_param = training_state.agent_optim_param
        meta_grad, carry_out = jax.grad(self.agent_update_and_meta_loss, has_aux=True)(
            meta_param, carry_in
        )
        training_state, env_state = carry_out
        # Reduce mean gradient across batch an cores
        meta_grad = lax.pmean(meta_grad, axis_name="batch")
        meta_grad = lax.pmean(meta_grad, axis_name="core")
        carry_out = [meta_grad, training_state, env_state, seed, lr]
        return carry_out

    def get_training_state(self, seed, obs):
        agent_param = self.agent_net.init(seed, obs)
        training_state = TrainingState(
            agent_param=agent_param,
            agent_optim_param=self.agent_optimizer.get_optim_param(),
            agent_optim_state=self.agent_optimizer.init(agent_param),
        )
        return training_state

    def reset_agent_training(
        self,
        training_state,
        env_state,
        reset_index,
        seed,
        optim_param,
        iter_num,
        agent_reset_interval,
        obs,
    ):
        # Select the new one if iter_num % agent_reset_interval == reset_index
        def f_select(n_s, o_s):
            return lax.select(iter_num % agent_reset_interval == reset_index, n_s, o_s)

        # Generate a new training_state and env_state
        new_training_state = self.get_training_state(seed, obs)
        new_env_state = self.env.reset(seed)
        # Select the new training_state
        training_state = tree_util.tree_map(
            f_select, new_training_state, training_state
        )
        env_state = tree_util.tree_map(f_select, new_env_state, env_state)
        # Update optim_param
        training_state = training_state.replace(agent_optim_param=optim_param)
        return training_state, env_state

    def train(self):
        seed = self.seed
        # Initialize pmap_train_one_iteration and carries: hidden_state, agent_param, agent_optim_state, env_states, step_seeds
        carries = dict()
        pmap_train_one_iterations = dict()
        pvmap_reset_agent_training = dict()
        for i, env_name in enumerate(self.env_names):
            # Generate random seeds for env and agent
            seed, env_seed, agent_seed = random.split(seed, num=3)
            # Set environment and agent network
            self.env, self.agent_net = self.envs[i], self.agent_nets[i]
            # Initialize agent parameter and optimizer
            dummy_obs = self.env.render_obs(self.env.reset(env_seed))[None, :]
            pvmap_reset_agent_training[env_name] = jax.pmap(
                jax.vmap(
                    functools.partial(self.reset_agent_training, obs=dummy_obs),
                    in_axes=(0, 0, 0, 0, None, None, None),
                ),
                in_axes=(0, 0, 0, 0, None, None, None),
            )
            # We initialize core_count*batch_size different agent parameters and optimizer states.
            pvmap_get_training_state = jax.pmap(
                jax.vmap(self.get_training_state, in_axes=(0, None)), in_axes=(0, None)
            )
            agent_seed, *agent_seeds = random.split(
                agent_seed, self.core_count * self.batch_size + 1
            )
            agent_seeds = self.reshape(jnp.stack(agent_seeds))
            training_states = pvmap_get_training_state(agent_seeds, dummy_obs)
            # Intialize env_states over cores and batch
            seed, *env_seeds = random.split(seed, self.core_count * self.batch_size + 1)
            env_states = jax.vmap(self.env.reset)(jnp.stack(env_seeds))
            env_states = tree_util.tree_map(self.reshape, env_states)
            seed, *step_seeds = random.split(
                seed, self.core_count * self.batch_size + 1
            )
            step_seeds = self.reshape(jnp.stack(step_seeds))
            # Save in carries dict
            carries[env_name] = [training_states, env_states, step_seeds, -1]
            # Replicate the training process over multiple cores
            batched_learn = jax.vmap(
                self.learn,
                in_axes=([0, 0, 0, None],),
                out_axes=[None, 0, 0, 0, None],
                axis_name="batch",
            )
            pmap_train_one_iterations[env_name] = jax.pmap(
                batched_learn,
                in_axes=([0, 0, 0, None],),
                out_axes=[None, 0, 0, 0, None],
                axis_name="core",
            )

        self.meta_param = self.agent_optimizer.get_optim_param()
        self.meta_optim_state = self.meta_optimizer.init(self.meta_param)
        # Train for self.iterations for each env
        for t in range(1, self.iterations + 1):
            meta_grads = []
            start_time = time.time()
            for i, env_name in enumerate(self.env_names):
                # Set environment and agent network
                self.env, self.agent_net = self.envs[i], self.agent_nets[i]
                # Reset agent training: agent_param, hidden_state, env_state
                # and update meta parameter (i.e. optim_param)
                training_states, env_states = carries[env_name][0], carries[env_name][1]
                seed, *reset_seeds = random.split(
                    seed, self.core_count * self.batch_size + 1
                )
                reset_seeds = self.reshape(jnp.stack(reset_seeds))
                training_states, env_states = pvmap_reset_agent_training[env_name](
                    training_states,
                    env_states,
                    self.reset_indexes[env_name],
                    reset_seeds,
                    self.meta_param,
                    t,
                    self.reset_intervals[i],
                )
                carries[env_name][0], carries[env_name][1] = training_states, env_states
                # Train for one iteration
                carry_in = carries[env_name]
                carry_out = pmap_train_one_iterations[env_name](carry_in)
                # Update carries
                carries[env_name] = carry_out[1:]
                # Gather meta grad and process
                meta_grad = carry_out[0]
                if self.max_norm > 0:
                    g_norm = self.global_norm(meta_grad)
                    meta_grad = tree_util.tree_map(
                        lambda x: (x / g_norm.astype(x.dtype)) * self.max_norm,
                        meta_grad,
                    )
                meta_grads.append(meta_grad)
            # Update meta paramter
            meta_grad = tree_transpose(meta_grads)
            meta_grad = tree_util.tree_map(lambda x: jnp.mean(x, axis=0), meta_grad)
            # Update meta parameter
            meta_param_update, self.meta_optim_state = self.meta_optimizer.update(
                meta_grad, self.meta_optim_state
            )
            self.meta_param = optax.apply_updates(self.meta_param, meta_param_update)
            # Reset agent_optimizer with new meta_param
            self.agent_optimizer.reset_optimizer(self.meta_param)
            # Show log
            if t % self.cfg["display_interval"] == 0:
                step_count = t * self.macro_step
                speed = self.macro_step / (time.time() - start_time)
                eta = (self.train_steps - step_count) / speed / 60 if speed > 0 else -1
                self.logger.info(
                    f"<{self.config_idx}> Step {step_count}/{self.train_steps} Iteration {t}/{self.iterations}: Speed={speed:.2f} (steps/s), ETA={eta:.2f} (mins)"
                )
            # Save meta param
            if (self.cfg["save_param"] > 0 and t % self.cfg["save_param"] == 0) or (
                t == self.iterations
            ):
                self.save_model_param(
                    self.meta_param, self.cfg["logs_dir"] + f"param{t}.pickle"
                )
