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
from jax import lax, random

from agents.A2C import A2C


class A2C2(A2C):
    """
    Implementation of A2C for gridworlds, only compatible with STAR.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def learn(self, carry_in):
        training_state, env_state, seed = carry_in
        seed, step_seed = random.split(seed)
        # Generate one rollout and compute the gradient
        (agent_loss, (env_state, rollout)), agent_grad = jax.value_and_grad(
            self.compute_agent_loss, has_aux=True
        )(training_state.agent_param, env_state, step_seed)
        # Reduce mean gradients across batch an cores
        agent_grad = lax.pmean(agent_grad, axis_name="batch")
        agent_grad = lax.pmean(agent_grad, axis_name="core")
        # Update model parameters
        agent_optim_state = self.agent_optimizer.update(
            agent_grad, training_state.agent_optim_state, agent_loss
        )
        # Set new training_state
        training_state = training_state.replace(
            agent_param=training_state.agent_optim_state.params,
            agent_optim_state=agent_optim_state,
        )
        carry_out = [training_state, env_state, seed]
        logs = dict(done=rollout.done, reward=rollout.reward)
        return carry_out, logs
