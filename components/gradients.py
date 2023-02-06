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

"""Brax training gradient utility functions."""

import jax
import optax
from typing import Callable, Optional


def loss_and_pgrad(loss_fn: Callable[..., float],
                   pmap_axis_name: Optional[str],
                   has_aux: bool = False):
  g = jax.value_and_grad(loss_fn, has_aux=has_aux)

  def h(*args, **kwargs):
    value, grad = g(*args, **kwargs)
    return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

  return g if pmap_axis_name is None else h


def gradient_update_fn(loss_fn: Callable[..., float],
                       optimizer: optax.GradientTransformation,
                       pmap_axis_name: Optional[str],
                       has_aux: bool = False):
  """Wrapper of the loss function that apply gradient updates.

  Args:
    loss_fn: The loss function.
    optimizer: The optimizer to apply gradients.
    pmap_axis_name: If relevant, the name of the pmap axis to synchronize
      gradients.
    has_aux: Whether the loss_fn has auxiliary data.

  Returns:
    A function that takes the same argument as the loss function plus the
    optimizer state. The output of this function is the loss, the new parameter,
    and the new optimizer state.
  """
  loss_and_pgrad_fn = loss_and_pgrad(
      loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)

  def f(*args, optimizer_state):
    value, grads = loss_and_pgrad_fn(*args)
    params_update, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(args[0], params_update)
    return value, params, optimizer_state, grads, params_update

  return f


def gradient_update_fn_with_optim_param(loss_fn: Callable[..., float],
                       optimizer: optax.GradientTransformation,
                       pmap_axis_name: Optional[str],
                       has_aux: bool = False):
  """Wrapper of the loss function that apply gradient updates.

  Args:
    loss_fn: The loss function.
    optimizer: The optimizer to apply gradients.
    pmap_axis_name: If relevant, the name of the pmap axis to synchronize
      gradients.
    has_aux: Whether the loss_fn has auxiliary data.

  Returns:
    A function that takes the same argument as the loss function plus the
    optimizer state. The output of this function is the loss, the new parameter,
    and the new optimizer state.
  """
  loss_and_pgrad_fn = loss_and_pgrad(
      loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)

  def f(*args, optim_param, optimizer_state):
    value, grads = loss_and_pgrad_fn(*args)
    params_update, optimizer_state = optimizer.update_with_param(optim_param, grads, optimizer_state)
    params = optax.apply_updates(args[0], params_update)
    return value, params, optimizer_state

  return f