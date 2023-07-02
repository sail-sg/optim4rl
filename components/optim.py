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
import chex
import optax
import pickle
import flax.linen as nn
from typing import Sequence
from jax import numpy as jnp
from jax import lax, random, tree_util
from learned_optimization.learned_optimizers.adafac_nominal import MLPNomLOpt


@chex.dataclass
class OptimState:
  """Contains training state for the learner."""
  hidden_state: chex.ArrayTree
  optim_param: chex.ArrayTree


def set_optimizer(optimizer_name, optimizer_kwargs, key):
  if optimizer_name in ['LinearOptim', 'Optim4RL', 'L2LGD2']:
    optimizer = OptimizerWrapper(optimizer_name, optimizer_kwargs, key)
  elif optimizer_name == 'Star':
    optimizer = StarWrapper(optimizer_name, optimizer_kwargs, key)
  else:
    gradient_clip = optimizer_kwargs['gradient_clip']
    del optimizer_kwargs['gradient_clip']
    if gradient_clip > 0:
      optimizer = optax.chain(
        optax.clip(gradient_clip),
        getattr(optax, optimizer_name.lower())(**optimizer_kwargs)
      )
    else:
      optimizer = getattr(optax, optimizer_name.lower())(**optimizer_kwargs)
  return optimizer


def set_meta_optimizer(optimizer_name, optimizer_kwargs, key):
  if 'gradient_clip' in optimizer_kwargs.keys():
    gradient_clip = optimizer_kwargs['gradient_clip']
    del optimizer_kwargs['gradient_clip']
  else:
    gradient_clip = -1
  
  if gradient_clip > 0:
    optimizer = optax.chain(
      optax.clip(gradient_clip),
      getattr(optax, optimizer_name.lower())(**optimizer_kwargs)
    )
  else:
    optimizer = getattr(optax, optimizer_name.lower())(**optimizer_kwargs)
  return optimizer


class Optim4RL(nn.Module):
  name: str = 'Optim4RL'
  mlp_dims: Sequence[int] = (16, 16)
  hidden_size: int = 8
  learning_rate: float = 1.0
  gradient_clip: float = -1.0
  eps: float = 1e-18
  eps_root: float = 1e-18
  bias: float = 1.0

  def setup(self):
    # Set RNNs
    self.rnn1 = nn.GRUCell()
    self.rnn2 = nn.GRUCell()
    # Set MLPs
    layer_dims1, layer_dims2 = list(self.mlp_dims), list(self.mlp_dims)
    layer_dims1.append(2)
    layer_dims2.append(1)
    layers1, layers2 = [], []
    for i in range(len(layer_dims1)):      
      layers1.append(nn.Dense(layer_dims1[i]))
      layers1.append(nn.relu)
      layers2.append(nn.Dense(layer_dims2[i]))
      layers2.append(nn.relu)
    layers1.pop()
    layers2.pop()
    self.mlp1 = nn.Sequential(layers1)
    self.mlp2 = nn.Sequential(layers2)

  def __call__(self, h, g):
    # Clip the gradient to prevent large update
    g = lax.select(self.gradient_clip > 0, jnp.clip(g, -self.gradient_clip, self.gradient_clip), g)
    g_sign = jnp.sign(g)
    g_log = jnp.log(jnp.abs(g) + self.eps)
    # Expand parameter dimension so that the network is "coodinatewise"
    g_sign = lax.stop_gradient(g_sign[..., None])
    g_log = lax.stop_gradient(g_log[..., None])
    g_input = jnp.concatenate([g_sign, g_log], axis=-1)
    # RNN
    h1, h2 = h
    # Compute m: 1st pseudo moment estimate
    h1, x1 = self.rnn1(h1, g_input)
    o1 = self.mlp1(x1)
    # Add a small bias so that m_sign=1 initially
    m_sign_raw = jnp.tanh(o1[..., 0]+self.bias)
    m_sign = lax.stop_gradient(2.0*(m_sign_raw >= 0.0) - 1.0 - m_sign_raw) + m_sign_raw
    m = g_sign[..., 0] * m_sign * jnp.exp(o1[..., 1])
    # Compute v: 2nd pseudo moment estimate
    h2, x2 = self.rnn2(h2, 2.0*g_log)
    o2 = self.mlp2(x2)
    sqrt_v = jnp.sqrt(jnp.exp(o2[..., 0]) + self.eps_root)
    # Compute the parameter update
    out = -self.learning_rate * m / sqrt_v
    return (h1, h2), out

  def init_hidden_state(self, params):
    # Use fixed random key since default state init fn is just zeros.
    h = (
      nn.GRUCell.initialize_carry(random.PRNGKey(0), params.shape, self.hidden_size),
      nn.GRUCell.initialize_carry(random.PRNGKey(0), params.shape, self.hidden_size)
    )
    return h


class LinearOptim(nn.Module):
  name: str = 'LinearOptim'
  mlp_dims: Sequence[int] = (16, 16)
  hidden_size: int = 8
  learning_rate: float = 1.0
  gradient_clip: float = -1.0
  eps: float = 1e-18
  
  def setup(self):
    # Set RNN
    self.rnn = nn.GRUCell()
    # Set MLP
    layer_dims = list(self.mlp_dims)
    layer_dims.append(3)
    layers = []
    for i in range(len(layer_dims)):
      layers.append(nn.Dense(layer_dims[i]))
      layers.append(nn.relu)
    layers.pop()
    self.mlp = nn.Sequential(layers)

  def __call__(self, h, g):
    # Clip the gradient to prevent large update
    g = lax.select(self.gradient_clip > 0, jnp.clip(g, -self.gradient_clip, self.gradient_clip), g)
    g = lax.stop_gradient(g)
    g_sign = jnp.sign(g)
    g_log = jnp.log(jnp.abs(g) + self.eps)
    # Expand parameter dimension so that the network is "coodinatewise"
    g_sign = lax.stop_gradient(g_sign[..., None])
    g_log = lax.stop_gradient(g_log[..., None])
    g_input = jnp.concatenate([g_sign, g_log], axis=-1)
    h, x = self.rnn(h, g_input)
    outs = self.mlp(x)
    # Slice outs into several elements
    o1, o2, o3 = outs[..., 0], outs[..., 1], outs[..., 2]
    # Compute the output: Delta theta
    out = -self.learning_rate * (jnp.exp(o1) * g + jnp.exp(o2) * o3)
    return h, out

  def init_hidden_state(self, params):
    # Use fixed random key since default state init fn is just zeros.
    h = nn.GRUCell.initialize_carry(random.PRNGKey(0), params.shape, self.hidden_size)
    return h


class L2LGD2(nn.Module):
  """
  Implementaion of [Learning to learn by gradient descent by gradient descent](http://arxiv.org/abs/1606.04474)
  Note that this implementation is not exactly the same as the original paper.
  For example, we use a slightly different gradient processing.
  """
  name: str = 'L2LGD2'
  mlp_dims: Sequence[int] = (16, 16)
  hidden_size: int = 8
  learning_rate: float = 1.0
  gradient_clip: float = -1.0
  eps: float = 1e-18

  def setup(self):
    # Set RNN
    self.rnn = nn.GRUCell()
    # Set MLP
    layer_dims = list(self.mlp_dims)
    layer_dims.append(1)
    layers = []
    for i in range(len(layer_dims)):
      layers.append(nn.Dense(layer_dims[i]))
      layers.append(nn.relu)
    layers.pop()
    self.mlp = nn.Sequential(layers)

  def __call__(self, h, g):
    # Clip the gradient to prevent large update
    g = lax.select(self.gradient_clip > 0, jnp.clip(g, -self.gradient_clip, self.gradient_clip), g)
    g = lax.stop_gradient(g)
    g_sign = jnp.sign(g)
    g_log = jnp.log(jnp.abs(g) + self.eps)
    # Expand parameter dimension so that the network is "coodinatewise"
    g_sign = lax.stop_gradient(g_sign[..., None])
    g_log = lax.stop_gradient(g_log[..., None])
    g_input = jnp.concatenate([g_sign, g_log], axis=-1)
    h, x = self.rnn(h, g_input)
    outs = self.mlp(x)
    # Slice outs into several elements
    o = outs[..., 0]
    # Compute the output: Delta theta
    out = -self.learning_rate * jnp.exp(o) * g
    return h, out

  def init_hidden_state(self, params):
    # Use fixed random key since default state init fn is just zeros.
    h = nn.GRUCell.initialize_carry(random.PRNGKey(0), params.shape, self.hidden_size)
    return h


def load_model_param(filepath):
  f = open(filepath, 'rb')
  model_param = pickle.load(f)
  model_param = tree_util.tree_map(jnp.array, model_param)
  f.close()
  return model_param


class OptimizerWrapper(object):
  """Optimizer Wrapper for learned optimizers: Optim4RL, LinearOptim, and L2LGD2."""
  def __init__(self, optimizer_name, cfg, seed):
    self.seed = seed
    self.optimizer_name = optimizer_name
    cfg['name'] = optimizer_name
    cfg['mlp_dims'] = tuple(cfg['mlp_dims'])
    cfg.setdefault('param_load_path', '')
    self.param_load_path = cfg['param_load_path']
    del cfg['param_load_path']
    # Set RNN optimizer
    if optimizer_name == 'Optim4RL':
      self.optimizer = Optim4RL(**cfg)
      self.is_rnn_output = lambda x: type(x)==tuple and type(x[0])==tuple and type(x[1])!=tuple
    elif optimizer_name in ['LinearOptim', 'L2LGD2']:
      self.optimizer = LinearOptim(**cfg)
      self.is_rnn_output = lambda x: type(x)==tuple and type(x[0])!=tuple and type(x[1])!=tuple
    # Initialize param for RNN optimizer
    if len(self.param_load_path) > 0:
      self.optim_param = load_model_param(self.param_load_path)
    else:
      dummy_grad = jnp.array([0.0])
      dummy_hidden_state = self.optimizer.init_hidden_state(dummy_grad)
      self.optim_param = self.optimizer.init(self.seed, dummy_hidden_state, dummy_grad)

  def init(self, param):
    """
    Initialize optim_state, i.e. hidden_state of RNN optimizer + optim parameter
    optim_state = optimizer.init(param)
    """
    hidden_state = tree_util.tree_map(self.optimizer.init_hidden_state, param)
    optim_state = OptimState(hidden_state=hidden_state, optim_param=self.optim_param)
    return optim_state

  def update(self, grad, optim_state, params=None):
    """param_update, optim_state = optimizer.update(grad, optim_state)"""
    out = jax.tree_util.tree_map(
      lambda grad, hidden: self.optimizer.apply(optim_state.optim_param, hidden, grad),
      grad,
      optim_state.hidden_state
    )
    # Split output into hidden_state and agent parameter update
    hidden_state = jax.tree_util.tree_map(lambda x: x[0], out, is_leaf=self.is_rnn_output)
    param_update = jax.tree_util.tree_map(lambda x: x[1], out, is_leaf=self.is_rnn_output)
    optim_state = optim_state.replace(hidden_state=hidden_state)
    return param_update, optim_state

  def update_with_param(self, optim_param, grad, optim_state, lr=1.0):
    """
    param_update, optim_state = optimizer.update(optim_param, grad, optim_state)
    Used for training the optimizer.
    """
    out = jax.tree_util.tree_map(
      lambda grad, hidden: self.optimizer.apply(optim_param, hidden, grad),
      grad,
      optim_state.hidden_state
    )
    # Split output into hidden_state and agent parameter update
    hidden_state = jax.tree_util.tree_map(lambda x: x[0], out, is_leaf=self.is_rnn_output)
    param_update = jax.tree_util.tree_map(lambda x: lr * x[1], out, is_leaf=self.is_rnn_output)
    optim_state = optim_state.replace(hidden_state=hidden_state)
    return param_update, optim_state


class StarWrapper(object):
  """Optimizer Wrapper for STAR."""
  def __init__(self, optimizer_name, cfg, seed):
    self.seed = seed
    self.optimizer_name = optimizer_name
    self.train_steps = cfg['train_steps']
    cfg.setdefault('param_load_path', '')
    self.param_load_path = cfg['param_load_path']
    del cfg['param_load_path'], cfg['train_steps']
    # Set Star optimizer
    assert optimizer_name == 'Star', 'Only Star is supported.'
    self.star_net = MLPNomLOpt(**cfg)
    # Initialize param for Star optimizer
    if len(self.param_load_path) > 0:
      optim_param = load_model_param(self.param_load_path)
    else:
      optim_param = self.star_net.init(self.seed)
    # Setup optimizer
    self.reset_optimizer(optim_param)

  def reset_optimizer(self, optim_param):
    self.optimizer = self.star_net.opt_fn(optim_param, is_training=True)

  def init(self, param):
    """Initialize optim_state"""
    optim_state = self.optimizer.init(params=param, num_steps=self.train_steps)
    return optim_state

  def get_optim_param(self):
    return self.optimizer.theta

  def update(self, grad, optim_state, loss):
    optim_state = self.optimizer.update(optim_state, grad, loss)
    return optim_state

  def update_with_param(self, optim_param, grad, optim_state, loss):
    """
    optim_state = optimizer.update(optim_param, grad, optim_state, loss)
    Used for training the optimizer.
    """
    self.optimizer.theta = optim_param # This is important, do not remove
    optim_state = self.optimizer.update(optim_state, grad, loss)
    return optim_state