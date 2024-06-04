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

import jax
import flax
import optax
import flax.linen as nn
import jax.numpy as jnp
from copy import deepcopy
from typing import Sequence
from jax import lax, random, tree_util
from flax.linen.initializers import zeros_init
from components.star import MLPNomLOpt

from utils.helper import load_model_param


activations = {
  'ReLU': nn.relu,
  'ELU': nn.elu,
  'Softplus': nn.softplus,
  'LeakyReLU': nn.leaky_relu,
  'Tanh': jnp.tanh,
  'Sigmoid': nn.sigmoid,
  'Exp': jnp.exp
}


@flax.struct.dataclass
class OptimState:
  """Contains training state for the learner."""
  hidden_state: flax.core.FrozenDict
  optim_param: flax.core.FrozenDict
  iteration: jnp.ndarray


def set_optim(optim_name, original_optim_cfg, key):
  optim_cfg = deepcopy(original_optim_cfg)
  if optim_name in ['LinearOptim', 'L2LGD2'] or 'Optim4RL' in optim_name:
    optim = OptimizerWrapper(optim_name, optim_cfg, key)
  elif optim_name == 'Star':
    optim = StarWrapper(optim_name, optim_cfg, key)
  else:
    optim_cfg.setdefault('grad_clip', -1)
    optim_cfg.setdefault('grad_norm', -1)
    grad_clip = optim_cfg['grad_clip']
    grad_norm = optim_cfg['grad_norm']
    del optim_cfg['grad_clip'], optim_cfg['grad_norm']
    if grad_clip > 0:
      optim = optax.chain(
        optax.clip(grad_clip),
        getattr(optax, optim_name.lower())(**optim_cfg)
      )
    elif grad_norm > 0:
      optim = optax.chain(
        optax.clip_by_global_norm(grad_norm),
        getattr(optax, optim_name.lower())(**optim_cfg)
      )
    else:
      optim = getattr(optax, optim_name.lower())(**optim_cfg)
  return optim


class Optim4RL(nn.Module):
  name: str = 'Optim4RL'
  rnn_hidden_act: str = 'Tanh'
  mlp_dims: Sequence[int] = ()
  hidden_size: int = 8
  learning_rate: float = 1.0
  eps: float = 1e-8
  out_size1: int = 1
  out_size2: int = 1

  def setup(self):
    # Set up RNNs
    act_fn = activations[self.rnn_hidden_act]
    self.rnn1 = nn.GRUCell(features=self.hidden_size, activation_fn=act_fn)
    self.rnn2 = nn.GRUCell(features=self.hidden_size, activation_fn=act_fn)
    # Set up MLPs
    layer_dims1, layer_dims2 = list(self.mlp_dims), list(self.mlp_dims)
    layer_dims1.append(self.out_size1)
    layer_dims2.append(self.out_size2)
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

  def __call__(self, h, g, t=0):
    # Expand parameter dimension so that the network is "coodinatewise"
    g = lax.stop_gradient(g[..., None])
    g_square = lax.stop_gradient(jnp.square(g))
    g_sign = lax.stop_gradient(jnp.sign(g))
    # RNN
    h1, h2 = h
    # Compute m: 1st pseudo moment estimate
    h1, x1 = self.rnn1(h1, g)
    o1 = self.mlp1(x1)
    m = g_sign * jnp.exp(o1)
    # Compute v: 2nd pseudo moment estimate
    h2, x2 = self.rnn2(h2, g_square)
    o2 = self.mlp2(x2)
    rsqrt_v = lax.rsqrt(jnp.exp(o2) + self.eps)
    # Compute the output: Delta theta
    out = -self.learning_rate * m * rsqrt_v
    return jnp.array([h1, h2]), out[..., 0]

  def init_hidden_state(self, param):
    # Use fixed random key since default state init fn is just zeros.
    seed = random.PRNGKey(0)
    mem_shape = param.shape + (self.hidden_size,)
    h = jnp.array([
      zeros_init()(seed, mem_shape),
      zeros_init()(seed, mem_shape)
    ])
    return h


class LinearOptim(nn.Module):
  name: str = 'LinearOptim'
  mlp_dims: Sequence[int] = ()
  hidden_size: int = 8
  learning_rate: float = 1.0
  eps: float = 1e-8
  out_size: int = 3
  
  def setup(self):
    # Set up RNN
    self.rnn = nn.GRUCell(features=self.hidden_size)
    # Set up MLP
    layer_dims = list(self.mlp_dims)
    layer_dims.append(self.out_size)
    layers = []
    for i in range(len(layer_dims)):      
      layers.append(nn.Dense(layer_dims[i]))
      layers.append(nn.relu)
    layers.pop()
    self.mlp = nn.Sequential(layers)

  def __call__(self, h, g, t=0):
    # Expand parameter dimension so that the network is "coodinatewise"
    g = lax.stop_gradient(g[..., None])
    h, x = self.rnn(h, g)
    outs = self.mlp(x)
    # Slice outs into several elements
    o1, o2, o3 = outs[..., 0], outs[..., 1], outs[..., 2]
    # Compute the output: Delta theta
    out = -self.learning_rate * (jnp.exp(o1) * g[..., 0] + jnp.exp(o2) * o3)
    return h, out

  def init_hidden_state(self, param):
    # Use fixed random key since default state init fn is just zeros.
    seed = random.PRNGKey(0)
    mem_shape = param.shape + (self.hidden_size,)
    h = jnp.array(zeros_init()(seed, mem_shape))
    return h


class L2LGD2(LinearOptim):
  name: str = 'L2LGD2'
  mlp_dims: Sequence[int] = ()
  hidden_size: int = 8
  learning_rate: float = 1.0
  p: int = 10
  out_size: int = 1

  def setup(self):
    super().setup()
    self.f_select = jax.vmap(lambda s, x, y: lax.select(s>=-1.0, x, y))

  def __call__(self, h, g, t=0):
    # Expand parameter dimension so that the network is "coodinatewise"
    g = lax.stop_gradient(g[..., None])
    g_sign = jnp.sign(g)
    g_log = jnp.log(jnp.abs(g) + self.eps) / self.p
    g_in1 = jnp.concatenate([g_log, g_sign], axis=-1).reshape((-1,2))
    g_in2 = jnp.concatenate([-1.0*jnp.ones_like(g), jnp.exp(self.p)*g], axis=-1).reshape((-1,2))
    g_in = self.f_select(g_log.reshape(-1), g_in1, g_in2)
    g_in = g_in.reshape(g.shape[:-1]+(2,))
    h, x = self.rnn(h, g_in)
    outs = self.mlp(x)
    # Compute the output: Delta theta
    out = -self.learning_rate * jnp.exp(outs) * g
    return h, out[..., 0]
  

class OptimizerWrapper(object):
  """Optimizer Wrapper for learned optimizers."""
  def __init__(self, optim_name, cfg, seed):
    self.seed = seed
    self.optim_name = optim_name
    cfg['name'] = optim_name
    if 'mlp_dims' in cfg.keys():
      cfg['mlp_dims'] = tuple(cfg['mlp_dims'])
    cfg.setdefault('param_load_path', '')
    self.param_load_path = cfg['param_load_path']
    cfg.setdefault('grad_clip', -1.0)
    self.grad_clip = cfg['grad_clip']
    del cfg['param_load_path'], cfg['grad_clip']
    # Set RNN optimizer
    assert optim_name in ['LinearOptim', 'L2LGD2'] or 'Optim4RL' in optim_name, f'{optim_name} is not supported.'
    self.optim = eval(optim_name)(**cfg)
    self.is_rnn_output = lambda x: type(x)==tuple and type(x[0])!=tuple and type(x[1])!=tuple
    # Initialize param for RNN optimizer
    if len(self.param_load_path)>0:
      self.optim_param = load_model_param(self.param_load_path)
    else:
      dummy_grad = jnp.array([0.0])
      dummy_hidden_state = self.optim.init_hidden_state(dummy_grad)
      self.optim_param = self.optim.init(self.seed, dummy_hidden_state, dummy_grad)

  def init(self, param):
    """
    Initialize optim_state, i.e. hidden_state of RNN optimizer
    optim_state = optim.init(param)
    """
    hidden_state = tree_util.tree_map(self.optim.init_hidden_state, param)
    optim_state = OptimState(hidden_state=hidden_state, optim_param=self.optim_param, iteration=0)
    return optim_state

  def update(self, grad, optim_state):
    """param_update, optim_state = optim.update(grad, optim_state)"""
    # Clip the gradient to prevent large update
    grad = lax.cond(
      self.grad_clip > 0,
      lambda x: jax.tree_util.tree_map(lambda g: jnp.clip(g, -self.grad_clip, self.grad_clip), x),
      lambda x: x,
      grad
    )
    out = jax.tree_util.tree_map(
      lambda hidden, grad: self.optim.apply(optim_state.optim_param, hidden, grad, optim_state.iteration),
      optim_state.hidden_state,
      grad
    )
    # Split output into hidden_state and agent parameter update
    hidden_state = jax.tree_util.tree_map(lambda x: x[0], out, is_leaf=self.is_rnn_output)
    param_update = jax.tree_util.tree_map(lambda x: x[1], out, is_leaf=self.is_rnn_output)
    optim_state = optim_state.replace(hidden_state=hidden_state)
    optim_state = optim_state.replace(iteration=optim_state.iteration+1)
    return param_update, optim_state

  def update_with_param(self, optim_param, grad, optim_state, lr=1.0):
    """
    param_update, optim_state = optim.update(optim_param, grad, optim_state)
    Used for training the optimizer.
    """
    grad = lax.cond(
      self.grad_clip > 0,
      lambda x: jax.tree_util.tree_map(lambda g: jnp.clip(g, -self.grad_clip, self.grad_clip), x),
      lambda x: x,
      grad
    )
    out = jax.tree_util.tree_map(
      lambda hidden, grad: self.optim.apply(optim_param, hidden, grad, optim_state.iteration),
      optim_state.hidden_state,
      grad,
    )
    # Split output into hidden_state and agent parameter update
    hidden_state = jax.tree_util.tree_map(lambda x: x[0], out, is_leaf=self.is_rnn_output)
    param_update = jax.tree_util.tree_map(lambda x: lr * x[1], out, is_leaf=self.is_rnn_output)
    optim_state = optim_state.replace(hidden_state=hidden_state)
    optim_state = optim_state.replace(iteration=optim_state.iteration+1)
    return param_update, optim_state


class StarWrapper(object):
  """Optimizer Wrapper for STAR."""
  def __init__(self, optim_name, cfg, seed):
    self.seed = seed
    self.optim_name = optim_name
    self.train_steps = cfg['train_steps']
    cfg.setdefault('param_load_path', '')
    self.param_load_path = cfg['param_load_path']
    cfg.setdefault('grad_clip', -1.0)
    self.grad_clip = cfg['grad_clip']
    # Rename lr to step_mult
    cfg['step_mult'] = cfg['learning_rate']
    cfg.setdefault('nominal_stepsize', 1e-3)
    del cfg['param_load_path'], cfg['train_steps'], cfg['grad_clip'], cfg['learning_rate']
    # Set Star optimizer
    assert optim_name == 'Star', 'Only Star is supported.'
    self.star_net = MLPNomLOpt(**cfg)
    # Initialize param for Star optimizer
    if len(self.param_load_path)>0:
      optim_param = load_model_param(self.param_load_path)
    else:
      optim_param = self.star_net.init(self.seed)
    # Setup optimizer
    self.reset_optimizer(optim_param)

  def reset_optimizer(self, optim_param):
    self.optim = self.star_net.opt_fn(optim_param, is_training=True)

  def init(self, param):
    """Initialize optim_state"""
    optim_state = self.optim.init(params=param, num_steps=self.train_steps)
    return optim_state

  def get_optim_param(self):
    return self.optim.theta

  def update(self, grad, optim_state, loss):
    grad = lax.cond(
      self.grad_clip > 0,
      lambda x: jax.tree_util.tree_map(lambda g: jnp.clip(g, -self.grad_clip, self.grad_clip), x),
      lambda x: x,
      grad
    )
    optim_state = self.optim.update(optim_state, grad, loss)
    return optim_state

  def update_with_param(self, optim_param, grad, optim_state, loss):
    """
    param_update, optim_state = optim.update(optim_param, grad, optim_state)
    Used for training the optimizer.
    """
    grad = lax.cond(
      self.grad_clip > 0,
      lambda x: jax.tree_util.tree_map(lambda g: jnp.clip(g, -self.grad_clip, self.grad_clip), x),
      lambda x: x,
      grad
    )
    self.optim.theta = optim_param # This is important
    optim_state = self.optim.update(optim_state, grad, loss)
    return optim_state