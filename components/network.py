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
from jax import lax
from jax import numpy as jnp
import flax.linen as nn
from flax.linen.initializers import lecun_uniform

from typing import Any, Callable, Sequence
Initializer = Callable[..., Any]


class MLP(nn.Module):
  """Multilayer Perceptron"""
  layer_dims: Sequence[int]
  hidden_act: str = 'ReLU'
  output_act: str = 'Linear'
  kernel_init: Initializer = lecun_uniform()
  
  def setup(self):
    # Create layers
    layers = []
    for i in range(len(self.layer_dims)):      
      layers.append(nn.Dense(self.layer_dims[i], kernel_init=self.kernel_init))
      layers.append(getattr(nn, self.hidden_act.lower()))
    layers.pop()  
    if self.output_act != 'Linear':
      layers.append(getattr(nn, self.output_act.lower()))
    self.mlp = nn.Sequential(layers)

  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))  # flatten
    return self.mlp(x)


"""
C(N1-N2-...) represents convolutional layers with N1, N2, ... filters/features for each layer. 
D(N) represents a dense layer with N units.
"""
C16_D32 = nn.Sequential([
  nn.Conv(features=16, kernel_size=(2, 2)),
  nn.relu,
  lambda x: x.reshape((x.shape[0], -1)), # flatten
  nn.Dense(32),
  nn.relu
])

D32 = nn.Sequential([
  lambda x: x.reshape((x.shape[0], -1)), # flatten
  nn.Dense(32),
  nn.relu
])

D64_64 = nn.Sequential([
  lambda x: x.reshape((x.shape[0], -1)), # flatten
  nn.Dense(64),
  nn.relu,
  nn.Dense(64),
  nn.relu
])

def select_feature_net(env_name):
  # Select a feature net
  if 'small' in env_name:
    return D32
  elif 'big' in env_name:
    return C16_D32
  elif env_name in ['random_walk']:
    return lambda x: x
  elif env_name in ['Pendulum-v1', 'CartPole-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Acrobot-v1'] or 'bsuite' in env_name:
    return D64_64


class ActorVCriticNet(nn.Module):
  action_size: int
  env_name: str
  
  def setup(self):
    self.feature_net = select_feature_net(self.env_name)
    self.actor_net = nn.Dense(self.action_size)
    if self.env_name in ['random_walk']:
      use_bias = False
    else:
      use_bias = True
    self.critic_net = nn.Dense(1, use_bias=use_bias)

  def __call__(self, obs):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Compute state value and action ditribution logits
    v = self.critic_net(phi).squeeze()
    action_logits = self.actor_net(phi)
    return action_logits, v


class RobustRNN(nn.Module):
  name: str = 'RobustRNN'
  rnn_type: str = 'GRU'
  mlp_dims: Sequence[int] = ()
  hidden_size: int = 8
  out_size: int = 1
  eps: float = 1e-18
  
  def setup(self):
    # Set up RNN
    if self.rnn_type == 'LSTM':
      self.rnn = nn.OptimizedLSTMCell()
    elif self.rnn_type == 'GRU':
      self.rnn = nn.GRUCell()
    # Set up MLP
    layers = []
    layer_dims = list(self.mlp_dims)
    layer_dims.append(self.out_size)
    for i in range(len(layer_dims)):      
      layers.append(nn.Dense(layer_dims[i]))
      layers.append(nn.relu)
    layers.pop()
    self.mlp = nn.Sequential(layers)

  def __call__(self, h, g):
    g_sign = jnp.sign(g)
    g_log = jnp.log(jnp.abs(g) + self.eps)
    g_sign = lax.stop_gradient(g_sign[..., None])
    g_log = lax.stop_gradient(g_log[..., None])
    g_input = jnp.concatenate([g_sign, g_log], axis=-1)
    h, x = self.rnn(h, g_input)
    outs = self.mlp(x)
    out = g_sign[..., 0] * jnp.exp(outs[..., 0])
    return h, out

  def init_hidden_state(self, params):
    # Use fixed random key since default state init fn is just zeros.
    if self.rnn_type == 'LSTM':
      h = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0), params.shape, self.hidden_size)
    elif self.rnn_type == 'GRU':
      h = nn.GRUCell.initialize_carry(jax.random.PRNGKey(0), params.shape, self.hidden_size)
    return h


class NormalRNN(RobustRNN):
  name: str = 'NormalRNN'
  rnn_type: str = 'GRU'
  mlp_dims: Sequence[int] = ()
  hidden_size: int = 8
  out_size: int = 1

  def __call__(self, h, g):
    # Expand parameter dimension so that the network is "coodinatewise"
    g = lax.stop_gradient(g[..., None])
    g_input = g
    h, x = self.rnn(h, g_input)
    outs = self.mlp(x)
    out = outs[..., 0]
    return h, out