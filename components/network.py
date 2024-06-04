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
import distrax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import lecun_uniform

from typing import Any, Callable, Sequence


Initializer = Callable[..., Any]


class MLP(nn.Module):
  '''Multilayer Perceptron'''
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

D256 = nn.Sequential([
  lambda x: x.reshape((x.shape[0], -1)), # flatten
  nn.Dense(256),
  nn.relu
])


class MNIST_CNN(nn.Module):
  output_dim: int = 10
  
  def setup(self):
    self.feature_net = C16_D32
    self.head = nn.Dense(self.output_dim)
  
  def __call__(self, obs):
    phi = self.feature_net(obs)
    logits = self.head(phi)
    return logits


def select_feature_net(env_name):
  # Select a feature net
  if 'small' in env_name:
    return D32
  elif 'big' in env_name:
    return C16_D32
  elif env_name in ['random_walk']:
    return lambda x: x
  elif env_name in ['catch']:
    return D256


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