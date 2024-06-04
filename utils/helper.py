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

import os
import random
import pickle
import psutil
import datetime
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, tree_util


def get_time_str():
  return datetime.datetime.now().strftime("%y.%m.%d-%H:%M:%S")


def rss_memory_usage():
  '''
  Return the resident memory usage in MB
  '''
  process = psutil.Process(os.getpid())
  mem = process.memory_info().rss / float(2 ** 20)
  return mem


def set_random_seed(seed):
  '''
  Set all random seeds
  '''
  random.seed(seed)
  np.random.seed(seed)


def make_dir(dir):
  if not os.path.exists(dir): 
    os.makedirs(dir, exist_ok=True)


jitted_split = partial(jit, static_argnames=['num'])(jax.random.split)


def pytree2array(values):
  leaves = tree_util.tree_leaves(lax.stop_gradient(values))
  a = jnp.concatenate(leaves, axis=None)
  return a


def tree_stack(trees, axis=0):
  '''
  From: https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
  Takes a list of trees and stacks every corresponding leaf.
  For example, given two trees ((a, b), c) and ((a', b'), c'), returns
  ((stack(a, a'), stack(b, b')), stack(c, c')).
  Useful for turning a list of objects into something you can feed to a
  vmapped function.
  '''
  leaves_list = []
  treedef_list = []
  for tree in trees:
    leaves, treedef = tree_util.tree_flatten(tree)
    leaves_list.append(leaves)
    treedef_list.append(treedef)
  grouped_leaves = zip(*leaves_list)
  result_leaves = [jnp.stack(l, axis=axis) for l in grouped_leaves]
  return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
  '''
  From: https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
  Takes a tree and turns it into a list of trees. Inverse of tree_stack.
  For example, given a tree ((a, b), c), where a, b, and c all have first
  dimension k, will make k trees
  [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
  Useful for turning the output of a vmapped function into normal objects.
  '''
  leaves, treedef = tree_util.tree_flatten(tree)
  n_trees = leaves[0].shape[0]
  new_leaves = [[] for _ in range(n_trees)]
  for leaf in leaves:
    for i in range(n_trees):
      new_leaves[i].append(leaf[i])
  new_trees = [treedef.unflatten(l) for l in new_leaves]
  return new_trees


def tree_transpose(list_of_trees):
  '''
  Convert a list of trees of identical structure into a single tree of lists.
  We can replace tree_stack with tree_transpose.
  JAX also provides jax.tree_transpose, 
  which is more verbose, but allows you specify the structure of the inner and outer Pytree for more flexibility.
  '''
  return tree_util.tree_map(lambda *xs: jnp.array(xs), *list_of_trees)


def tree_concatenate(trees):
  '''
  Adapted from tree_stack.
  Takes a list of trees and stacks every corresponding leaf.
  For example, given two trees ((a, b), c) and ((a', b'), c'), returns
  ((concatenate(a, a'), concatenate(b, b')), concatenate(c, c')).
  '''
  leaves_list = []
  treedef_list = []
  for tree in trees:
    leaves, treedef = tree_util.tree_flatten(tree)
    leaves_list.append(leaves)
    treedef_list.append(treedef)
  grouped_leaves = zip(*leaves_list)
  result_leaves = [jnp.concatenate(l) for l in grouped_leaves]
  return treedef_list[0].unflatten(result_leaves)


def save_model_param(model_param, filepath):
  with open(filepath, 'wb') as f:
    pickle.dump(model_param, f)


def load_model_param(filepath):
  f = open(filepath, 'rb')
  model_param = pickle.load(f)
  model_param = tree_util.tree_map(jnp.array, model_param)
  f.close()
  return model_param