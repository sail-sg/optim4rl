import jax
import numpy as np
from jax import random
import jax.numpy as jnp


def load_data(dataset, seed=None, batch_size=1):
  dataset = dataset.lower()
  assert dataset == 'mnist'
  dummy_input = jnp.ones([1, 28, 28, 1])
  data_path = f'./data/{dataset}/'
  train_file, test_file = np.load(data_path+'train.npz'), np.load(data_path+'test.npz')
  train_data = dict(
    x = jnp.float32(train_file['x']) / 255.,
    y = jnp.array(train_file['y'])
  )
  test_data = dict(
    x = jnp.float32(test_file['x']) / 255.,
    y = jnp.array(test_file['y'])
  )
  train_data = make_batches(train_data, batch_size, seed)
  data_loader = {
    'Train': train_data,
    'Test': test_data,
    'dummy_input': dummy_input
  }
  return data_loader


def make_batches(data, batch_size, seed=None):
  # Sort data according y label
  index = jnp.argsort(data['y'])
  data['x'] = data['x'][index]
  data['y'] = data['y'][index]
  # Find indexes for each label
  label_idxs = np.where(data['y'][1:] != data['y'][:-1])[0] + 1
  label_idxs = np.insert(label_idxs, 0, 0)
  min_num_batch = np.min(label_idxs[1:] - label_idxs[:-1]) // batch_size
  # Truncate into batches and shuffle the training dataset
  assert seed is not None, 'Need a random seed.'
  new_idxs = jnp.array([], dtype=int)  
  for i in range(0, len(label_idxs)):
    seed, shuffle_seed = random.split(seed)
    unshuffled_idxs = jnp.array([], dtype=int)
    idxs = jnp.array(range(label_idxs[i], label_idxs[i]+min_num_batch*batch_size), dtype=int)
    unshuffled_idxs = jnp.append(unshuffled_idxs, idxs)
    shuffled_idxs = jax.random.permutation(shuffle_seed, unshuffled_idxs, independent=True)
    new_idxs = jnp.append(new_idxs, shuffled_idxs)
  data['x'] = data['x'][new_idxs]
  data['y'] = data['y'][new_idxs]
  return data