import os
import numpy as np
import tensorflow_datasets as tfds
os.environ['NO_GCE_CHECK'] = 'true'


dataset = 'mnist'
ds_builder = tfds.builder(dataset, data_dir='./data/')
ds_builder.download_and_prepare()
train_data = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
test_data = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
np.savez(f'./data/{dataset}/train.npz', x=train_data['image'], y=train_data['label'])
np.savez(f'./data/{dataset}/test.npz', x=test_data['image'], y=test_data['label'])