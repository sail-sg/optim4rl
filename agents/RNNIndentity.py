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

import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jax import lax, random, tree_util

from components import network
from components.optim import set_optimizer
from utils.logger import Logger


class RNNIndentity(object):
    """ "
    Train an RNN to approximate the identity function with agent gradients as input.
    """

    def __init__(self, cfg):
        self.agent_name = cfg["meta_net"]["name"]
        self.cfg = cfg
        self.config_idx = cfg["config_idx"]
        self.logger = Logger(cfg["logs_dir"])
        self.epoch = int(self.cfg["epoch"])
        self.log_path = cfg["logs_dir"] + "result_Train.feather"
        self.clip_ratio = 0.1
        # Create model
        self.model = self.create_meta_net()
        # Set optimizer
        self.optimizer = set_optimizer(
            cfg["optimizer"]["name"], cfg["optimizer"]["kwargs"], None
        )
        # Load data
        self.batches = self.load_data(self.cfg["seq_len"], self.cfg["datapath"])

    def create_meta_net(self):
        self.cfg["meta_net"]["mlp_dims"] = tuple(self.cfg["meta_net"]["mlp_dims"])
        return getattr(network, self.cfg["meta_net"]["name"])(**self.cfg["meta_net"])

    def load_data(self, seq_len, datapath):
        npzfile = np.load(datapath)
        xs = npzfile["x"]  # shape=(batch_size, len)
        self.batch_size, self.num_batch = xs.shape[0], xs.shape[1] // seq_len
        self.logger.info(
            f"dataset size: {xs.shape}, batch_size: {self.batch_size}, num_batch: {self.num_batch}"
        )
        batches = []  # shape=(num_batch, batch_size, seq_len)
        for i in range(self.num_batch):
            start = i * seq_len
            x = xs[:, start : start + seq_len]
            batches.append([x, x])
        batches = np.array(batches)
        return jax.device_put(batches)

    def compute_loss(self, param, hidden_state, batch):
        x, y = batch[0], batch[1]
        hidden_state, pred_y = lax.scan(
            f=lambda hidden, x_in: self.model.apply(param, hidden, x_in),
            init=hidden_state,
            xs=x,
        )
        loss = jnp.mean(jnp.square(pred_y - y))
        # Compute the accuracy of pred_y in y*(1+/-clip_ratio)
        mask = (
            (y >= 0)
            & (pred_y >= (1 - self.clip_ratio) * y)
            & (pred_y <= (1 + self.clip_ratio) * y)
        )
        mask = mask | (y < 0) & (pred_y <= (1 - self.clip_ratio) * y) & (
            pred_y >= (1 + self.clip_ratio) * y
        )
        perf = jnp.mean(mask)
        return loss, (lax.stop_gradient(perf), hidden_state)

    def train_step(self, param, hidden_state, optim_state, batch):
        (loss, (perf, hidden_state)), grad = jax.value_and_grad(
            self.compute_loss, has_aux=True
        )(param, hidden_state, batch)
        # Reduce mean gradient and mean loss across batch
        grad = lax.pmean(grad, axis_name="batch")
        loss = lax.pmean(loss, axis_name="batch")
        perf = lax.pmean(perf, axis_name="batch")
        param_update, optim_state = self.optimizer.update(grad, optim_state)
        param = optax.apply_updates(param, param_update)
        return param, hidden_state, optim_state, loss, perf

    def train(self):
        # Initialize model parameter
        seed = random.PRNGKey(self.cfg["seed"])
        dummy_input = jnp.array([0.0])
        dummy_hidden_state = self.model.init_hidden_state(dummy_input)
        param = self.model.init(seed, dummy_hidden_state, dummy_input)

        # Set optimizer state
        optim_state = self.optimizer.init(param)
        # Start training
        batched_train_step = jax.vmap(
            self.train_step,
            in_axes=(None, 0, None, 1),
            out_axes=(None, 0, None, None, None),
            axis_name="batch",
        )
        loss_list, perf_list = [], []
        start_time = time.time()
        self.best_perf = 0.0

        def f_loop(carry, batch):
            param, hidden_state, optim_state = carry
            param, hidden_state, optim_state, loss, perf = batched_train_step(
                param, hidden_state, optim_state, batch
            )
            carry = (param, hidden_state, optim_state)
            logs = dict(loss=loss, perf=perf)
            return carry, logs

        dummy_input = jnp.zeros((self.batch_size,))
        for i in range(1, self.epoch + 1):
            hidden_state = self.model.init_hidden_state(dummy_input)
            carry, logs = lax.scan(
                f=f_loop, init=(param, hidden_state, optim_state), xs=self.batches
            )
            param, hidden_state, optim_state = carry
            epoch_loss = jnp.mean(logs["loss"])
            epoch_perf = jnp.mean(logs["perf"])
            loss_list.append(epoch_loss)
            perf_list.append(epoch_perf)
            if epoch_perf > self.best_perf:
                self.best_perf = epoch_perf
                if self.cfg["save_param"]:
                    self.save_model_param(param, self.cfg["logs_dir"] + "param.pickle")
            if i % self.cfg["display_interval"] == 0:
                speed = (time.time() - start_time) / i
                eta = (self.epoch - i) * speed / 60 if speed > 0 else -1
                self.logger.info(
                    f"<{self.config_idx}> Epoch {i}/{self.epoch}: Loss={epoch_loss:.8f}, Perf={epoch_perf:.8f}, Speed={speed:.2f} (s/epoch), ETA={eta:.2f} (mins)"
                )
            if self.best_perf < 0.01 and i >= 4:
                self.logger.info(
                    f"Early stop at epoch {i} due to bad best performance: {self.best_perf:.8f}."
                )
                break
        self.logger.info(f"Best performance: {self.best_perf:.8f}.")
        self.save_logs(loss_list, perf_list)

    def save_logs(self, loss_list, perf_list):
        loss_list = np.array(jax.device_get(loss_list))
        perf_list = np.array(jax.device_get(perf_list))
        result = {
            "Agent": self.agent_name,
            "Epoch": np.array(range(len(loss_list))),
            "Loss": loss_list,
            "Perf": perf_list,
        }
        result = pd.DataFrame(result)
        result["Agent"] = result["Agent"].astype("category")
        result.to_feather(self.log_path)

    def save_model_param(self, model_param, filepath):
        f = open(filepath, "wb")
        pickle.dump(model_param, f)
        f.close()

    def load_model_param(self, filepath):
        f = open(filepath, "rb")
        model_param = pickle.load(f)
        model_param = tree_util.tree_map(jnp.array, model_param)
        f.close()
        return model_param
