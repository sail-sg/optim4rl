# Optim4RL

Optim4RL is a framework of learning to optimize for reinforcement learning, introduced in our paper [Learning to Optimize for Reinforcement Learning](https://arxiv.org/abs/2302.01470).

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
  - [Hyperparameter](#hyperparameter)
  - [Experiment](#experiment)
  - [Analysis](#analysis)
- [Citation](#citation)
- [License](#license)
- [Acknowledgement](#acknowledgement)
- [Disclaimer](#disclaimer)

## Installation

1. Install `learned_optimization`:

   ```bash
   git clone https://github.com/google/learned_optimization.git
   cd learned_optimization
   pip install -e .
   ```

2. Install [JAX](https://github.com/google/jax):

- TPU:

  ```bash
  pip install "jax[tpu]>=0.3.23" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  ```

- GPU: See [JAX GPU (CUDA) installation](https://github.com/google/jax#pip-installation-gpu-cuda) for details. An example:

  ```bash
  pip install "jax[cuda11_cudnn82]>=0.3.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  ```

3. Install [Brax](https://github.com/google/brax):

   ```bash
   pip install git+https://github.com/google/brax.git
   ```

4. Install other packages: see `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Hyperparameter

All hyperparameters, including parameters for grid search, are stored in a configuration file in the directory `configs`. To run an experiment, a configuration index is first used to generate a configuration dict corresponding to this specific configuration index. Then we run an experiment defined by this configuration dict. All results, including log files, are saved in the directory `logs`. Please refer to the code for details.

For example, run the experiment with configuration file `sds_a2c.json` and configuration index `1`:

```bash
python main.py --config_file ./configs/sds_a2c.json --config_idx 1
```

To do a grid search, we first calculate the number of total combinations in a configuration file (e.g. `sds_a2c.json`):

```bash
python utils/sweeper.py
```

The output will be:

`The number of total combinations in sds_a2c.json: 12`

Then we run through all configuration indexes from `1` to `12`. The simplest way is using a bash script:

```bash
for index in {1..12}
do
  python main.py --config_file ./configs/sds_a2c.json --config_idx $index
done
```

[Parallel](https://www.gnu.org/software/parallel/) is usually a better choice to schedule a large number of jobs:

```bash
parallel --eta --ungroup python main.py --config_file ./configs/sds_a2c.json --config_idx {1} ::: $(seq 1 12)
```

Any configuration index with the same remainder (divided by the number of total combinations) should have the same configuration dict (except the random seed if `generate_random_seed` is `True`). So for multiple runs, we just need to add the number of total combinations to the configuration index. For example, 5 runs for configuration index `1`:

```bash
for index in 1 13 25 37 49
do
  python main.py --config_file ./configs/sds_a2c.json --config_idx $index
done
```

Or a simpler way:

```bash
parallel --eta --ungroup python main.py --config_file ./configs/sds_a2c.json --config_idx {1} ::: $(seq 1 12 60)
```

### Experiment

- Benchmark classical optimizers: run `*_a2c.json` or `*_ppo.json`.
- Collect agent gradients and parameter updates during training: run `*_collect.json`.
- Approximate the identity function with RNNs, given agent gradients as input:
  1. Collect agent gradients and parameter updates by running `bdl_collect.json`.
  2. Run `bdl_identity.json`.
- Meta-learn optimizers and test them:
  1. Train optimizers by running `*_meta.json` or `*_star.json`. The meta-parameters at different training stages will be saved in corresponding log directories. Note that for some experiments, more than 1 GPU/TPU is needed due to a large GPU/TPU memory requirement.
  2. Use the paths of saved meta-parameters as the values for `param_load_path` in `*_lopt.json`.
  3. Run `*_lopt.json` to test learned optimizers with various meta-parameters. See `sds_lopt.json` for an example.

### Analysis

To analyze the experimental results, just run:

```bash
python analysis_*.py
```

Inside `analysis_*.py`, `unfinished_index` will print out the configuration indexes of unfinished jobs based on the existence of the result file. `memory_info` will print out the memory usage information and generate a histogram to show the distribution of memory usages in the directory `logs/sds_a2c/0`. Similarly, `time_info` will print out the time information and generate a histogram to show the time distribution in the directory `logs/sds_a2c/0`. Finally, `analyze` will generate `csv` files that store training and test results. Please check `analysis_*.py` for more details. More functions are available in `utils/plotter.py`.

## Citation

If you find this work useful to your research, please cite our paper.

```bibtex
@article{lan2023learning,
  title={Learning to Optimize for Reinforcement Learning},
  author={Lan, Qingfeng and Mahmood, A. Rupam and Yan, Shuicheng and Xu, Zhongwen},
  journal={arXiv preprint arXiv:2302.01470},
  year={2023}
}
```

## License

`Optim4RL` is distributed under the terms of the [Apache2](https://www.apache.org/licenses/LICENSE-2.0) license.

## Acknowledgement

We thank the following projects which provide great references:

- [Brax](https://github.com/google/brax)
- [Discovering Reinforcement Learning Algorithms](https://github.com/epignatelli/discovering-reinforcement-learning-algorithms)
- [learned_optimization](https://github.com/google/learned_optimization)
- [Explorer](https://github.com/qlan3/Explorer)

## Disclaimer

This is not an official Sea Limited or Garena Online Private Limited product.