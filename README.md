# Optim4RL

This is the official implementation of *Optim4RL*, a learning to optimize framework for reinforcement learning, introduced in our RLC 2024 paper [Learning to Optimize for Reinforcement Learning](https://arxiv.org/abs/2302.01470).


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

1. Install [JAX](https://github.com/google/jax) 0.4.19: See [Installing JAX](https://jax.readthedocs.io/en/latest/installation.html) for details. For example,

  ```bash
  pip install --upgrade "jax[cuda12_pip]==0.4.19" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  ```
    
2. Install other packages: see `requirements.txt`.

  ```bash
  pip install -r requirements.txt
  ```

3. Install `learned_optimization`:

  ```bash
  git clone https://github.com/google/learned_optimization.git
  cd learned_optimization
  pip install -e . && cd ..
  ```


## Usage

### Hyperparameter

All hyperparameters, including parameters for grid search, are stored in a configuration file in the directory `configs`. To run an experiment, a configuration index is first used to generate a configuration dict corresponding to this specific configuration index. Then we run an experiment defined by this configuration dict. All results, including log files, are saved in the directory `logs`. Please refer to the code for details.

For example, run the experiment with configuration file `a2c_catch.json` and configuration index `1`:

```bash
python main.py --config_file ./configs/a2c_catch.json --config_idx 1
```

To do a grid search, we first calculate the number of total combinations in a configuration file (e.g. `a2c_catch.json`):

```bash
python utils/sweeper.py
```

The output will be:

`The number of total combinations in a2c_catch.json: 2`

Then we run through all configuration indexes from `1` to `2`. The simplest way is using a bash script:

```bash
for index in {1..2}
do
  python main.py --config_file ./configs/a2c_catch.json --config_idx $index
done
```

[Parallel](https://www.gnu.org/software/parallel/) is usually a better choice to schedule a large number of jobs:

```bash
parallel --eta --ungroup python main.py --config_file ./configs/a2c_catch.json --config_idx {1} ::: $(seq 1 2)
```

Any configuration index with the same remainder (divided by the number of total combinations) should have the same configuration dict (except the random seed if `generate_random_seed` is `True`). So for multiple runs, we just need to add the number of total combinations to the configuration index. For example, 5 runs for configuration index `1`:

```bash
for index in 1 3 5 7 9
do
  python main.py --config_file ./configs/a2c_catch.json --config_idx $index
done
```

Or a simpler way:

```bash
parallel --eta --ungroup python main.py --config_file ./configs/a2c_catch.json --config_idx {1} ::: $(seq 1 2 10)
```

Please check `run.sh` for the details of all experiments.


### Experiment

- Benchmark classical optimizers: run `a2c_*.json` or `ppo_*.json`.
- Collect agent gradients and parameter updates during training: run `collect_*.json`.
- Meta-learn optimizers and test them:
  1. Train optimizers by running `meta_*.json`. The meta-parameters at different training stages will be saved in corresponding log directories. Note that for some experiments, more than 1 GPU/TPU (e.g., 4) is needed due to a large GPU/TPU memory requirement. For example, check `meta_rl_catch.json`.
  2. Use the paths of saved meta-parameters as the values for `param_load_path` in `lopt_*.json`.
  3. Run `lopt_*.json` to test learned optimizers with various meta-parameters. For example, check `lopt_rl_catch.json`.


### Analysis

To analyze the experimental results, just run:

```bash
python analysis_*.py
```

Inside `analysis_*.py`, `unfinished_index` will print out the configuration indexes of unfinished jobs based on the existence of the result file. `memory_info` will print out the memory usage information and generate a histogram to show the distribution of memory usages in the directory `logs/a2c_catch/0`. Similarly, `time_info` will print out the time information and generate a histogram to show the time distribution in the directory `logs/a2c_catch/0`. Finally, `analyze` will generate `csv` files that store training and test results. Please check `analysis_*.py` for more details. More functions are available in `utils/plotter.py`.


## Citation

If you find this work useful to your research, please cite our paper.

```bibtex
@inproceedings{lan2024learning,
  title={Learning to Optimize for Reinforcement Learning},
  author={Lan, Qingfeng and Mahmood, A. Rupam and Yan, Shuicheng and Xu, Zhongwen},
  booktitle={Reinforcement Learning Conference},
  year={2024},
  url={https://openreview.net/forum?id=JQuEXGj2r1}
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