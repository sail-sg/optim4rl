clear

# Download MNIST
python download.py

# Collect
python main.py --config_file ./configs/collect_mnist.json --config_idx 1
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/collect_bdl.json --config_idx {1} ::: $(seq 1 2)

# A2C
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/a2c_catch.json --config_idx {1} ::: $(seq 1 20)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/a2c_grid.json --config_idx {1} ::: $(seq 1 120)

# Meta A2C jobs
## Catch
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_rl_catch.json --config_idx {1} ::: $(seq 1 20)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_l2l_catch.json --config_idx {1} ::: $(seq 1 20)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_lin_catch.json --config_idx {1} ::: $(seq 1 20)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_star_catch.json --config_idx {1} ::: $(seq 1 20)
## sdl
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_rl_sdl.json --config_idx {1} ::: $(seq 1 40)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_rlp_sdl.json --config_idx {1} ::: $(seq 1 10)
## bdl
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_rl_bdl.json --config_idx {1} ::: $(seq 1 40)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_rlp_bdl.json --config_idx {1} ::: $(seq 1 10)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_l2l_bdl.json --config_idx {1} ::: $(seq 1 20)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_lin_bdl.json --config_idx {1} ::: $(seq 1 20)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_star_bdl.json --config_idx {1} ::: $(seq 1 20)
## Gridworld
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_rl_grid.json --config_idx {1} ::: $(seq 1 96)

# Lopt A2C jobs
### catch
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_rl_catch.json --config_idx {1} ::: $(seq 1 200)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_l2l_catch.json --config_idx {1} ::: $(seq 1 200)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_lin_catch.json --config_idx {1} ::: $(seq 1 200)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_star_catch.json --config_idx {1} ::: $(seq 1 200)
## sdl
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_rl_sdl.json --config_idx {1} ::: $(seq 1 400)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_rlp_sdl.json --config_idx {1} ::: $(seq 1 100)
## bdl
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_rl_bdl.json --config_idx {1} ::: $(seq 1 400)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_rlp_bdl.json --config_idx {1} ::: $(seq 1 100)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_lin_bdl.json --config_idx {1} ::: $(seq 1 400)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_l2l_bdl.json --config_idx {1} ::: $(seq 1 400)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_star_bdl.json --config_idx {1} ::: $(seq 1 400)

# PPO
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/ppo_ant.json --config_idx {1} ::: $(seq 1 20)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/ppo_humanoid.json --config_idx {1} ::: $(seq 1 20)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/ppo_pendulum.json --config_idx {1} ::: $(seq 1 20)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/ppo_walker2d.json --config_idx {1} ::: $(seq 1 20)

# Meta PPO jobs
## Ant
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_rl_ant.json --config_idx {1} ::: $(seq 1 50)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_rlp_ant.json --config_idx {1} ::: $(seq 1 10)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_l2l_ant.json --config_idx {1} ::: $(seq 1 50)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_lin_ant.json --config_idx {1} ::: $(seq 1 50)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_star_ant.json --config_idx {1} ::: $(seq 1 50)
## Humanoid
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_rl_humanoid.json --config_idx {1} ::: $(seq 1 50)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_rlp_humanoid.json --config_idx {1} ::: $(seq 1 10)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_l2l_humanoid.json --config_idx {1} ::: $(seq 1 50)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_lin_humanoid.json --config_idx {1} ::: $(seq 1 50)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/meta_star_humanoid.json --config_idx {1} ::: $(seq 1 50)

# Lopt PPO jobs
## Ant
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_rl_ant.json --config_idx {1} ::: $(seq 1 500)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_rlp_ant.json --config_idx {1} ::: $(seq 1 100)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_lin_ant.json --config_idx {1} ::: $(seq 1 500)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_l2l_ant.json --config_idx {1} ::: $(seq 1 500)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_star_ant.json --config_idx {1} ::: $(seq 1 500)
## Humanoid
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_rl_humanoid.json --config_idx {1} ::: $(seq 1 500)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_rlp_humanoid.json --config_idx {1} ::: $(seq 1 100)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_lin_humanoid.json --config_idx {1} ::: $(seq 1 500)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_l2l_humanoid.json --config_idx {1} ::: $(seq 1 500)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_star_humanoid.json --config_idx {1} ::: $(seq 1 500)

## Lopt: Gridworld --> Brax
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_rl_grid_ant.json --config_idx {1} ::: $(seq 1 960)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_rl_grid_humanoid.json --config_idx {1} ::: $(seq 1 960)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_rl_grid_pendulum.json --config_idx {1} ::: $(seq 1 960)
parallel --eta --ungroup --j 1 python main.py --config_file ./configs/lopt_rl_grid_walker2d.json --config_idx {1} ::: $(seq 1 960)