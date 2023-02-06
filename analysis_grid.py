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

from utils.plotter import Plotter
from utils.sweeper import memory_info, time_info, unfinished_index


def get_process_result_dict(result, config_idx, mode="Train"):
    result_dict = {
        "Env": result["Env"][0],
        "Agent": result["Agent"][0],
        "Config Index": config_idx,
        "Return (mean)": result["Return"][-20:].mean(skipna=False),
    }
    return result_dict


def get_csv_result_dict(result, config_idx, mode="Train"):
    result_dict = {
        "Env": result["Env"][0],
        "Agent": result["Agent"][0],
        "Config Index": config_idx,
        "Return (mean)": result["Return (mean)"].mean(skipna=False),
        "Return (se)": result["Return (mean)"].sem(ddof=0),
    }
    return result_dict


cfg = {
    "exp": "exp_name",
    "merged": True,
    "x_label": "Step",
    "y_label": "Return",
    "rolling_score_window": -1,
    "hue_label": "Agent",
    "show": False,
    "imgType": "png",
    "ci": "sd",
    "x_format": None,
    "y_format": None,
    "xlim": {"min": None, "max": None},
    "ylim": {"min": None, "max": None},
    "EMA": True,
    "loc": "upper left",
    "sweep_keys": ["agent_optimizer/name", "agent_optimizer/kwargs/learning_rate"],
    "sort_by": ["Return (mean)", "Return (se)"],
    "ascending": [False, True],
    "runs": 1,
}


def analyze(exp, runs=1):
    cfg["exp"] = exp
    cfg["runs"] = runs
    plotter = Plotter(cfg)

    modes = []
    if "bdl" in exp:
        modes.append("big_dense_long")
    if "bss" in exp:
        modes.append("big_sparse_short")
    if "sdl" in exp:
        modes.append("small_dense_long")
    if "sds" in exp:
        modes.append("small_dense_short")
    if "short" in exp:
        modes = [
            "small_sparse_short",
            "small_dense_short",
            "big_sparse_short",
            "big_dense_short",
        ]
    if "long" in exp:
        modes = [
            "small_sparse_long",
            "small_dense_long",
            "big_sparse_long",
            "big_dense_long",
        ]
    if "grid" in exp:
        modes = [
            "small_sparse_short",
            "small_sparse_long",
            "small_dense_short",
            "small_dense_long",
            "big_sparse_short",
            "big_sparse_long",
            "big_dense_short",
            "big_dense_long",
        ]

    sweep_keys_dict = dict(
        a2c=["agent_optimizer/name", "agent_optimizer/kwargs/learning_rate"],
        collect=[
            "agent_optimizer/name",
            "agent_optimizer/kwargs/learning_rate",
            "env/reward_scaling",
        ],
        lopt=[
            "agent_optimizer/name",
            "agent_optimizer/kwargs/learning_rate",
            "agent_optimizer/kwargs/param_load_path",
        ],
        meta=[
            "agent_optimizer/name",
            "agent_optimizer/kwargs/learning_rate",
            "meta_optimizer/kwargs/learning_rate",
        ],
        online=[
            "agent_optimizer/name",
            "agent_optimizer/kwargs/learning_rate",
            "meta_optimizer/kwargs/learning_rate",
        ],
        star=[
            "agent_optimizer/name",
            "agent_optimizer/kwargs/step_mult",
            "agent_optimizer/kwargs/nominal_stepsize",
            "agent_optimizer/kwargs/weight_decay",
            "meta_optimizer/kwargs/learning_rate",
        ],
    )
    algo = exp.split("_")[-1].rstrip("0123456789")
    plotter.sweep_keys = sweep_keys_dict[algo]

    for mode in modes:
        plotter.csv_results(mode, get_csv_result_dict, get_process_result_dict)
        plotter.plot_results(mode=mode, indexes="all")


if __name__ == "__main__":
    exp, runs = "sds_lopt", 10
    unfinished_index(exp, runs=runs)
    memory_info(exp, runs=runs)
    time_info(exp, runs=runs)
    analyze(exp, runs=runs)
