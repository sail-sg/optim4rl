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
        "Config Index": config_idx,
        "Loss (mean)": result["Loss"][-5:].mean(skipna=False),
        "Perf (mean)": result["Perf"][-5:].mean(skipna=False),
    }
    return result_dict


def get_csv_result_dict(result, config_idx, mode="Train"):
    result_dict = {
        "Config Index": config_idx,
        "Loss (mean)": result["Loss (mean)"].mean(skipna=False),
        "Perf (mean)": result["Perf (mean)"].mean(skipna=False),
    }
    return result_dict


cfg = {
    "exp": "exp_name",
    "merged": True,
    "x_label": "Epoch",
    "y_label": "Perf",
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
    "sweep_keys": ["meta_net/name", "optimizer/kwargs/learning_rate"],
    "sort_by": ["Perf (mean)", "Loss (mean)"],
    "ascending": [False, True],
    "runs": 1,
}


def analyze(exp, runs=1):
    cfg["exp"] = exp
    cfg["runs"] = runs
    plotter = Plotter(cfg)

    plotter.csv_results("Train", get_csv_result_dict, get_process_result_dict)
    plotter.plot_results(mode="Train", indexes="all")


if __name__ == "__main__":
    exp, runs = "bdl_identity", 10
    unfinished_index(exp, runs=runs)
    memory_info(exp, runs=runs)
    time_info(exp, runs=runs)
    analyze(exp, runs=runs)
