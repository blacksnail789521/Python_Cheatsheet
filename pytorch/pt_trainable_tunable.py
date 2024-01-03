import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import random
from tqdm import tqdm
import numpy as np
import os
import sys
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from pprint import pprint
import multiprocessing
from functools import wraps
from prettytable import PrettyTable
import re

from pt_train_MNIST import Exp_Classification
from visual import plot_predictions
from tools import (
    set_seed,
    change_dict_to_args,
    print_formatted_dict,
    select_best_metrics,
    suppress_print,
    extract_model_params_into_metrics,
    terminate_early_trial,
)


@extract_model_params_into_metrics
@suppress_print
@terminate_early_trial()
# @terminate_early_trial({"test_acc": -1})
def trainable(
    tunable_params: dict,  # Place tunable parameters first for Ray Tune
    fixed_params: dict,
    enable_ray_tune: bool = True,
) -> dict:
    # Set configs
    configs = change_dict_to_args({**fixed_params, **tunable_params})

    # Train the model
    exp = Exp_Classification(configs)
    metrics = exp.train()
    print_formatted_dict(metrics)

    # Plot predicitons
    if enable_ray_tune == False:
        print("---------------------------------------")
        print("Plotting predictions ...")
        plot_predictions(exp.model, exp.test_loader, enable_show=True)

    return select_best_metrics(metrics, target_mode="test", target_metric="acc")


def tunable(
    tunable_params: dict, fixed_params: dict, verbose: int = 1
) -> tune.ExperimentAnalysis:
    # * Setup GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = fixed_params["gpus"]
    max_concurrent_trials = len(fixed_params["gpus"].split(","))

    # * Set up metric and mode
    metric = "test_acc"
    mode = "max"

    # * Set up reporter
    metric_columns = ["time_total_s", "test_acc", "test_mf1", "test_kappa"]
    parameter_columns = [
        "model_name",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "epochs",
    ]
    reporter = CLIReporter(
        metric_columns=metric_columns,
        parameter_columns=parameter_columns,
        sort_by_metric=True,
        max_progress_rows=10,
        max_error_rows=1,
        max_column_length=200,
        # max_report_frequency=10,
    )

    # * Run
    output_path = str(Path(f"./ray_results").resolve())
    analysis = tune.run(
        tune.with_parameters(
            trainable, fixed_params=fixed_params, enable_ray_tune=True
        ),
        resources_per_trial={
            "cpu": fixed_params["num_workers"],
            "gpu": 1,
        },
        config=tunable_params,
        metric=metric,
        mode=mode,
        num_samples=fixed_params["num_trials"],
        max_concurrent_trials=max_concurrent_trials,
        progress_reporter=reporter,
        local_dir=output_path,
        verbose=verbose,
        raise_on_failed_trial=False,
    )

    # * Save analysis report as csv
    analysis_df = analysis.results_df
    analysis_df = analysis_df.reset_index()  # reset index to get trial_id
    analysis_df = analysis_df.drop(
        columns=[
            "experiment_id",
            "hostname",
            "node_ip",
            "time_this_iter_s",
            "time_since_restore",
            "timesteps_since_restore",
            "training_iteration",
            "timesteps_total",
            "date",
            "timestamp",
            "pid",
            "done",
            "episodes_total",
            "iterations_since_restore",
            "warmup_time",
            "experiment_tag",
        ]
    )
    analysis_df = analysis_df.sort_values(
        by=[metric], ascending=(True if mode == "min" else False)
    )
    experiment_name = Path(str(analysis.get_best_logdir())).parts[-2]
    analysis_df.to_csv(Path(output_path, experiment_name, "analysis.csv"), index=False)

    return analysis


def get_tunable_params(enable_ray_tune: bool = False) -> dict:
    if enable_ray_tune == False:
        tunable_params = {
            "model_name": "MLP",
            "model_params": {
                "MLP": {
                    "num_layers": 3,
                    "use_bn": True,
                },
                "CNN": {
                    "num_conv_layers": 3,
                    "use_bn": True,
                },
            },
            "batch_size": 256,
            "optim": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "epochs": 1,
            # "epochs": 3,
        }
    else:
        tunable_params = {
            "model_name": tune.choice(["MLP", "CNN"]),
            "model_params": {
                "MLP": {
                    "num_layers": tune.choice([2, 3, 4]),
                    "use_bn": tune.choice([True, False]),
                },
                "CNN": {
                    "num_conv_layers": tune.choice([1, 2, 3]),
                    "use_bn": tune.choice([True, False]),
                },
            },
            "batch_size": tune.choice([32, 64, 128, 256]),
            "optim": tune.choice(["Adam", "AdamW"]),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.uniform(0, 0.1),
            "epochs": tune.choice([1, 3, 5, 10]),
        }

    return tunable_params


if __name__ == "__main__":
    """-----------------------------------------------"""
    # enable_ray_tune = False
    enable_ray_tune = True

    num_trials = 10
    # num_trials = 100

    # gpus = "0"
    gpus = "0,1,2,3"

    # start_trial_id = 0
    start_trial_id = 8

    use_self_defined_params = False
    # use_self_defined_params = True  # use this to debug
    """-----------------------------------------------"""

    # Setup fixed params
    fixed_params = {
        "num_workers": 0,
        "use_tqdm": True,
        "use_amp": False,
        "checkpoint_dir": None,
        "num_trials": num_trials,
        "gpus": gpus,
        "start_trial_id": start_trial_id,
    }

    # Setup tunable params
    tunable_params = get_tunable_params(enable_ray_tune)
    if use_self_defined_params and enable_ray_tune == False:
        # TODO: add self-defined params here to debug
        # Remember to turn on use_self_defined_params and turn off enable_ray_tune
        tunable_params = {}

    # Set all random seeds (Python, NumPy, PyTorch)
    set_seed(42)

    # Run
    if enable_ray_tune == False:
        return_metrics = trainable(tunable_params, fixed_params, enable_ray_tune=False)
    else:
        analysis = tunable(tunable_params, fixed_params)

    print("### Done ###")
