from pathlib import Path
import os
from ray import tune
from ray.tune import CLIReporter
import argparse
from typing import Callable
import pandas as pd

from utils.tools import set_seed, print_formatted_dict
from utils.ray_tune_tools import (
    suppress_print,
    terminate_early_trial,
    get_experiment_trial_folder,
)
from main import get_args_from_parser
from main import trainable as trainable_without_ray_tune

# Use old reporting format
os.environ["RAY_AIR_NEW_OUTPUT"] = "0"


def get_tunable_params(enable_ray_tune: bool = False) -> dict:

    def choice(options, default):
        return tune.choice(options) if enable_ray_tune else default

    def loguniform(low, high, default):
        return tune.loguniform(low, high) if enable_ray_tune else default

    def uniform(low, high, default):
        return tune.uniform(low, high) if enable_ray_tune else default

    tunable_params = {
        "model_name": choice(["MLP", "CNN"], "MLP"),
        "model_params": {
            "MLP": {
                "num_layers": choice([1, 2, 3], 3),
                "use_bn": choice([True, False], True),
            },
            "CNN": {
                "num_conv_layers": choice([1, 2, 3], 3),
                "use_bn": choice([True, False], True),
            },
        },
        "optim": choice(["Adam", "AdamW"], "AdamW"),
        "learning_rate": loguniform(1e-4, 1e-1, 0.001),
        "weight_decay": uniform(0, 0.1, 0.01),
        "epochs": choice(
            [1],
            # [1, 3, 5, 10],
            1,
        ),
        "lr_scheduler": choice(
            [
                "StepLR",
                "ExponentialLR",
                "ReduceLROnPlateau",
                "CosineAnnealingLR",
                "CyclicLR",
                "OneCycleLR",
            ],
            "StepLR",
        ),
        "lr_scheduler_params": {
            "StepLR": {
                "step_size": choice([1, 3, 5], 1),
                "gamma": uniform(0.1, 0.9, 0.5),
            },
            "ExponentialLR": {
                "gamma": uniform(0.1, 0.9, 0.5),
            },
            "ReduceLROnPlateau": {
                "factor": uniform(0.1, 0.9, 0.5),
                "patience": choice([5, 10, 20], 10),
            },
            "CosineAnnealingLR": {
                "T_max": choice([1, 3, 5], 2),
            },
            "CyclicLR": {
                "max_lr": loguniform(1e-1, 1, 0.1),
                "step_size_up": choice([1, 3, 5], 3),
                "step_size_down": choice([1, 3, 5], 3),
            },
            "OneCycleLR": {
                "max_lr": loguniform(1e-1, 1, 0.1),
            },
        },
    }

    return tunable_params


@suppress_print
@terminate_early_trial()
# @terminate_early_trial({"test_acc": -1})
def trainable(
    tunable_params: dict,  # Place tunable parameters first for Ray Tune
    fixed_params: dict,
    args: argparse.Namespace,
    enable_ray_tune: bool = False,
    start_trial_id: int = 0,
) -> dict:
    # Update root_path
    experiment_folder, trial_folder = get_experiment_trial_folder()
    fixed_params["root_path"] = Path(
        fixed_params["root_path"], "ray_results", experiment_folder, trial_folder
    )

    # Run the main function
    return trainable_without_ray_tune(
        tunable_params,
        fixed_params,
        args,
        # enable_plot=False,
        enable_plot=True,
    )


def get_tune_main_metric_mode(fixed_params: dict) -> tuple[str, str, list, list]:
    # Set up metric and mode
    metric = "test_acc"
    mode = "max"

    # Set up metric_columns
    metric_columns = ["time_total_s", "test_acc", "test_mf1", "test_kappa"]

    # Set up parameter_columns
    parameter_columns = ["model_name", "learning_rate", "weight_decay", "epochs"]

    return metric, mode, metric_columns, parameter_columns


def tunable(
    tunable_params: dict,
    fixed_params: dict,
    args: argparse.Namespace,
    get_metric_mode: Callable,
    trainable: Callable,
) -> tuple[tune.ExperimentAnalysis, pd.DataFrame]:
    # * Setup GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = fixed_params["gpus"]
    max_concurrent_trials = len(fixed_params["gpus"].split(","))

    # * Set up metric, mode, and metric_columns
    metric, mode, metric_columns, parameter_columns = get_metric_mode(fixed_params)

    # * Set up reporter
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
            trainable, fixed_params=fixed_params, args=args, enable_ray_tune=True
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
        storage_path=output_path,
        verbose=1,
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
        ],
        errors="ignore",
    )
    analysis_df = analysis_df.sort_values(
        by=[metric], ascending=(True if mode == "min" else False)
    )
    analysis_df.to_csv(Path(analysis.experiment_path, "analysis.csv"), index=False)

    return analysis, analysis_df


if __name__ == "__main__":
    """-----------------------------------------------"""
    batch_size = 256

    num_workers = 4

    # enable_ray_tune = False
    enable_ray_tune = True

    num_trials = 4
    # num_trials = 10
    # num_trials = 100

    # gpus = "0"
    gpus = "0,1,2,3"

    start_trial_id = 0
    # start_trial_id = 50

    use_self_defined_params = False
    # use_self_defined_params = True  # use this to debug
    """-----------------------------------------------"""
    # Set all random seeds (Python, NumPy, PyTorch)
    set_seed(42)

    # Setup args
    args = get_args_from_parser()
    args.overwrite_args = True

    # Setup fixed params
    fixed_params = {
        "root_path": Path.cwd(),
        "batch_size": batch_size,
        "num_workers": num_workers,
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

    # Run
    if enable_ray_tune == False:
        return_metrics = trainable(
            tunable_params, fixed_params, args, enable_ray_tune=False
        )
        print_formatted_dict(return_metrics)
    else:
        analysis, analysis_df = tunable(
            tunable_params, fixed_params, args, get_tune_main_metric_mode, trainable
        )

    print("### Done ###")
