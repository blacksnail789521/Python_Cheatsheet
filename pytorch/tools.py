import os
import sys
from functools import wraps
from pathlib import Path
import re
import random
import numpy as np
import torch
import argparse
from typing import Callable


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def change_dict_to_args(configs: dict) -> argparse.Namespace:
    args = argparse.Namespace()
    for key, value in configs.items():
        setattr(args, key, value)
    return args


def print_formatted_dict(d: dict) -> None:
    for key, value in d.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


def select_best_metrics(
    metrics: dict, target_mode: str = "test", target_metric: str = "acc"
) -> dict:
    best_metrics = {}

    # Find the epoch with the best target metric
    target_metric_values = metrics[target_mode][target_metric]
    best_epoch_index = target_metric_values.index(max(target_metric_values))
    best_metrics["best_epoch"] = best_epoch_index + 1  # Epoch starts from 1

    # Now gather metrics from all modes for this epoch
    for mode, mode_metrics in metrics.items():
        for metric_name, metric_values in mode_metrics.items():
            best_metrics[f"{mode}_{metric_name}"] = metric_values[best_epoch_index]

    return best_metrics


def suppress_print(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Don't need to execute this decorator if Ray Tune is not enabled
        enable_ray_tune = kwargs.get("enable_ray_tune", None)
        assert enable_ray_tune is not None, "enable_ray_tune should be specified"
        if not enable_ray_tune:
            return func(*args, **kwargs)

        # Disable printing
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

        result = func(*args, **kwargs)

        # Re-enable printing
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        return result

    return wrapper


def extract_model_params_into_metrics(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        return_metrics = func(*args, **kwargs)

        tunable_params = args[0]  # Assuming the first argument is always tunable_params

        def format_value(val):
            """Converts a float to a string with 3 decimal places."""
            if isinstance(val, float):
                return f"{val:.3f}"
            return val

        model_name = tunable_params["model_name"]
        params = tunable_params["model_params"].get(model_name, {})
        formatted_params_list = [f"{k}: {format_value(v)}" for k, v in params.items()]
        formatted_params_str = "\n".join(formatted_params_list)
        chosen_model_params = {"selected_model_params": formatted_params_str}
        return_metrics.update(chosen_model_params)

        return return_metrics

    return wrapper


def terminate_early_trial(return_value: dict = {"test_acc": 0}) -> Callable:
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Don't need to execute this decorator if Ray Tune is not enabled
            enable_ray_tune = kwargs.get("enable_ray_tune", None)
            assert enable_ray_tune is not None, "enable_ray_tune should be specified"
            if not enable_ray_tune:
                return func(*args, **kwargs)

            def extract_trial_id(working_dir):
                match = re.search(r"trainable_.{5}_([0-9]{5})_", working_dir)
                if match:
                    return int(match.group(1))
                else:
                    raise NotImplementedError

            current_dir = str(Path.cwd())
            trial_id = extract_trial_id(current_dir)

            fixed_params = kwargs.get("fixed_params", {})
            start_trial_id = fixed_params.get("start_trial_id", 0)
            if trial_id < start_trial_id:
                return return_value

            return func(*args, **kwargs)

        return wrapper
    return decorator