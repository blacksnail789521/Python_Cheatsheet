import os
import sys
from functools import wraps
from pathlib import Path
import re
from typing import Callable, Any
from ray import train, tune
import numpy as np
import threading


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


def terminate_early_trial(default_return_metrics: dict = {"test_acc": 0}) -> Callable:
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Don't need to execute this decorator if Ray Tune is not enabled
            enable_ray_tune = kwargs.get("enable_ray_tune", None)
            assert enable_ray_tune is not None, "enable_ray_tune should be specified"
            if not enable_ray_tune:
                return func(*args, **kwargs)

            # Extract the start_trial_id and default_return_metrics from fixed_params
            fixed_params = kwargs.get("fixed_params", {})
            start_trial_id = fixed_params.get("start_trial_id", 0)
            return_metrics = fixed_params.get(
                "default_return_metrics", default_return_metrics
            )

            def extract_trial_id(working_dir):
                match = re.search(r"trainable_.{5}_([0-9]{5})_", working_dir)
                if match:
                    return int(match.group(1))
                else:
                    raise NotImplementedError

            current_dir = str(Path.cwd())
            trial_id = extract_trial_id(current_dir)

            if trial_id < start_trial_id:
                return return_metrics

            return func(*args, **kwargs)

        return wrapper

    return decorator


def timeout_decorator(
    max_runtime_s: int = 1000, default_return_metrics: dict = {"test_acc": 0}
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract the max_runtime_s and default_return_metrics from fixed_params
            fixed_params = kwargs.get("fixed_params", {})
            timeout_seconds = fixed_params.get("max_runtime_s", max_runtime_s)
            return_metrics = fixed_params.get(
                "default_return_metrics", default_return_metrics
            )

            result = [return_metrics]  # Store the result in a mutable container

            def target():
                result[0] = func(*args, **kwargs)

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                # If the thread is still alive after the timeout, it means the function timed out
                thread.join()  # Ensure the thread is cleaned up properly
                return return_metrics

            return result[0]

        return wrapper

    return decorator


def get_experiment_trial_folder() -> tuple[str, str]:

    trial_path = Path(train.get_context().get_trial_dir())
    trial_folder = trial_path.name
    experiment_folder = trial_path.parent.parent.name

    return experiment_folder, trial_folder


def convert_np_to_native(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()  # Convert to native Python int or float
    return value  # Return unchanged if it's not a NumPy type


def create_tune_function(enable_ray_tune: bool):
    def choice(options: list[Any], default: Any, sample_once: bool = False) -> Any:
        if enable_ray_tune:
            if sample_once:
                sampled_value = np.random.choice(options)
                return tune.sample_from(
                    lambda spec: convert_np_to_native(sampled_value)
                )
            else:
                return tune.choice(options)
        else:
            return default

    def loguniform(
        low: float, high: float, default: float, sample_once: bool = False
    ) -> Any:
        if enable_ray_tune:
            if sample_once:
                sampled_value = np.exp(np.random.uniform(np.log(low), np.log(high)))
                return tune.sample_from(
                    lambda spec: convert_np_to_native(sampled_value)
                )
            else:
                return tune.loguniform(low, high)
        else:
            return default

    def uniform(
        low: float, high: float, default: float, sample_once: bool = False
    ) -> Any:
        if enable_ray_tune:
            if sample_once:
                sampled_value = np.random.uniform(low, high)
                return tune.sample_from(
                    lambda spec: convert_np_to_native(sampled_value)
                )
            else:
                return tune.uniform(low, high)
        else:
            return default

    return choice, loguniform, uniform
