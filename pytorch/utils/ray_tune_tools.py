import os
import sys
from functools import wraps
from pathlib import Path
import re
from typing import Callable, Any
from ray import train, tune
import numpy as np
import multiprocessing
import json
import random
import math


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


def extract_model_kwargs_into_metrics(func: Callable) -> Callable:
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
        params = tunable_params["model_kwargs"].get(model_name, {})
        formatted_params_list = [f"{k}: {format_value(v)}" for k, v in params.items()]
        formatted_params_str = "\n".join(formatted_params_list)
        chosen_model_kwargs = {"selected_model_kwargs": formatted_params_str}
        return_metrics.update(chosen_model_kwargs)

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
                match = re.search(r"trainable_.{5}_([0-9]{5})", working_dir)
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
    max_runtime_s: int | None = None, default_return_metrics: dict = {"test_acc": 0}
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract the max_runtime_s from kwargs, if it exists; otherwise, use the default
            fixed_params = kwargs.get("fixed_params", {})
            timeout_seconds = fixed_params.get("max_runtime_s", max_runtime_s)
            return_metric = fixed_params.get(
                "default_return_metrics", default_return_metrics
            )

            if timeout_seconds is None:
                # No timeout specified, execute the function normally
                return func(*args, **kwargs)

            result = multiprocessing.Manager().list([return_metric])
            exception = multiprocessing.Queue()  # Use a queue to pass exceptions

            def target(result_container, exception_queue):
                try:
                    result_container[0] = func(*args, **kwargs)
                except Exception as e:
                    exception_queue.put(e)

            process = multiprocessing.Process(target=target, args=(result, exception))
            process.start()
            process.join(timeout=timeout_seconds)

            if process.is_alive():
                # If the process is still alive after the timeout, it means the function timed out
                process.terminate()  # Forcefully terminate the process
                process.join()  # Ensure the process is cleaned up properly
                return return_metric

            # If there was an exception in the process, raise it here
            if not exception.empty():
                raise exception.get()

            return result[0]

        return wrapper

    return decorator


def get_experiment_trial_folder() -> tuple[str, str]:

    trial_path = Path(train.get_context().get_trial_dir())
    trial_folder = trial_path.name
    experiment_folder = trial_path.parent.parent.name

    return experiment_folder, trial_folder


def create_tune_function(
    enable_ray_tune: bool, tunable_params: dict
) -> tuple[Callable, Callable, Callable, Callable, Callable]:
    def choice(options: list[Any], default: Any) -> Any:
        return default if not enable_ray_tune else tune.choice(options)

    def loguniform(bounds: tuple[float, float], default: float) -> Any:
        low, high = bounds
        return default if not enable_ray_tune else tune.loguniform(low, high)

    def uniform(bounds: tuple[float, float], default: float) -> Any:
        low, high = bounds
        return default if not enable_ray_tune else tune.uniform(low, high)

    def sample_from(func: Callable, default: Any) -> Any:
        return default if not enable_ray_tune else tune.sample_from(func)

    def copy_param(key: str) -> Any:
        if enable_ray_tune:
            return tune.sample_from(lambda config: config[key])
        else:
            return tunable_params[key]

    return choice, loguniform, uniform, sample_from, copy_param


class PythonEncoder(json.JSONEncoder):
    def iterencode(self, obj, _one_shot=False):
        # Encode the object using the parent class method
        json_iter = super().iterencode(obj, _one_shot)
        # Replace the JSON-specific values with Python equivalents as we iterate
        for chunk in json_iter:
            yield chunk.replace("true", "True").replace("false", "False").replace(
                "null", "None"
            )


def store_custom_tunable_params(tunable_params: dict, output_root_path: Path):
    # Store custom_tunable_params with corresponding model kwargs
    model_name = tunable_params.get("model_name")
    model_kwargs = tunable_params.get("model_kwargs", {}).get(model_name, {})
    custom_tunable_params = {
        key: value
        for key, value in tunable_params.items()
        if key != "model_kwargs"  # Exclude the full model_kwargs from the copy
    }
    custom_tunable_params["model_kwargs"] = {model_name: model_kwargs}

    # Save custom_tunable_params to a txt file with Python-compatible values
    with open(Path(output_root_path, "custom_tunable_params.txt"), "w") as f:
        json.dump(custom_tunable_params, f, cls=PythonEncoder, indent=4)
