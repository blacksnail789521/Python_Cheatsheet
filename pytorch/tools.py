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
