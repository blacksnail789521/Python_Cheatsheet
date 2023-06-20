import numpy as np
import torch
import random
import multiprocessing
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from pt_train_MNIST import trainable


def tune_models(
    tunable_params: dict,
    fixed_params: dict,
    num_trials: int = 10,
    max_concurrent_trials: int = 1,
) -> None:
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        # max_t=100, # (default) max epochs
        # grace_period=1, # (default) min epochs
    )

    reporter = CLIReporter(
        metric_columns=[
            "train_loss",
            "val_loss",
            "val_acc",
            "test_loss",
            "test_acc",
            "training_iteration",
        ]
    )

    result = tune.run(
        tune.with_parameters(trainable, fixed_params=fixed_params),
        resources_per_trial={
            "cpu": multiprocessing.cpu_count() // max_concurrent_trials,
            "gpu": 1,
        },
        config=tunable_params,
        num_samples=num_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=f"./ray_results/{fixed_params['model_name']}",
    )

    best_trial = result.get_best_trial("val_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print(f"Best trial result: {best_trial.last_result}")


if __name__ == "__main__":
    """-----------------------------------------------"""
    model_name = "MLP"
    # model_name = "CNN"

    # num_trials = 10
    num_trials = 3

    # max_concurrent_trials = 4
    max_concurrent_trials = 1
    """-----------------------------------------------"""

    fixed_params = {
        "model_name": model_name,
    }
    tunable_params = {
        "batch_size": tune.choice([32, 64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.uniform(0, 0.1),
        "epochs": tune.choice([1, 3, 5, 10]),
    }
    if fixed_params["model_name"] == "MLP":
        tunable_params["num_layers"] = tune.choice([2, 3, 4])
    elif fixed_params["model_name"] == "CNN":
        tunable_params["num_conv_layers"] = tune.choice([1, 2, 3])

    # Set all random seeds (Python, NumPy, PyTorch)
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set the precision of the matrix multiplication
    torch.set_float32_matmul_precision("high")

    # Train the model
    tune_models(
        tunable_params,
        fixed_params,
        num_trials=num_trials,
        max_concurrent_trials=max_concurrent_trials,
    )

    print("### Done ###")
