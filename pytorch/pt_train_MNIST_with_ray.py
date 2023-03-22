import torch

# import lightning as L
import pytorch_lightning as L  # 2.0.0
from datetime import datetime
from ray import air, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
import multiprocessing
import pandas as pd
import os
import subprocess

from pt_train_MNIST import trainable


def tune_models(
    fixed_params: dict,
    tunable_params: dict,
    num_trials: int = 10,
    max_concurrent_trials: int = 1,
) -> pd.DataFrame:
    """
    1. Define the search space (tunable_params)
    2. Define the search algorithm (tune_config/search_alg) # Default is RandomSearch
    3. Define the scheduler (tune_config/scheduler) # Default is FIFO, but we want to use ASHA
    4. Define the number of trials (tune_config/num_samples)
    5. Define the metric (tune_config/metric)
    6. Define the mode (tune_config/mode)
    """

    # Pass in a Trainable class or function, along with a search space "config".
    tuner = tune.Tuner(
        trainable=tune.with_resources(
            tune.with_parameters(
                trainable,  # We will feed tunable_params to trainable
                fixed_params=fixed_params,
                ray_tune=True,
                use_lightning_data_module=True,
                data_dir=os.getcwd(),  # Avoid redownloading the data
            ),
            resources={
                "cpu": multiprocessing.cpu_count() // max_concurrent_trials,
                "gpu": 1
                if fixed_params["use_gpu"]
                else 0,  # We can only use 1 GPU with ddp in Lightning 2.0.0
            },
        ),
        param_space=tunable_params,
        tune_config=tune.TuneConfig(
            num_samples=num_trials,
            # metric="val_loss",
            # mode="min",
            metric="val_accuracy",
            mode="max",
            # search_alg=OptunaSearch(sampler=TPESampler()),
            scheduler=AsyncHyperBandScheduler(  # Same as ASHAScheduler
                time_attr="training_iteration",  # (default)
                # max_t=100, # (default) max epochs
                # grace_period=1, # (default) min epochs
            ),
            max_concurrent_trials=max_concurrent_trials,  # default = 0 (unlimited)
        ),
        run_config=air.RunConfig(
            name=f"tune_MNIST_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            verbose=2,  # default = 2
            local_dir="./ray_results",
        ),
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)

    # If we are running multiple trials, we need to kill the processes
    if max_concurrent_trials > 1:
        subprocess.run(["pkill", "-9", "-f", "ray::ImplicitFunc.train"])

    return results.get_dataframe()


if __name__ == "__main__":
    fixed_params = {
        "loss": "cross_entropy",
        "metrics": ["cross_entropy", "accuracy"],
        # We must initialize the torchmetrics inside the model
        "use_gpu": True,  # if True, please use script to run the code
    }
    tunable_params = {
        # "batch_size": tune.choice([32, 64, 128, 256]),
        "batch_size": tune.choice([512]),
        "optimizer": tune.choice(["Adam", "NAdam", "SGD"]),
        "lr": tune.choice([0.01, 0.001, 0.0001]),
        "num_layers": tune.choice([1, 2, 3, 4]),
        "l2_weight": tune.choice([0.01, 0.001, 0.0001]),
        "epochs": tune.choice([3, 5, 10]),
    }

    # Set all random seeds (Python, NumPy, PyTorch)
    L.seed_everything(seed=0)

    # Set the precision of the matrix multiplication
    torch.set_float32_matmul_precision("high")

    # Tune the model
    results_df = tune_models(
        fixed_params, tunable_params, num_trials=16, max_concurrent_trials=4
    )
