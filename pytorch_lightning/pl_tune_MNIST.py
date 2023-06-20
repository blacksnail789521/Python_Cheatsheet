import torch

# import lightning as L
import pytorch_lightning as L  # 2.0.0
from datetime import datetime
from ray import air, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
import multiprocessing
import pandas as pd
import subprocess
from pathlib import Path

from pl_train_MNIST import trainable


def tune_models(
    tunable_params: dict,
    fixed_params: dict,
) -> pd.DataFrame:
    """
    1. Define the search space (tunable_params)
    2. Define the search algorithm (tune_config/search_alg) # Default is RandomSearch
    3. Define the scheduler (tune_config/scheduler) # Default is FIFO, but we want to use ASHA
    4. Define the number of trials (tune_config/num_samples)
    5. Define the metric (tune_config/metric)
    6. Define the mode (tune_config/mode)
    """

    # Define the directory to save the results
    default_root_dir = Path(
        "ray_results",
        fixed_params["model_name"],
        f"tune_models",
        datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    )

    # Pass in a Trainable class or function, along with a search space "config".
    tuner = tune.Tuner(
        trainable=tune.with_resources(
            tune.with_parameters(
                trainable,  # We will feed tunable_params to trainable
                fixed_params=fixed_params,
                ray_tune=True,
                use_lightning_data_module=True,
                data_dir=str(Path.cwd()),  # Avoid redownloading the data
            ),
            resources={
                "cpu": multiprocessing.cpu_count()
                // fixed_params["max_concurrent_trials"],
                "gpu": 1
                if fixed_params["use_gpu"]
                else 0,  # We can only use 1 GPU with ddp in Lightning 2.0.0
            },
        ),
        param_space=tunable_params,
        tune_config=tune.TuneConfig(
            num_samples=fixed_params["num_trials"],
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
            max_concurrent_trials=fixed_params[
                "max_concurrent_trials"
            ],  # default = 0 (unlimited)
        ),
        run_config=air.RunConfig(
            name=str(default_root_dir.name),
            verbose=2,  # default = 2
            local_dir=str(default_root_dir.parent),
        ),
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)

    # Kill all the subprocesses
    subprocess.run(["pkill", "-9", "-f", "ray::ImplicitFunc.train"])

    return results.get_dataframe()


if __name__ == "__main__":
    """-----------------------------------------------"""
    model_name = "MLP"
    # model_name = "CNN"

    # use_gpu = False
    use_gpu = True

    # num_trials = 100
    # num_trials = 20
    num_trials = 4

    max_concurrent_trials = 4
    # max_concurrent_trials = 1
    """-----------------------------------------------"""

    fixed_params = {
        "model_name": model_name,
        "loss": "cross_entropy",
        "metrics": ["cross_entropy", "accuracy"],
        # We must initialize the torchmetrics inside the model
        "use_gpu": use_gpu,  # if True, please use script to run the code
        # ------------------------------------------------------------
        # The above parameters are shared for both training and tuning
        # ------------------------------------------------------------
        "num_trials": num_trials,
        "max_concurrent_trials": max_concurrent_trials,
    }
    tunable_params = {
        # "batch_size": tune.choice([32, 64, 128, 256, 512]),
        "batch_size": tune.choice([512]),
        "optimizer": tune.choice(["Adam", "NAdam", "SGD"]),
        "lr": tune.choice([0.01, 0.001, 0.0001]),
        "l2_weight": tune.choice([0.01, 0.001, 0.0001]),
        "epochs": tune.choice([3, 5, 10]),
    }
    if fixed_params["model_name"] == "MLP":
        tunable_params["num_layers"] = tune.choice([2, 3, 4])
        tunable_params["test"] = tune.choice(["[1, 1]", "[2, 2]"])
        # Use ast.literal_eval to convert the string to a list
    elif fixed_params["model_name"] == "CNN":
        tunable_params["num_conv_layers"] = tune.choice([1, 2, 3])

    # Set all random seeds (Python, NumPy, PyTorch)
    L.seed_everything(seed=42, workers=True)

    # Set the precision of the matrix multiplication
    torch.set_float32_matmul_precision("high")

    # Tune the model
    results_df = tune_models(tunable_params, fixed_params)

    print("### Done ###")
