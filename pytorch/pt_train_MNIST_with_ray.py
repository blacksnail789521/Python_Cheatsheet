import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
from datetime import datetime
from ray import air, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
import multiprocessing

from pt_train_MNIST import trainable


if __name__ == "__main__":
    other_kwargs = {
        "loss": nn.CrossEntropyLoss(),
        "metrics": {
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10),
            "cross_entropy": nn.CrossEntropyLoss(),
        },
        "use_gpu": False,  # if True, please use script to run the code
    }
    param_space = {
        "batch_size": tune.choice([32, 64, 128, 256]),
        "optimizer": tune.choice(["Adam", "NAdam", "SGD"]),
        "lr": tune.choice([0.01, 0.001, 0.0001]),
        "num_layers": tune.choice([1, 2, 3, 4]),
        "l2_weight": tune.choice([0.01, 0.001, 0.0001]),
        "epochs": tune.choice([3, 5, 10]),
    }

    # Set all random seeds (Python, NumPy, PyTorch)
    pl.seed_everything(seed=0)

    """
    1. Define the search space (param_space)
    2. Define the search algorithm (tune_config/search_alg) # Default is RandomSearch
    3. Define the scheduler (tune_config/scheduler) # Default is FIFO, but we want to use ASHA
    4. Define the number of trials (tune_config/num_samples)
    5. Define the metric (tune_config/metric)
    6. Define the mode (tune_config/mode)
    """

    # Pass in a Trainable class or function, along with a search space "config".
    tuner = tune.Tuner(
        trainable=tune.with_resources(
            tune.with_parameters(trainable, other_kwargs=other_kwargs),
            resources={"cpu": multiprocessing.cpu_count(), "gpu": 0},
        ),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=5,
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
            max_concurrent_trials=1,  # default = 0 (unlimited)
        ),
        run_config=air.RunConfig(
            name=f"tune_MNIST_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            verbose=2,  # default = 2
            local_dir="./ray_results",
        ),
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)
    df = results.get_dataframe()
