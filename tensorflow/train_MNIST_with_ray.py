from datetime import datetime
from typing import Dict
from ray import air, tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCheckpointCallback
from train_MNIST import load_MNIST, DNN, train_model


def trainable(config: Dict):

    # Load data
    train_ds, test_ds = load_MNIST(batch_size=config["batch_size"])

    # Get the model
    model = DNN(
        num_layers=config["num_layers"],
        l2_weight=config["l2_weight"],
        optimizer=config["optimizer"],
    )

    # Train
    train_model(
        train_ds,
        test_ds,
        model,
        epochs=config["epochs"],
        additional_callbacks=[
            TuneReportCheckpointCallback(
                metrics={"loss": "val_loss", "accuracy": "val_accuracy"},
                # filename="checkpoint", # (default)
            )
        ],
    )


if __name__ == "__main__":

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
        trainable=tune.with_resources(trainable, resources={"cpu": 1, "gpu": 0}),
        param_space={
            "batch_size": tune.choice([32, 64, 128, 256]),
            "optimizer": tune.choice(["adam", "Nadam", "sgd"]),
            "num_layers": tune.choice([1, 2, 3, 4]),
            "l2_weight": tune.choice([0.01, 0.001, 0.0001]),
            "epochs": tune.choice([3, 5, 10]),
            # "epochs": tune.choice([1]),
        },
        tune_config=tune.TuneConfig(
            num_samples=1,
            # metric="loss",
            # mode="min",
            metric="accuracy",
            mode="max",
            # search_alg=OptunaSearch(sampler=TPESampler()),
            scheduler=AsyncHyperBandScheduler(  # Same as ASHAScheduler
                time_attr="training_iteration",  # (default)
                # max_t=100, # (default) max epochs
                # grace_period=1, # (default) min epochs
            ),
            max_concurrent_trials=2,  # default = 0 (unlimited)
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
