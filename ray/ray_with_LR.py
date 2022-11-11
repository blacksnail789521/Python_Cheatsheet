from ray import air, tune

# from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler


def loss_func(x, a, b):  # Define an objective function. (We want to minimize it.)
    return a * (x**0.5) + b


def train_LR(
    config,
):  # Pass a "config" dictionary into your trainable.

    # Emulate losses for 20 epoches (it should be decreasing)
    for x in range(20, 0, -1):
        loss = loss_func(x, config["a"], config["b"])

        yield {"loss": loss}  # Send the loss to Tune.


if __name__ == "__main__":

    # Pass in a Trainable class or function, along with a search space "config".
    tuner = tune.Tuner(
        trainable=tune.with_resources(train_LR, resources={"cpu": 1, "gpu": 0}),
        param_space={"a": tune.choice([2, 3, 4]), "b": tune.uniform(1, 10)},
        tune_config=tune.TuneConfig(
            num_samples=100,
            metric="loss",
            mode="min",
            # search_alg=OptunaSearch(sampler=TPESampler()),
            # scheduler=ASHAScheduler(),
        ),
        run_config=air.RunConfig(
            name="trial_test",
            verbose=2,
            stop=lambda trial_id, result: result["loss"] >= 20
            and result["training_iteration"] >= 5,
        ),
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)
    df = results.get_dataframe()
