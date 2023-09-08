import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import random
from tqdm import tqdm
import numpy as np
import os
import sys
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from pprint import pprint
import multiprocessing
from functools import wraps
from prettytable import PrettyTable
import re

from pt_load_MNIST import load_MNIST, show_data
from models.MLP import MLP
from models.CNN import CNN

MODEL_MAP = {
    "MLP": MLP,
    "CNN": CNN,
}


def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    checkpoint_dir: Path | None = Path("checkpoints"),
    epochs: int = 3,
    lr: float = 0.001,
    weight_decay: float = 0.0,
) -> tuple[nn.Module, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # # Load checkpoint if it exists
    # if checkpoint_dir:
    #     checkpoint_dir.mkdir(parents=True, exist_ok=True)
    #     if Path(checkpoint_dir, "checkpoint.pth").exists():
    #         model_state, optimizer_state = torch.load(
    #             Path(checkpoint_dir, "checkpoint.pth")
    #         )
    #         model.load_state_dict(model_state)
    #         optimizer.load_state_dict(optimizer_state)

    # Train the model
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    for epoch in range(epochs):
        model.train()
        train_losses = []
        val_losses = []
        test_losses = []

        progress = tqdm(
            train_dl, desc=f"Epoch {epoch + 1}/{epochs}, Training Loss: {0}"
        )
        for inputs, targets in progress:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            progress.set_description(
                f"Epoch {epoch + 1}/{epochs}, Training Loss: {np.mean(train_losses)}"
            )

        # Validate the model (both on validation and test sets)
        model.eval()
        with torch.no_grad():
            correct_val = 0
            total_val = 0
            for inputs, targets in val_dl:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

            correct_test = 0
            total_test = 0
            for inputs, targets in test_dl:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()

        # Print the results
        print(
            f"Epoch {epoch+1}/{epochs}, Validation Loss: {np.mean(val_losses)}, "
            f"Validation Acc: {correct_val / total_val}, "
            f"Test Loss: {np.mean(test_losses)}, "
            f"Test Acc: {correct_test / total_test}"
        )

        # Save the model and optimizer
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir, "checkpoint.pth")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save((model.state_dict(), optimizer.state_dict()), checkpoint_path)

        # Save the metrics
        metrics["train_loss"].append(np.mean(train_losses))
        metrics["val_loss"].append(np.mean(val_losses))
        metrics["val_acc"].append(correct_val / total_val)
        metrics["test_loss"].append(np.mean(test_losses))
        metrics["test_acc"].append(correct_test / total_test)

    # Use the last epoch's metrics
    for key, value in metrics.items():
        metrics[key] = value[-1]

    return model, metrics


def plot_predictions(
    model: nn.Module,
    test_dl: DataLoader,
    enable_show: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Show the first 5 predictions
    x, y = next(iter(test_dl))
    for i in range(5):
        plt.imshow(x[i, 0, :, :], cmap="gray")
        y_pred = model(x[i].unsqueeze(0).to(device))
        plt.title(f"Label: {y[i]}, Prediction: {torch.argmax(y_pred)}")

        # Save the figure
        plot_folder = Path("plots")
        plot_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_folder / f"prediction_{i}.png")
        if enable_show:
            plt.show()
        plt.close()


def get_model(tunable_params: dict) -> nn.Module:
    model_name = tunable_params["model_name"]
    model_params = tunable_params["model_params"][model_name]

    # Dynamically get the model class based on the model name
    ModelClass = MODEL_MAP.get(model_name, None)
    if ModelClass is None:
        raise ValueError(f"Unknown model name: {model_name}")

    model = ModelClass(**model_params)
    return model


def suppress_print(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Don't need to execute this decorator if Ray Tune is not enabled
        enable_ray_tune = kwargs.get("enable_ray_tune", True)
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


def extract_model_params_into_metrics(func):
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


def terminate_early_trial(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Don't need to execute this decorator if Ray Tune is not enabled
        enable_ray_tune = kwargs.get("enable_ray_tune", True)
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
            return {"test_acc": 0.0}

        return func(*args, **kwargs)

    return wrapper


@extract_model_params_into_metrics
@suppress_print
@terminate_early_trial
def trainable(
    tunable_params: dict,  # Place tunable parameters first for Ray Tune
    fixed_params: dict,
    enable_ray_tune: bool = True,
    start_trial_id: int = 0,
) -> dict[str, float]:
    # Load data
    train_dl, test_dl = load_MNIST(
        batch_size=tunable_params["batch_size"],
        max_concurrent_trials=fixed_params.get("max_concurrent_trials", 1),
    )
    val_dl = test_dl
    if not enable_ray_tune:
        show_data(train_dl)  # Show the data

    # Get the model
    model = get_model(tunable_params)
    if not enable_ray_tune:
        print(model)

    # Train
    print("---------------------------------------")
    print("Training ...")
    model, return_metrics = train_model(
        model,
        train_dl,
        val_dl,
        test_dl,
        epochs=tunable_params["epochs"],
        lr=tunable_params["lr"],
    )

    # Plot predicitons
    print("---------------------------------------")
    print("Plotting predictions ...")
    plot_predictions(model, test_dl)

    return return_metrics


def tunable(tunable_params: dict, fixed_params: dict, verbose: int = 1) -> None:
    # Set up reporter
    metric_columns = {
        "selected_model_params": "model_params",
        "time_total_s": "time (s)",
        "train_loss": "train_loss",
        "val_loss": "val_loss",
        "val_acc": "val_acc",
        "test_loss": "test_loss",
        "test_acc": "test_acc",
    }
    parameter_columns = ["model_name", "batch_size", "lr", "weight_decay", "epochs"]
    reporter = CLIReporter(
        metric_columns=metric_columns,
        parameter_columns=parameter_columns,
        sort_by_metric=True,
        max_progress_rows=10,
        max_error_rows=3,
        max_column_length=200,
        # max_report_frequency=10,
    )

    analysis = tune.run(
        tune.with_parameters(
            trainable, fixed_params=fixed_params, enable_ray_tune=True
        ),
        resources_per_trial={
            "cpu": multiprocessing.cpu_count() // fixed_params["max_concurrent_trials"],
            "gpu": 1,
        },
        config=tunable_params,
        metric="test_acc",
        mode="max",
        num_samples=fixed_params["num_trials"],
        max_concurrent_trials=fixed_params["max_concurrent_trials"],
        progress_reporter=reporter,
        local_dir="./ray_results",
        verbose=verbose,
    )


def get_tunable_params(enable_ray_tune: bool = False) -> dict:
    if enable_ray_tune == False:
        tunable_params = {
            "model_name": "MLP",
            "model_params": {
                "MLP": {
                    "num_layers": 3,
                    "use_bn": True,
                },
                "CNN": {
                    "num_conv_layers": 3,
                    "use_bn": True,
                },
            },
            "batch_size": 256,
            "lr": 0.001,
            "weight_decay": 0.01,
            "epochs": 3,
        }
    else:
        tunable_params = {
            "model_name": tune.choice(["MLP", "CNN"]),
            "model_params": {
                "MLP": {
                    "num_layers": tune.choice([2, 3, 4]),
                    "use_bn": tune.choice([True, False]),
                },
                "CNN": {
                    "num_conv_layers": tune.choice([1, 2, 3]),
                    "use_bn": tune.choice([True, False]),
                },
            },
            "batch_size": tune.choice([32, 64, 128, 256]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "weight_decay": tune.uniform(0, 0.1),
            "epochs": tune.choice([1, 3, 5, 10]),
        }

    return tunable_params


if __name__ == "__main__":
    """-----------------------------------------------"""
    # enable_ray_tune = False
    enable_ray_tune = True

    num_trials = 6
    # num_trials = 100

    # max_concurrent_trials = 1
    max_concurrent_trials = 4

    # start_trial_id = 0
    start_trial_id = 2
    """-----------------------------------------------"""

    # Set all random seeds (Python, NumPy, PyTorch)
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    fixed_params = {
        "num_trials": num_trials,
        "max_concurrent_trials": max_concurrent_trials,
        "start_trial_id": start_trial_id,
    }
    tunable_params = get_tunable_params(enable_ray_tune)

    if enable_ray_tune == False:
        return_metrics = trainable(tunable_params, fixed_params, enable_ray_tune=False)
        table = PrettyTable(["Metric", "Value"])
        [
            table.add_row([k, v])
            for k, v in return_metrics.items()
            if k != "selected_model_params"
        ]
        print(table)
    else:
        tunable(tunable_params, fixed_params)

    print("### Done ###")
