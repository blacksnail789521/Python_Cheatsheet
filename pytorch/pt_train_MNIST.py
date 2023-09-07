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

from pt_load_MNIST import load_MNIST, show_data
from models.MLP import MLP
from models.CNN import CNN


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

    # Load checkpoint if it exists
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if Path(checkpoint_dir, "checkpoint.pth").exists():
            model_state, optimizer_state = torch.load(
                Path(checkpoint_dir, "checkpoint.pth")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

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


def get_model(tunable_params: dict, fixed_params: dict) -> nn.Module:
    if fixed_params["model_name"] == "MLP":
        model = MLP(tunable_params["num_layers"])
    elif fixed_params["model_name"] == "CNN":
        model = CNN(tunable_params["num_conv_layers"])
    else:
        raise ValueError(f"Unknown model name: {fixed_params['model_name']}")

    return model


def trainable(
    tunable_params: dict,  # Place tunable parameters first for Ray Tune
    fixed_params: dict,
    enable_ray_tune: bool = True,
) -> dict[str, float]:
    # Disable print
    if enable_ray_tune:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    # Load data
    train_dl, test_dl = load_MNIST(
        batch_size=tunable_params["batch_size"],
        max_concurrent_trials=fixed_params.get("max_concurrent_trials", 1),
    )
    val_dl = test_dl
    if not enable_ray_tune:
        show_data(train_dl)  # Show the data

    # Get the model
    model = get_model(tunable_params, fixed_params)
    if not enable_ray_tune:
        print(model)

    # Train
    print("---------------------------------------")
    print("Training ...")
    model, metrics = train_model(
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

    # Re-enable print
    if enable_ray_tune:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout  # type: ignore
        sys.stderr = original_stderr  # type: ignore

    return metrics


if __name__ == "__main__":
    """-----------------------------------------------"""
    model_name = "MLP"
    # model_name = "CNN"

    epochs = 3
    # epochs = 1

    # verbose = False
    verbose = True
    """-----------------------------------------------"""

    fixed_params = {
        "model_name": model_name,
    }
    tunable_params = {
        "batch_size": 256,
        "lr": 0.001,
        "weight_decay": 0.01,
        "epochs": epochs,
    }
    if fixed_params["model_name"] == "MLP":
        tunable_params["num_layers"] = 3
    elif fixed_params["model_name"] == "CNN":
        tunable_params["num_conv_layers"] = 3

    # Set all random seeds (Python, NumPy, PyTorch)
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set the precision of the matrix multiplication
    torch.set_float32_matmul_precision("high")

    # Train the model
    metrics = trainable(tunable_params, fixed_params, enable_ray_tune=False)
    print(metrics)

    print("### Done ###")
