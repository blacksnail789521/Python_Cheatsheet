import torch
import torch.nn as nn
from torch.utils import data
import os
from tensorflow.keras.datasets import mnist
from torchvision import transforms
from datetime import datetime
from typing import List, Tuple, Union, Dict
import netron
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchmetrics
import numpy as np


class MNIST_Dataset(data.Dataset):
    def __init__(self, mode: str = "train") -> None:
        # Load numpy data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Load data based on mode
        if mode == "train":
            self.x = x_train
            self.y = y_train
        elif mode == "test":
            self.x = x_test
            self.y = y_test

        # One-hot encoding for y
        self.y = np.eye(10)[self.y]

        # Normalize
        self.x = self.x / 255.0

        # Add a dimension (for channel) (only for the images, a.k.a. x)
        # For pytorch, the channel dimension is the second dimension
        self.x = self.x[:, None, :, :]

        # Change to correct type
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def load_MNIST(
    batch_size: int = 256,
) -> Tuple[data.DataLoader, data.DataLoader]:

    # Get ds
    train_ds = MNIST_Dataset(mode="train")
    test_ds = MNIST_Dataset(mode="test")

    # Get loader
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_ds, batch_size=32, shuffle=False)

    return train_loader, test_loader


def show_data(test_loader: data.DataLoader) -> None:

    # Get the first batch
    x, y = next(iter(test_loader))
    x, y = x.numpy(), y.numpy()

    # Show the shape
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")

    # Show the first image and its label
    # (remember that the channel dimension is the second dimension)
    plt.imshow(x[0, 0, :, :], cmap="gray")
    plt.title(f"Label: {np.argmax(y[0])}")
    plt.show()


class DNN(pl.LightningModule):
    def __init__(
        self,
        num_layers: int = 2,
        l2_weight: float = 0.01,
        optimizer: str = "Adam",
        lr: float = 0.001,
        loss: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
        metrics: List[Dict[str, Union[torchmetrics.Metric, nn.modules.loss._Loss]]] = [
            {
                "accuracy": torchmetrics.Accuracy(),
                "cross_entropy": nn.CrossEntropyLoss(),
            }
        ],
    ) -> None:

        super(DNN, self).__init__()
        self.save_hyperparameters(
            ignore=["loss", "metrics"]
        )  # We can access the hyperparameters via self.hparams

        assert (
            self.hparams.num_layers >= 1
        ), "We should have at least one layer because the output layer is counted."

        # Define the model
        """
        self.dnn = nn.Sequential(
            nn.Flatten(),

            -----------------------------------------
            nn.Linear(28 * 28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            -----------------------------------------
            
            nn.Linear(128, 10),
            nn.Softmax(dim=1),
        )
        """

        self.layers = []
        self.layers.append(nn.Flatten())
        current_dim = 28 * 28
        for _ in range(self.hparams.num_layers - 1):
            self.layers.append(nn.Linear(current_dim, 128))
            self.layers.append(nn.BatchNorm1d(128))
            self.layers.append(nn.ReLU(inplace=True))
            current_dim = 128
        self.layers.append(nn.Linear(current_dim, 10))
        self.layers.append(nn.Softmax(dim=1))
        self.dnn = nn.Sequential(*self.layers)

        # Define loss and metrics
        self.loss = loss
        self.metrics = metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.dnn(x)

        return y

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:

        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2_weight
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        return [optimizer], [scheduler]

    def log_loss_and_metrics(
        self, mode: str, loss: torch.Tensor, y_pred: torch.Tensor, y: torch.Tensor
    ) -> None:

        if mode == "train":
            prefix = ""
        else:
            prefix = f"{mode}_"

        self.log(f"{prefix}loss", loss, prog_bar=True)
        for metric in self.metrics:
            metric_name = list(metric.keys())[0]
            if metric_name == "accuracy":
                # torchmetrics.Accuracy() can only take y in the shape of (N,)
                metric_value = metric[metric_name](y_pred, torch.argmax(y, dim=1))
            else:
                metric_value = metric[metric_name](y_pred, y)
            self.log(f"{prefix}{metric_name}", metric_value, prog_bar=True)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log_loss_and_metrics("train", loss, y_pred, y)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:

        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log_loss_and_metrics("val", loss, y_pred, y)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:

        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log_loss_and_metrics("test", loss, y_pred, y)

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        x, y = batch
        y_pred = self(x)

        return y_pred


def plot_model_with_netron(model: nn.Module, name: str = "DNN") -> None:

    # Save the model
    model_path = os.path.join("models", f"{name}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)  # Don't use .state_dict()

    # Plot the model
    netron.start(model_path, address=8081)


def train_model(
    train_loader: data.DataLoader,
    test_loader: data.DataLoader,
    model: pl.LightningModule,
    epochs: int = 3,
    additional_callbacks: List = [],
) -> pl.Trainer:

    # Set callbacks
    # (We don't need to set the tensorboard logger because it is set by default)
    early_stopping_with_TerminateOnNaN = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, mode="min", verbose=True, check_finite=True
    )
    callbacks = [early_stopping_with_TerminateOnNaN]
    callbacks.extend(additional_callbacks)

    # Train the model
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=None,  # default
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, test_loader)

    return trainer


def predict_with_model(
    model: pl.LightningModule,
    trainer: pl.Trainer,
    test_loader: data.DataLoader,
) -> None:

    # Get all the predictions (y_pred_list[0].shape: (32, 10))
    y_pred_list = trainer.predict(model, dataloaders=test_loader)
    y_pred = y_pred_list[0]  # Extract the first batch

    # Show the first 5 predictions
    x, y = next(iter(test_loader))
    for i in range(5):
        plt.imshow(x[i, 0, :, :], cmap="gray")
        plt.title(f"Label: {np.argmax(y[i])}, Prediction: {np.argmax(y_pred[i])}")
        plt.show()


if __name__ == "__main__":

    config = {
        "batch_size": 256,
        "optimizer": "Adam",
        "lr": 0.001,
        "num_layers": 2,
        "l2_weight": 0.01,
        "epochs": 3,
    }
    other_kwargs = {
        "loss": nn.CrossEntropyLoss(),
        "metrics": [
            {
                "accuracy": torchmetrics.Accuracy(),
                "cross_entropy": nn.CrossEntropyLoss(),
            }
        ],
    }

    # Set all raodom seeds (Python, NumPy, PyTorch)
    pl.seed_everything(seed=0)

    # Load data
    train_loader, test_loader = load_MNIST(batch_size=config["batch_size"])

    # Show the data
    show_data(test_loader)

    # Get the model
    model = DNN(
        num_layers=config["num_layers"],
        l2_weight=config["l2_weight"],
        optimizer=config["optimizer"],
        lr=config["lr"],
        loss=other_kwargs["loss"],
        metrics=other_kwargs["metrics"],
    )
    print(model)

    # Plot the model
    # plot_model_with_netron(model)

    # Train
    print("---------------------------------------")
    print("Training ...")
    trainer = train_model(train_loader, test_loader, model, epochs=config["epochs"])

    # Evaluate
    print("---------------------------------------")
    print("Evaluating ...")
    # The length of the loss_list corresponds to the number of dataloaders used.
    loss_list = trainer.test(dataloaders=test_loader)

    # Predict
    print("---------------------------------------")
    print("Predicting ...")
    predict_with_model(model, trainer, test_loader)
