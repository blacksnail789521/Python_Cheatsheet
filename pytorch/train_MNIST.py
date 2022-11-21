import torch
import torch.nn as nn
from torch.utils import data
import os
from tensorflow.keras.datasets import mnist
from torchvision import transforms
from datetime import datetime
from typing import List, Tuple, Union
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
        self.mode = mode
        if self.mode == "train":
            self.x = x_train
            self.y = y_train
        elif self.mode == "test":
            self.x = x_test
            self.y = y_test
        elif self.mode == "predict":
            self.x = x_test

        # Add a dimension (for channel) (only for the images, a.k.a. x)
        # For pytorch, the channel dimension is the second dimension
        self.x = self.x[:, None, :, :]

        # Change the image type to float32
        self.x = self.x.astype(np.float32)

        # Normalize
        self.x = self.x / 255.0

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "train" or self.mode == "test":
            return self.x[idx], self.y[idx]
        elif self.mode == "predict":
            return self.x[idx]


def load_MNIST(
    batch_size: int = 256,
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:

    # Get ds
    train_ds = MNIST_Dataset(mode="train")
    test_ds = MNIST_Dataset(mode="test")
    predict_ds = MNIST_Dataset(mode="predict")

    # Get loader
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_ds, batch_size=32, shuffle=False)
    predict_loader = data.DataLoader(predict_ds, batch_size=32, shuffle=False)

    return train_loader, test_loader, predict_loader


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
    plt.title(f"Label: {y[0]}")
    plt.show()


class DNN(pl.LightningModule):
    def __init__(
        self,
        num_layers: int = 2,
        l2_weight: float = 0.01,
        optimizer: str = "Adam",
    ) -> None:

        super(DNN, self).__init__()
        assert (
            num_layers >= 1
        ), "We should have at least one layer because the output layer is counted."
        self.save_hyperparameters()  # We can access the hyperparameters via self.hparams

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
        )
        """

        self.layers = []
        self.layers.append(nn.Flatten())
        for _ in range(self.hparams.num_layers - 1):
            self.layers.append(nn.Linear(28 * 28, 128))
            self.layers.append(nn.BatchNorm1d(128))
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(128, 10))
        self.dnn = nn.Sequential(*self.layers)

        # Define loss and metrics
        self.loss = nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.dnn(x)

        return y

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:

        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), lr=1e-3, weight_decay=self.hparams.l2_weight
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        return [optimizer], [scheduler]

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.acc(y_pred, y), prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:

        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.acc(y_pred, y), prog_bar=True)


def plot_model_with_netron(model: nn.Module, name: str = "DNN") -> None:

    # Save the model
    model_path = os.path.join("models", f"{name}.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)  # Don't use .state_dict()

    # Plot the model
    netron.start(model_path, address=8080)


def train_model(
    train_loader: data.DataLoader,
    test_loader: data.DataLoader,
    model: pl.LightningModule,
    epochs: int = 3,
    additional_callbacks: List = [],
) -> pl.Trainer:

    # # Set callbacks
    # callbacks = [
    #     pl.callbacks.ModelCheckpoint(
    #         monitor="val_acc",
    #         dirpath="models",
    #         filename="DNN-{epoch:02d}-{val_acc:.2f}",
    #         save_top_k=1,
    #         mode="max",
    #     ),
    #     pl.callbacks.EarlyStopping(monitor="val_acc", patience=3, mode="max"),
    # ]
    # callbacks.extend(additional_callbacks)

    # Train the model
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=None,  # default
        # callbacks=callbacks,
    )
    trainer.fit(model, train_loader, test_loader)

    return trainer


def predict_with_model(
    model: pl.LightningModule,
    trainer: pl.Trainer,
    predict_loader: data.DataLoader,
    test_loader: data.DataLoader,
) -> None:

    # Get all the predictions (y_pred_list[0].shape: (32, 10))
    # (We must use the predict_loader because it's the one without labels)
    y_pred_list = trainer.predict(model, dataloaders=predict_loader)
    y_pred = y_pred_list[0]  # Extract the first batch

    # Show the first 5 predictions
    x, y = next(iter(test_loader))
    for i in range(5):
        plt.imshow(x[i, 0, :, :], cmap="gray")
        plt.title(f"Label: {y[i]}, Prediction: {np.argmax(y_pred[i])}")
        plt.show()


if __name__ == "__main__":

    config = {
        "batch_size": 256,
        "optimizer": "Adam",
        "num_layers": 2,
        "l2_weight": 0.01,
        "epochs": 1,
    }

    # Load data
    train_loader, test_loader, predict_loader = load_MNIST(
        batch_size=config["batch_size"]
    )

    # Show the data
    show_data(test_loader)

    # Get the model
    model = DNN(
        num_layers=config["num_layers"],
        l2_weight=config["l2_weight"],
        optimizer=config["optimizer"],
    )
    print(model)

    # plot_model_with_netron(model)

    # Train
    print("---------------------------------------")
    print("Training ...")
    trainer = train_model(train_loader, test_loader, model, epochs=config["epochs"])

    # Predict
    print("---------------------------------------")
    print("Predicting...")
    predict_with_model(model, trainer, predict_loader, test_loader)
