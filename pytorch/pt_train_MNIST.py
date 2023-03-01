import torch
import torch.nn as nn
from torch.utils import data
import os
import netron
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchmetrics
import numpy as np
from datetime import datetime
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from pt_load_MNIST import load_MNIST, show_data


class DNN(pl.LightningModule):
    def __init__(
        self,
        num_layers: int = 2,
        l2_weight: float = 0.01,
        optimizer: str = "Adam",
        lr: float = 0.001,
        loss: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
        metrics: dict[str, torchmetrics.Metric | nn.modules.loss._Loss] = {
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10),
            "cross_entropy": nn.CrossEntropyLoss(),
        },
    ) -> None:
        super(DNN, self).__init__()
        self.save_hyperparameters(
            ignore=["loss", "metrics"]
        )  # We can access the hyperparameters via self.hparams
        self.l2_weight = l2_weight
        self.optimizer = optimizer
        self.lr = lr

        assert (
            num_layers >= 1
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
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(current_dim, 128))
            self.layers.append(nn.BatchNorm1d(128))
            self.layers.append(nn.ReLU(inplace=True))
            current_dim = 128
        self.layers.append(nn.Linear(current_dim, 10))
        # self.layers.append(nn.Softmax(dim=1))
        # We don't need to use this because nn.CrossEntropyLoss() already includes softmax
        # Also, BCEWithLogitsLoss = Sigmoid + BCELoss
        self.dnn = nn.Sequential(*self.layers)

        # Define loss and metrics
        self.loss = loss
        self.metrics = metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dnn(x)

        return y

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]
    ]:
        optimizer = getattr(torch.optim, self.optimizer)(
            self.parameters(), lr=self.lr, weight_decay=self.l2_weight
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        return [optimizer], [scheduler]

    def shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], mode: str
    ) -> torch.Tensor:
        x, y = batch

        # Get outputs
        y_pred = self(x)

        # Compute loss
        loss = self.loss(y_pred, y)

        # Logging
        sync_dist = False if mode == "train" else True
        log_config = {"sync_dist": sync_dist, "prog_bar": True, "on_epoch": True}
        self.log(f"{mode}_loss", loss, **log_config)
        for metric_name, metric in self.metrics.items():
            self.log(f"{mode}_{metric_name}", metric(y_pred, y), **log_config)

        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.shared_step(batch, mode="train")
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss = self.shared_step(batch, mode="val")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss = self.shared_step(batch, mode="test")

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
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
    train_dl: data.DataLoader,
    test_dl: data.DataLoader,
    model: pl.LightningModule,
    epochs: int = 3,
    additional_callbacks: list = [],
    use_gpu: bool = False,
) -> pl.Trainer:
    # Set callbacks
    # (We don't need to set the tensorboard logger because it is set by default)
    early_stopping_with_TerminateOnNaN = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, mode="min", verbose=True, check_finite=True
    )
    callbacks = [early_stopping_with_TerminateOnNaN]
    callbacks.extend(additional_callbacks)

    # Set trainer
    default_root_dir = os.path.join(
        "ray_results",
        "tune_MNIST_000",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    os.makedirs(os.path.join(default_root_dir, "lightning_logs"), exist_ok=True)
    device_config = {}
    if not use_gpu:
        device_config["accelerator"] = "cpu"
    else:
        device_config["accelerator"] = "gpu"
        device_config["devices"] = [0, 1, 2, 3]
        device_config[
            "strategy"
        ] = "ddp_find_unused_parameters_false"  # Allow to have unused parameters
    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        max_epochs=epochs,
        log_every_n_steps=1,  # default: 50
        callbacks=callbacks,
        **device_config,
    )

    # Train the model
    trainer.fit(model, train_dl, test_dl)

    return trainer


def plot_predictions(
    model: pl.LightningModule,
    trainer: pl.Trainer,
    test_dl: data.DataLoader,
) -> None:
    # Get all the predictions (y_pred_list[0].shape: (32, 10))
    y_pred_list = trainer.predict(model, dataloaders=test_dl)
    y_pred = y_pred_list[0]  # Extract the first batch

    # Show the first 5 predictions
    x, y = next(iter(test_dl))
    for i in range(5):
        plt.imshow(x[i, 0, :, :], cmap="gray")
        plt.title(f"Label: {y[i]}, Prediction: {np.argmax(y_pred[i])}")
        plt.show()


def trainable(config: dict, other_kwargs: dict, ray_tune: bool = True) -> None:
    # Load data
    train_dl, test_dl = load_MNIST(batch_size=config["batch_size"])
    if not ray_tune:
        show_data(train_dl)  # Show the data

    # Get the model
    model = DNN(
        num_layers=config["num_layers"],
        l2_weight=config["l2_weight"],
        optimizer=config["optimizer"],
        lr=config["lr"],
        loss=other_kwargs["loss"],
        metrics=other_kwargs["metrics"],
    )
    if not ray_tune:
        print(model)

        # Plot the model
        # plot_model_with_netron(model)

    # Determine additional_callbacks (for logging/plotting purposes only)
    additional_callbacks = []
    if not ray_tune:
        # PyTorch Lightning will handle it automatically
        pass
    else:
        additional_callbacks.append(
            TuneReportCheckpointCallback(
                # metrics={"val_loss": "val_loss", "val_accuracy": "val_accuracy"},
                metrics=["val_loss", "val_accuracy"],
                # filename="checkpoint", # (default)
            )
        )

    # Train
    print("---------------------------------------")
    print("Training ...")
    trainer = train_model(
        train_dl,
        test_dl,
        model,
        epochs=config["epochs"],
        additional_callbacks=additional_callbacks,
        use_gpu=other_kwargs["use_gpu"],
    )

    if not ray_tune:
        # Evaluate
        print("---------------------------------------")
        print("Evaluating ...")
        # The length of the loss_list corresponds to the number of dataloaders used.
        loss_list = trainer.test(dataloaders=test_dl)

        # Predict
        print("---------------------------------------")
        print("Predicting ...")
        plot_predictions(model, trainer, test_dl)


if __name__ == "__main__":
    other_kwargs = {
        "loss": nn.CrossEntropyLoss(),
        "metrics": {
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10),
            "cross_entropy": nn.CrossEntropyLoss(),
        },
        "use_gpu": False,  # if True, please use script to run the code
    }
    config = {
        "batch_size": 256,
        "optimizer": "Adam",
        "lr": 0.001,
        "num_layers": 3,
        "l2_weight": 0.01,
        "epochs": 3,
    }

    # Set all raodom seeds (Python, NumPy, PyTorch)
    pl.seed_everything(seed=0)

    trainable(config, other_kwargs, ray_tune=False)
