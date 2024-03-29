import torch
import torch.nn as nn

# import lightning as L
import pytorch_lightning as L

import torchmetrics


class LightningModuleWrapper(L.LightningModule):
    def __init__(
        self,
        nn_model: nn.Module,
        l2_weight: float,
        optimizer: torch.optim.Optimizer | str,
        lr: float,
        loss: str,
        metrics: list[str],
        **kwargs,
    ):
        super().__init__()
        self.nn_model = nn_model
        self.name = nn_model.__class__.__name__
        self.save_hyperparameters(
            ignore=["nn_model"]
        )  # We can access the hyperparameters via self.hparams

        """Please note that loss must be differentiable."""

        # Check if the loss is supported
        supported_losses = ["cross_entropy"]
        assert (
            self.hparams.loss in supported_losses  # type: ignore
        ), f"{self.hparams.loss} is not supported."  # type: ignore

        # Define loss
        if self.hparams.loss == "cross_entropy":  # type: ignore
            self.loss = nn.CrossEntropyLoss()

        # Check if the metrics are supported
        supported_metrics = ["cross_entropy", "accuracy"]
        for metric in self.hparams.metrics:  # type: ignore
            assert metric in supported_metrics, f"{metric} is not supported."

        # Define all possible metrics
        if "cross_entropy" in self.hparams.metrics:  # type: ignore
            self.cross_entropy = nn.CrossEntropyLoss()
        if "accuracy" in self.hparams.metrics:  # type: ignore
            self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

        # DON'T DO THIS
        # self.metrics = {
        #     "cross_entropy": nn.CrossEntropyLoss(),
        #     "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10),
        # }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn_model(x)

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]
    ]:
        # If we pass a string, we need to use getattr to get the optimizer
        if isinstance(self.hparams.optimizer, str):  # type: ignore
            optimizer = getattr(torch.optim, self.hparams.optimizer)(  # type: ignore
                self.parameters(),
                lr=self.hparams.lr,  # type: ignore
                weight_decay=self.hparams.l2_weight,  # type: ignore
            )
        else:
            optimizer = self.hparams.optimizer(  # type: ignore
                self.parameters(),
                lr=self.hparams.lr,  # type: ignore
                weight_decay=self.hparams.l2_weight,  # type: ignore
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
        # (when training, on_epoch=True, on_step=True)
        # (when validating/testing, on_epoch=True, on_step=False)
        log_params = {"prog_bar": True, "on_epoch": True}
        self.log(f"{mode}_loss", loss, **log_params)
        for metric in self.hparams.metrics:  # type: ignore
            self.log(f"{mode}_{metric}", getattr(self, metric)(y_pred, y), **log_params)

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
