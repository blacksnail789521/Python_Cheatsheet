import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from collections import defaultdict

from dataset_loader.load_MNIST import load_MNIST, show_data
from models.MLP import MLP
from models.CNN import CNN
from utils.visual import visualize_metric
from utils.tools import EarlyStopping

MODEL_MAP = {
    "MLP": MLP,
    "CNN": CNN,
}


class Exp_Classification(object):
    def __init__(self, args) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self._get_data()
        self._get_model()
        self._set_criterion()
        self._set_optimizer()
        self._set_early_stopping()

    def _get_data(self):
        # Load data
        self.train_loader, self.test_loader = load_MNIST(
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )
        self.val_loader = self.test_loader
        show_data(self.train_loader)  # Show the data

    def _get_model(self):
        # Dynamically get the model class based on the model name
        ModelClass = MODEL_MAP.get(self.args.model_name, None)
        if ModelClass is None:
            raise ValueError(f"Unknown model name: {self.args.model_name}")

        self.model = ModelClass(**self.args.model_params[self.args.model_name]).to(
            self.device
        )

    def _set_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _set_optimizer(self):
        # optimizer
        self.optimizer = getattr(optim, self.args.optim)(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        # scheduler
        if self.args.lr_scheduler != "none":
            lr_scheduler_params = self.args.lr_scheduler_params[self.args.lr_scheduler]
            if self.args.lr_scheduler == "CyclicLR":
                lr_scheduler_params["base_lr"] = self.args.learning_rate
                lr_scheduler_params["cycle_momentum"] = (
                    True if self.args.optim == "SGD" else False
                )
            elif self.args.lr_scheduler == "OneCycleLR":
                lr_scheduler_params["steps_per_epoch"] = len(self.train_loader)
                lr_scheduler_params["epochs"] = self.args.epochs
            self.scheduler = getattr(optim.lr_scheduler, self.args.lr_scheduler)(
                self.optimizer, **lr_scheduler_params
            )

    def _set_early_stopping(self):
        self.early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=self.args.delta
        )

    def train(self) -> dict[str, defaultdict]:
        # * Load checkpoint if it exists
        if (
            self.args.checkpoint_loading_path
            and Path(self.args.checkpoint_loading_path).exists()
        ):
            print(f"Loading checkpoint from {self.args.checkpoint_loading_path} ...")
            model_state = torch.load(Path(self.args.checkpoint_loading_path), weights_only=True)
            self.model.load_state_dict(model_state)

        # Automatic Mixed Precision (some op. are fp32, some are fp16)
        scaler = torch.amp.GradScaler("cuda", enabled=self.args.use_amp)  # type: ignore

        # * Train the model
        metrics = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list),
        }
        for epoch in range(self.args.epochs):
            self.model.train()
            train_losses = []

            iter_data = (
                tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}/{self.args.epochs}, Training Loss: {0}",
                )
                if self.args.use_tqdm
                else self.train_loader
            )
            for i, (x, y) in enumerate(iter_data):
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                # ? 1. Zero grad
                self.optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=self.args.use_amp):  # type: ignore
                    # ? 2. Call the model
                    y_pred = self.model(x)

                    # ? 3. Calculate loss
                    loss = self.criterion(y_pred, y)
                    train_losses.append(loss.item())

                # ? 4. Backward
                scaler.scale(loss).backward()  # type: ignore
                scaler.step(self.optimizer)
                scaler.update()

                if self.args.use_tqdm:
                    iter_data.set_description(  # type: ignore
                        f"Epoch {epoch + 1}/{self.args.epochs}, Training Loss: {np.mean(train_losses)}"
                    )

            # * At the end of each epoch, we get all the metrics
            train_metrics = self.get_metrics(self.train_loader)
            val_metrics = self.get_metrics(self.val_loader)
            test_metrics = self.get_metrics(self.test_loader)
            for key in train_metrics.keys():
                metrics["train"][key].append(train_metrics[key])
                metrics["val"][key].append(val_metrics[key])
                metrics["test"][key].append(test_metrics[key])

            # * Show metrics for all the previous epochs
            visualize_metric(metrics, mode="table")
            visualize_metric(metrics, mode="plot")

            # * Early stopping
            self.early_stopping(
                val_metrics["loss"], self.model, self.args.checkpoint_saving_path
            )
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            # * Learning rate scheduler
            if self.args.lr_scheduler != "none":
                previous_lr = self.optimizer.param_groups[0]["lr"]
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1}/{self.args.epochs}, Learning Rate: {previous_lr} -> {current_lr}"
                    f" (lr_scheduler: {self.args.lr_scheduler}, "
                    f"lr_scheduler_params: {self.args.lr_scheduler_params[self.args.lr_scheduler]})"
                )

        return metrics

    def get_metrics(self, data_loader: DataLoader) -> dict[str, float]:
        total_preds = []
        total_trues = []
        total_loss = 0
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(
                tqdm(data_loader) if self.args.use_tqdm else data_loader
            ):
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                # ? 1. Zero grad
                pass

                with torch.amp.autocast("cuda", enabled=self.args.use_amp):  # type: ignore
                    # ? 2. Call the model
                    y_pred = self.model(x)

                    # ? 3. Calculate loss
                    loss = self.criterion(y_pred, y)
                    total_loss += loss.item()
                    total_samples += len(x)

                pred = (
                    torch.argmax(y_pred, dim=-1).detach().cpu().numpy()
                )  # (B, K) -> (B,)
                true = y.detach().cpu().numpy()  # (B,)

                total_preds.extend(pred)
                total_trues.extend(true)

        assert total_samples == len(total_preds) == len(total_trues)
        loss = total_loss / total_samples
        acc = float(accuracy_score(total_trues, total_preds))
        mf1 = float(f1_score(total_trues, total_preds, average="macro"))
        kappa = float(cohen_kappa_score(total_trues, total_preds))

        return {"loss": loss, "acc": acc, "mf1": mf1, "kappa": kappa}
