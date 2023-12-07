import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from pt_load_MNIST import load_MNIST, show_data
from models.MLP import MLP
from models.CNN import CNN
from visual import visualize_metric, plot_predictions
from tools import set_seed, change_dict_to_args, print_formatted_dict, select_best_metrics

MODEL_MAP = {
    "MLP": MLP,
    "CNN": CNN,
}


class Exp_Classification(object):
    def __init__(self, configs) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs = configs
        self._get_data()
        self._get_model()
        self._set_criterion()
        self._set_optimizer()

    def _get_data(self):
        # Load data
        self.train_loader, self.test_loader = load_MNIST(
            batch_size=self.configs.batch_size,
            num_workers=self.configs.num_workers,
        )
        self.val_loader = self.test_loader
        show_data(self.train_loader)  # Show the data

    def _get_model(self):
        model_name = self.configs.model_name
        model_params = self.configs.model_params[model_name]

        # Dynamically get the model class based on the model name
        ModelClass = MODEL_MAP.get(model_name, None)
        if ModelClass is None:
            raise ValueError(f"Unknown model name: {model_name}")

        self.model = ModelClass(**model_params).to(self.device)

    def _set_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _set_optimizer(self):
        self.optimizer = getattr(optim, self.configs.optim)(
            self.model.parameters(),
            lr=self.configs.learning_rate,
            weight_decay=self.configs.weight_decay,
        )

    def train(
        self,
        # checkpoint_dir: Path | None = Path("checkpoints"),
    ) -> dict[str, dict[str, list[float]]]:
        # Load checkpoint if it exists
        if self.configs.checkpoint_dir and self.configs.checkpoint_dir.exists():
            if Path(self.configs.checkpoint_dir, "checkpoint.pth").exists():
                model_state, optimizer_state = torch.load(
                    Path(self.configs.checkpoint_dir, "checkpoint.pth")
                )
                self.model.load_state_dict(model_state)
                self.optimizer.load_state_dict(optimizer_state)

        # Automatic Mixed Precision (some op. are fp32, some are fp16)
        scaler = torch.cuda.amp.GradScaler(enabled=self.configs.use_amp)

        # Train the model
        metrics = {
            "train": {"loss": [], "acc": [], "mf1": [], "kappa": []},
            "val": {"loss": [], "acc": [], "mf1": [], "kappa": []},
            "test": {"loss": [], "acc": [], "mf1": [], "kappa": []},
        }
        for epoch in range(self.configs.epochs):
            self.model.train()
            train_losses = []

            iter_data = (
                tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}/{self.configs.epochs}, Training Loss: {0}",
                )
                if self.configs.use_tqdm
                else self.train_loader
            )
            for i, (x, y) in enumerate(iter_data):
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                # ? 1. Zero grad
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.configs.use_amp):  # type: ignore
                    # ? 2. Call the model
                    y_pred = self.model(x)

                    # ? 3. Calculate loss
                    loss = self.criterion(y_pred, y)
                    train_losses.append(loss.item())

                # ? 4. Backward
                scaler.scale(loss).backward()  # type: ignore
                scaler.step(self.optimizer)
                scaler.update()

                if self.configs.use_tqdm:
                    iter_data.set_description(  # type: ignore
                        f"Epoch {epoch + 1}/{self.configs.epochs}, Training Loss: {np.mean(train_losses)}"
                    )

            # * At the end of each epoch, we get all the metrics
            train_loss, train_acc, train_f1, train_kappa = self.get_metrics(
                self.train_loader
            )
            metrics["train"]["loss"].append(train_loss)
            metrics["train"]["acc"].append(train_acc)
            metrics["train"]["mf1"].append(train_f1)
            metrics["train"]["kappa"].append(train_kappa)
            val_loss, valid_acc, valid_f1, valid_kappa = self.get_metrics(
                self.val_loader
            )
            metrics["val"]["loss"].append(val_loss)
            metrics["val"]["acc"].append(valid_acc)
            metrics["val"]["mf1"].append(valid_f1)
            metrics["val"]["kappa"].append(valid_kappa)
            test_loss, test_acc, test_f1, test_kappa = self.get_metrics(
                self.test_loader
            )
            metrics["test"]["loss"].append(test_loss)
            metrics["test"]["acc"].append(test_acc)
            metrics["test"]["mf1"].append(test_f1)
            metrics["test"]["kappa"].append(test_kappa)

            # * Show metrics for all the previous epochs
            visualize_metric(metrics, mode="table")
            visualize_metric(metrics, mode="plot")

            # Save the model and optimizer
            if self.configs.checkpoint_dir:
                checkpoint_path = Path(self.configs.checkpoint_dir, "checkpoint.pth")
                self.configs.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    (self.model.state_dict(), self.optimizer.state_dict()),
                    checkpoint_path,
                )
        
        return metrics

    def get_metrics(self, data_loader: DataLoader) -> tuple[float, float, float, float]:
        total_preds = []
        total_trues = []
        total_loss = 0
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(
                tqdm(data_loader) if self.configs.use_tqdm else data_loader
            ):
                x = x.float().to(self.device)
                y = y.long().to(self.device)

                # ? 1. Zero grad
                pass

                with torch.cuda.amp.autocast(enabled=self.configs.use_amp):  # type: ignore
                    # ? 2. Call the model
                    y_pred = self.model(x)

                    # ? 3. Calculate loss
                    loss = self.criterion(y_pred, y)
                    total_loss += loss.item()
                    total_samples += len(x)

                pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
                true = y.detach().cpu().numpy()

                total_preds.extend(pred)
                total_trues.extend(true)

        assert total_samples == len(total_preds) == len(total_trues)
        loss = total_loss / total_samples
        acc = float(accuracy_score(total_trues, total_preds))
        mf1 = float(f1_score(total_trues, total_preds, average="macro"))
        kappa = float(cohen_kappa_score(total_trues, total_preds))

        return loss, acc, mf1, kappa


if __name__ == "__main__":
    # Get fixed and tunable parameters
    fixed_params = {
        "num_workers": 0,
        "use_tqdm": True,
        "use_amp": False,
        # "use_amp": True,
        # "checkpoint_dir": Path("checkpoints"),
        "checkpoint_dir": None,
    }
    tunable_params = {
        # "model_name": "MLP",
        "model_name": "CNN",
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
        "optim": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        # "epochs": 1,
        "epochs": 3,
    }

    # Set all random seeds (Python, NumPy, PyTorch)
    set_seed(42)

    # Set configs
    configs = change_dict_to_args({**fixed_params, **tunable_params})

    # Train the model
    exp = Exp_Classification(configs)
    metrics = exp.train()
    print_formatted_dict(metrics)
    best_metrics = select_best_metrics(metrics, target_mode="test", target_metric="acc")
    print_formatted_dict(best_metrics)

    # Plot predicitons
    print("---------------------------------------")
    print("Plotting predictions ...")
    plot_predictions(exp.model, exp.test_loader, enable_show=True)

    print("### Done ###")
