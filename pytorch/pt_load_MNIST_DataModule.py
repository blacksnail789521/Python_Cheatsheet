import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import pytorch_lightning as L

# import lightning as L


class MNIST_DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 256,
        split: float = 0.8,
        shuffle: bool = True,
        max_concurrent_trials: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["data_dir", "shuffle", "max_concurrent_trials"]
        )  # We can access the hyperparameters via self.hparams
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.train_dl_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": multiprocessing.cpu_count() // max_concurrent_trials,
            "persistent_workers": True
            if max_concurrent_trials == 1
            else False,  # turn off with ray tune
        }
        self.non_train_dl_params = self.train_dl_params.copy()
        self.non_train_dl_params["shuffle"] = False

    def prepare_data(self) -> None:
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            ds = MNIST(self.data_dir, train=True, transform=self.transform)
            self.train_size = int(self.hparams.split * len(ds))  # type: ignore
            self.val_size = len(ds) - self.train_size
            self.ds_train, self.ds_val = random_split(
                ds, [self.train_size, self.val_size]
            )

        # Assign test dataset for use in dataloader(s)
        if stage in ["test", "predict"] or stage is None:
            self.ds_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_train, **self.train_dl_params)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_val, **self.non_train_dl_params)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_test, **self.non_train_dl_params)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_test, **self.non_train_dl_params)


def show_data(dataloader: DataLoader) -> None:
    # Get the first batch
    x, y = next(iter(dataloader))
    x, y = x.numpy(), y.numpy()

    # Show the shape
    print(f"x.shape: {x.shape}, x.dtype: {x.dtype}")
    print(f"y.shape: {y.shape}, y.dtype: {y.dtype}")

    # Show the first image and its label
    # (remember that the channel dimension is the second dimension)
    plt.imshow(x[0, 0, :, :], cmap="gray")
    plt.title(f"Label: {y[0]}")
    plt.show()


if __name__ == "__main__":
    # Load data
    dm = MNIST_DataModule(batch_size=256, split=0.8)
    dm.prepare_data()
    dm.setup()

    # Show the data

    show_data(dm.train_dataloader())
