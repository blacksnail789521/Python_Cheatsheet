import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import pytorch_lightning as pl


class MNIST_DataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = "./", batch_size: int = 256, split: float = 0.8
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["data_dir"]
        )  # We can access the hyperparameters via self.hparams
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.dl_config = {
            "batch_size": batch_size,
            "num_workers": multiprocessing.cpu_count(),
            "persistent_workers": True,
        }

    def prepare_data(self) -> None:
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.train_size = int(self.hparams.split * len(mnist_full))
            self.val_size = len(mnist_full) - self.train_size
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(
                mnist_full, [self.train_size, self.val_size]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, shuffle=True, **self.dl_config)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, shuffle=False, **self.dl_config)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, shuffle=False, **self.dl_config)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_predict, shuffle=False, **self.dl_config)


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
