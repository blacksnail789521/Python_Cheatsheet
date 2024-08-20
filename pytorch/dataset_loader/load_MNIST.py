import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing


class MNIST_Dataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        normalize: bool = True,
        one_hot: bool = False,
    ) -> None:
        # Load data based on mode
        self.x = x
        self.y = y

        # One-hot encoding for y
        if one_hot:
            self.y = np.eye(10)[self.y]

        # Normalize
        if normalize:
            self.x = self.x / 255.0

        # Add a dimension (for channel) (only for the images, a.k.a. x)
        # For PyTorch, the channel dimension is the second dimension
        self.x = self.x[:, None, :, :]

        # Change to correct type
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def load_MNIST(
    input_root_path: Path = Path.cwd(),
    normalize: bool = True,
    one_hot: bool = False,
    batch_size: int = 256,
    use_numpy: bool = False,
    shuffle: bool = True,
    num_workers: int = multiprocessing.cpu_count(),
) -> tuple[DataLoader, DataLoader]:
    if use_numpy:
        # from tensorflow.keras.datasets import mnist
        # (x_train, y_train), (x_test, y_test) = mnist.load_data()

        with np.load(
            "/home/blacksnail789521/.keras/datasets/mnist.npz", allow_pickle=True
        ) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]

        # Get ds
        train_ds = MNIST_Dataset(x_train, y_train, normalize, one_hot)
        test_ds = MNIST_Dataset(x_test, y_test, normalize, one_hot)
    else:
        # Get ds
        train_ds = MNIST(
            root=str(input_root_path / "dataset"),
            train=True,
            download=False,
            transform=transforms.ToTensor(),
            # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to
            # a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        )
        test_ds = MNIST(
            root=str(input_root_path / "dataset"),
            train=False,
            download=False,
            transform=transforms.ToTensor(),
        )

    # Get loader
    train_loader_params = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        # "persistent_workers": True
        # if max_concurrent_trials == 1
        # else False,  # turn off with ray tune
    }
    non_train_loader_params = train_loader_params.copy()
    non_train_loader_params["shuffle"] = False
    train_loader = DataLoader(train_ds, **train_loader_params)
    test_loader = DataLoader(test_ds, **non_train_loader_params)

    return train_loader, test_loader


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
    input_root_path = Path.cwd().parent
    # Load data
    train_loader, test_loader = load_MNIST(
        input_root_path=input_root_path, batch_size=256, use_numpy=False
    )
    # train_loader, test_loader = load_MNIST(batch_size=256, use_numpy=True)

    # Show the data
    show_data(train_loader)
