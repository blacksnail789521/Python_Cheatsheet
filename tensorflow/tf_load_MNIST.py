import tensorflow as tf
import os
from tensorflow.keras.datasets import mnist
from datetime import datetime
from typing import List, Tuple, Union
import netron
import matplotlib.pyplot as plt
import numpy as np


def load_MNIST(
    normalize: bool = True,
    one_hot: bool = False,
    batch_size: int = 256,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    # Load numpy data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # One-hot encoding for y (not recommended)
    if one_hot:
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]

    # Normalize
    if normalize:
        x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a dimension (for channel) (only for the images, a.k.a. x)
    # For TensorFlow, the channel dimension is the last dimension
    x_train = x_train[:, :, :, None]
    x_test = x_test[:, :, :, None]

    # Convert to dataset
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10000)
        .batch(batch_size)
    )
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(256)

    return train_ds, test_ds


def show_data(ds: tf.data.Dataset) -> None:
    
    # Get the first batch
    x, y = next(iter(ds))
    x, y = x.numpy(), y.numpy()

    # Show the shape
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")

    # Show the first image and its label
    # (remember that the channel dimension is the last dimension)
    plt.imshow(x[0, :, :, 0], cmap="gray")
    plt.title(f"Label: {y[0]}")
    plt.show()


if __name__ == "__main__":

    # Load data
    train_ds, test_ds = load_MNIST(batch_size=256)

    # Show the data
    show_data(train_ds)
