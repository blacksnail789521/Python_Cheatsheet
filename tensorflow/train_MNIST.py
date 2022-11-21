import tensorflow as tf
import os
from tensorflow.keras.datasets import mnist
from datetime import datetime
from typing import List, Tuple, Union
import netron
import matplotlib.pyplot as plt
import numpy as np


def load_MNIST(batch_size: int = 256) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    # Load numpy data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a dimension (for channel) (only for the images, a.k.a. x)
    x_train = x_train[:, :, :, None]
    x_test = x_test[:, :, :, None]

    # Convert to dataset
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10000)
        .batch(batch_size)
    )
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    return train_ds, test_ds


def show_data(test_ds: tf.data.Dataset) -> None:

    # Get the first batch
    x, y = next(iter(test_ds))
    x, y = x.numpy(), y.numpy()

    # Show the shape
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")

    # Show the first image and its label
    plt.imshow(x[0, :, :, 0], cmap="gray")
    plt.title(f"Label: {y[0]}")
    plt.show()


def DNN(
    num_layers: int = 2,
    l2_weight: float = 0.01,
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam",
) -> tf.keras.Model:

    assert (
        num_layers >= 1
    ), "We should have at least one layer because the output layer is counted."

    tf.keras.backend.clear_session()  # We don't want to mess up with model's name

    # Define the model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),

        -----------------------------------------
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        -----------------------------------------
        
        tf.keras.layers.Dense(10),
    ], name = model_name)
    """

    inputs = tf.keras.Input(shape=(28, 28), name="input")
    outputs = tf.keras.layers.Flatten()(inputs)
    for _ in range(num_layers - 1):
        outputs = tf.keras.layers.Dense(
            128,
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
        )(outputs)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.ReLU()(outputs)
    outputs = tf.keras.layers.Dense(
        10,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
    )(outputs)

    model = tf.keras.Model(inputs, outputs, name="DNN")

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # metrics=["accuracy"],  # Same as tf.keras.metrics.SparseCategoricalAccuracy()
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model


def plot_model_with_netron(model: tf.keras.Model, name: str = "DNN") -> None:

    # Save the model
    model_path = os.path.join("models", f"{name}.h5")  # Only support .h5
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    # Plot the model
    netron.start(model_path, address=8080)


def train_model(
    train_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    model: tf.keras.Model,
    epochs: int = 3,
    additional_callbacks: List = [],
) -> None:

    # Set callbacks (early_stopping, tensorboard, TerminateOnNaN)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, mode="min", verbose=1
    )
    tensorboard_path = os.path.join(
        "tensorboard", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    os.makedirs(tensorboard_path, exist_ok=True)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
    callbacks = [early_stopping, tensorboard, tf.keras.callbacks.TerminateOnNaN()]
    callbacks.extend(additional_callbacks)

    # Train the model
    model.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=callbacks)


def predict_with_model(model: tf.keras.Model, test_ds: tf.data.Dataset) -> None:

    # Get all the predictions (y_pred.shape: (10000, 10))
    y_pred = model.predict(test_ds)

    # Show the first 5 predictions
    x, y = next(iter(test_ds))
    for i in range(5):
        plt.imshow(x[i, :, :, 0], cmap="gray")
        plt.title(f"Label: {y[i]}, Prediction: {np.argmax(y_pred[i])}")
        plt.show()


if __name__ == "__main__":

    config = {
        "batch_size": 256,
        # "optimizer": tf.keras.optimizers.Adam(
        #     learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        #         initial_learning_rate=0.001,
        #         decay_steps=1000,
        #         decay_rate=0.9,
        #         staircase=True,
        #     )
        # ),
        "optimizer": "adam",
        "num_layers": 2,
        "l2_weight": 0.01,
        "epochs": 3,
    }

    # Load data
    train_ds, test_ds = load_MNIST(batch_size=config["batch_size"])

    # Show the data
    show_data(test_ds)

    # Get the model
    model = DNN(
        num_layers=config["num_layers"],
        l2_weight=config["l2_weight"],
        optimizer=config["optimizer"],
    )
    model.summary()

    # Plot the model
    plot_model_with_netron(model)
    # tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

    # Train
    print("---------------------------------------")
    print("Training ...")
    train_model(train_ds, test_ds, model, epochs=config["epochs"])

    # Predict
    print("---------------------------------------")
    print("Predicting ...")
    predict_with_model(model, test_ds)
