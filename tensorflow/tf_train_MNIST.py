import tensorflow as tf
import os
from datetime import datetime
from typing import Union
import netron
import matplotlib.pyplot as plt
import numpy as np
from ray.tune.integration.keras import TuneReportCheckpointCallback

from tf_load_MNIST import load_MNIST, show_data


def DNN(
    num_layers: int = 2,
    l2_weight: float = 0.01,
    optimizer: str = "Adam",
    lr: float = 0.001,
    loss: Union[str, tf.keras.losses.Loss] = "categorical_crossentropy",
    metrics: list[Union[str, tf.keras.metrics.Metric]] = [
        "accuracy",
        "categorical_crossentropy",
    ],
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
        tf.keras.layers.Softmax()
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
        activation="softmax",
    )(outputs)

    model = tf.keras.Model(inputs, outputs, name="DNN")

    # Compile the model
    model.compile(
        optimizer=getattr(tf.keras.optimizers, optimizer)(learning_rate=lr),
        loss=loss,
        metrics=metrics,
    )

    return model


def show_model(model: tf.keras.Model) -> dict[str, int]:
    # Show the model's summary
    print("---------------------------------------")
    model.summary()

    # Get the model name and the number of params
    model_name = model.name
    num_trainable_params = int(
        np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    )
    num_non_trainable_params = int(
        np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    )

    return {
        "model_name": model_name,
        "num_trainable_params": num_trainable_params,
        "num_non_trainable_params": num_non_trainable_params,
    }


def plot_model_with_netron(model: tf.keras.Model, name: str = "DNN") -> None:
    # Save the model
    model_path = os.path.join("saved_models", f"{name}.h5")  # Only support .h5
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    # Plot the model
    netron.start(model_path, address=8080)


def train_model(
    train_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    model: tf.keras.Model,
    epochs: int = 3,
    additional_callbacks: list = [],
) -> None:
    # Set callbacks (early_stopping, TerminateOnNaN)
    # TensorBoard would be added by Ray Tune
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, mode="min", verbose=1
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    callbacks.extend(additional_callbacks)

    # Train the model
    model.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=callbacks)


def plot_predictions(model: tf.keras.Model, test_ds: tf.data.Dataset) -> None:
    # Get all the predictions (y_pred.shape: (10000, 10))
    y_pred = model.predict(test_ds)

    # Show the first 5 predictions
    x, y = next(iter(test_ds))
    for i in range(5):
        plt.imshow(x[i, :, :, 0], cmap="gray")
        plt.title(f"Label: {y[i]}, Prediction: {np.argmax(y_pred[i])}")
        plt.show()


def trainable(config: dict, other_kwargs: dict, ray_tune: bool = True) -> None:
    # Load data
    train_ds, test_ds = load_MNIST(batch_size=config["batch_size"])
    if not ray_tune:
        show_data(train_ds)  # Show the data

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
        _ = show_model(model)  # Show the model

        # Plot the model
        # plot_model_with_netron(model)
        tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

    # Determine additional_callbacks (for logging/plotting purposes only)
    additional_callbacks = []
    if not ray_tune:
        tensorboard_path = os.path.join(
            "ray_results",
            "tune_MNIST_000",
            "tensorboard",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        os.makedirs(tensorboard_path, exist_ok=True)
        additional_callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path)
        )
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
    train_model(
        train_ds,
        test_ds,
        model,
        epochs=config["epochs"],
        additional_callbacks=additional_callbacks,
    )

    if not ray_tune:
        # Test
        print("---------------------------------------")
        print("Testing ...")
        test_loss_dict = model.evaluate(test_ds, return_dict=True)
        print(f"test_loss_dict: {test_loss_dict}")

        # Predict
        print("---------------------------------------")
        print("Predicting ...")
        plot_predictions(model, test_ds)


if __name__ == "__main__":
    other_kwargs = {
        # "loss": "categorical_crossentropy",  # one-hot encoding
        "loss": "sparse_categorical_crossentropy",  # label encoding
        # label encoding w/o softmax
        # "loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #
        "metrics": ["accuracy"],
    }
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
        "optimizer": "Adam",
        "lr": 0.001,
        "num_layers": 3,
        "l2_weight": 0.01,
        "epochs": 3,
    }

    # Set all random seeds (Python, NumPy, TensorFlow)
    tf.keras.utils.set_random_seed(seed=0)

    trainable(config, other_kwargs, ray_tune=False)
