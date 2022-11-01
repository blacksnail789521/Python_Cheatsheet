import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime


def load_MNIST(batch_size=128, show_shape=True):

    # Load ds (no batch, shuffle)
    (ds_train, ds_val), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.0, label

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    if show_shape:
        x, y = next(iter(ds_train))
        print(f"x.shape: {x.shape}")
        print(f"y.shape: {y.shape}")

    return ds_train, ds_val


def get_CNN():

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


def train_model(ds_train, ds_val, model, epochs=3):

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

    print("---------------------------------------")
    print("Training ...")
    model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=callbacks)


if __name__ == "__main__":

    # Load data
    ds_train, ds_val = load_MNIST()

    # Get the model
    model = get_CNN()

    # Train
    train_model(ds_train, ds_val, model)

    # Predict
    y_pred = model.predict(ds_val)
