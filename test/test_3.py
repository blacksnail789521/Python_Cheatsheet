import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.version.VERSION)
print(tf.keras.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)