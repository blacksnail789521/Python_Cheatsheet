import tensorflow as tf
import tensorflow_datasets as tfds


# def load_MNIST():
    
#     (ds_train, ds_test), ds_info = tfds.load(
#         'mnist',
#         split=['train', 'test'],
#         shuffle_files=True,
#         as_supervised=True,
#         with_info=True,
#     )

if __name__ == '__main__':
    
    # Load data
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label
    
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    
    # Get the model
    
    # Train
    
    # Predict