import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# EMNIST letters: 26 classes (a-z), labels are 1-26 in raw data
CLASS_NAMES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # A-Z
NUM_CLASSES = 26
IMAGE_SIZE = (28, 28, 1)


def preprocess(image, label):
    """Normalize image, fix EMNIST orientation, and zero-index the label."""
    image = tf.cast(image, tf.float32) / 255.0
    # EMNIST images are transposed relative to MNIST — fix orientation
    image = tf.transpose(image, perm=[1, 0, 2])
    # Raw labels are 1-26; shift to 0-25
    label = tf.cast(label, tf.int32) - 1
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def load_data(batch_size=128):
    """
    Load the EMNIST letters dataset.

    Returns:
        ds_train: batched training dataset
        ds_test:  batched test dataset
        CLASS_NAMES: list of character labels (A-Z)
    """
    (ds_train, ds_test), ds_info = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    print(f"Training samples : {ds_info.splits['train'].num_examples}")
    print(f"Test samples     : {ds_info.splits['test'].num_examples}")
    print(f"Classes          : {NUM_CLASSES}  ({CLASS_NAMES[0]} – {CLASS_NAMES[-1]})")

    ds_train = (
        ds_train
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(10_000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    ds_test = (
        ds_test
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds_train, ds_test, CLASS_NAMES


def get_sample_batch(ds, n=16):
    """Return n images and their integer labels from a dataset for visualization."""
    images, labels = [], []
    for img_batch, lbl_batch in ds.take(1):
        images = img_batch[:n].numpy()
        labels = np.argmax(lbl_batch[:n].numpy(), axis=1)
    return images, labels


if __name__ == "__main__":
    ds_train, ds_test, names = load_data()
    images, labels = get_sample_batch(ds_train)
    print("Sample batch shape:", images.shape)
    print("Sample labels:", [names[l] for l in labels])
