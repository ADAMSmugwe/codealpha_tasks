import tensorflow as tf
from data_loader import NUM_CLASSES, IMAGE_SIZE


def build_cnn(input_shape=IMAGE_SIZE, num_classes=NUM_CLASSES):
    """
    Convolutional Neural Network for handwritten character recognition.

    Architecture:
        Block 1 — Conv(32) → BN → Conv(32) → BN → MaxPool → Dropout
        Block 2 — Conv(64) → BN → Conv(64) → BN → MaxPool → Dropout
        Block 3 — Conv(128) → BN → MaxPool → Dropout
        Head    — Flatten → Dense(512) → BN → Dropout → Dense(num_classes)
    """
    inputs = tf.keras.Input(shape=input_shape, name="image")

    # --- Block 1 ---
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # --- Block 2 ---
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # --- Block 3 ---
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # --- Classification Head ---
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="HandwrittenCharCNN")
    return model


if __name__ == "__main__":
    model = build_cnn()
    model.summary()
