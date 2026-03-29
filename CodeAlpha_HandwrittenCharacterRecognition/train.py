import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

from data_loader import load_data
from model import build_cnn

SAVE_DIR = "saved_model"
HISTORY_PLOT = "training_history.png"


def plot_history(history, save_path=HISTORY_PLOT):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()


def train(epochs=30, batch_size=128, lr=1e-3):
    # ── Data ──────────────────────────────────────────────────────────────
    ds_train, ds_test, class_names = load_data(batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_cnn()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir="logs"),
    ]

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\nTraining for up to {epochs} epochs (batch_size={batch_size}, lr={lr})\n")
    history = model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=epochs,
        callbacks=callbacks,
    )

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(SAVE_DIR, "final_model.keras")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    plot_history(history)

    # ── Final evaluation ──────────────────────────────────────────────────
    loss, acc = model.evaluate(ds_test, verbose=0)
    print(f"\nTest accuracy : {acc * 100:.2f}%")
    print(f"Test loss     : {loss:.4f}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train handwritten character recognition CNN")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
