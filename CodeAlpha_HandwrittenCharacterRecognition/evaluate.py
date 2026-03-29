import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_data, CLASS_NAMES

MODEL_PATH = os.path.join("saved_model", "best_model.keras")


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at '{path}'. Run train.py first."
        )
    print(f"Loading model from {path}")
    return tf.keras.models.load_model(path)


def get_predictions(model, ds_test):
    """Run inference on the full test set and return true/predicted labels."""
    y_true, y_pred = [], []
    for images, labels in ds_test:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title("Confusion Matrix — EMNIST Letters", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_sample_predictions(model, ds_test, class_names, n=20, save_path="sample_predictions.png"):
    """Show a grid of test images with true vs predicted labels."""
    images, labels = [], []
    for img_batch, lbl_batch in ds_test.take(1):
        images = img_batch[:n].numpy()
        labels = np.argmax(lbl_batch[:n].numpy(), axis=1)

    preds = np.argmax(model.predict(images, verbose=0), axis=1)

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.5))
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(images[i].squeeze(), cmap="gray")
        color = "green" if preds[i] == labels[i] else "red"
        axes[i].set_title(
            f"T:{class_names[labels[i]]}  P:{class_names[preds[i]]}",
            color=color,
            fontsize=9,
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Sample predictions saved to {save_path}")
    plt.close()


def evaluate(model_path=MODEL_PATH, batch_size=128):
    model = load_model(model_path)
    _, ds_test, _ = load_data(batch_size=batch_size)

    # Overall accuracy
    loss, acc = model.evaluate(ds_test, verbose=1)
    print(f"\nTest Accuracy : {acc * 100:.2f}%")
    print(f"Test Loss     : {loss:.4f}\n")

    # Per-class report
    y_true, y_pred = get_predictions(model, ds_test)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Visualisations
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)
    plot_sample_predictions(model, ds_test, CLASS_NAMES)


if __name__ == "__main__":
    evaluate()
