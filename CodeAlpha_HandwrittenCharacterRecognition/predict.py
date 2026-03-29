"""
predict.py — Run inference on a single image or a random test sample.

Usage:
    # Predict on a custom image file
    python predict.py --image path/to/letter.png

    # Predict on N random samples from the test set
    python predict.py --demo 10
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from data_loader import CLASS_NAMES, load_data

MODEL_PATH = os.path.join("saved_model", "best_model.keras")


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at '{path}'. Run train.py first."
        )
    return tf.keras.models.load_model(path)


def preprocess_image(image_path):
    """
    Load an image from disk, convert to grayscale 28×28,
    and return a (1, 28, 28, 1) float32 tensor ready for inference.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Invert if background is white (EMNIST uses white-on-black)
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img


def predict_image(model, image_path, top_k=5):
    """Predict the character in a single image file and print top-k results."""
    img = preprocess_image(image_path)
    probs = model.predict(img, verbose=0)[0]

    top_indices = np.argsort(probs)[::-1][:top_k]
    print(f"\nPredictions for: {image_path}")
    print("-" * 35)
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. {CLASS_NAMES[idx]}  —  {probs[idx] * 100:.2f}%")

    predicted = CLASS_NAMES[top_indices[0]]
    confidence = probs[top_indices[0]] * 100

    # Show the image
    plt.figure(figsize=(3, 3))
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"Predicted: {predicted}  ({confidence:.1f}%)", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return predicted, confidence


def demo_predictions(model, n=10):
    """Predict on n random samples from the EMNIST test set."""
    _, ds_test, _ = load_data(batch_size=n)

    for images, labels in ds_test.take(1):
        images_np = images.numpy()
        true_labels = np.argmax(labels.numpy(), axis=1)

    probs = model.predict(images_np, verbose=0)
    pred_labels = np.argmax(probs, axis=1)

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.5))
    axes = axes.flatten()

    correct = 0
    for i in range(n):
        axes[i].imshow(images_np[i].squeeze(), cmap="gray")
        is_correct = pred_labels[i] == true_labels[i]
        correct += int(is_correct)
        color = "green" if is_correct else "red"
        axes[i].set_title(
            f"T:{CLASS_NAMES[true_labels[i]]}  P:{CLASS_NAMES[pred_labels[i]]}\n"
            f"{probs[i][pred_labels[i]] * 100:.1f}%",
            color=color,
            fontsize=8,
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Demo Predictions — {correct}/{n} correct  (green=correct, red=wrong)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("demo_predictions.png", dpi=150)
    print(f"Demo saved to demo_predictions.png  ({correct}/{n} correct)")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict handwritten characters")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to an image file to predict")
    group.add_argument("--demo", type=int, metavar="N",
                       help="Run demo on N random test samples (default 10)")
    args = parser.parse_args()

    model = load_model()

    if args.image:
        predict_image(model, args.image)
    else:
        demo_predictions(model, n=args.demo or 10)
