# Handwritten Character Recognition — CodeAlpha Internship Task 1

A deep learning project that recognises handwritten alphabetic characters (A–Z) using a Convolutional Neural Network trained on the EMNIST Letters dataset.

---

## Overview

| Item | Detail |
|---|---|
| Dataset | EMNIST Letters (88,800 train / 14,800 test) |
| Model | CNN — 3 conv blocks + dense head |
| Framework | TensorFlow / Keras |
| Classes | 26 (A – Z) |
| Target accuracy | ~90 %+ on test set |

---

## Project Structure

```
CodeAlpha_HandwrittenCharacterRecognition/
├── data_loader.py   — load & preprocess EMNIST dataset
├── model.py         — CNN architecture definition
├── train.py         — training script with callbacks
├── evaluate.py      — evaluation, confusion matrix, sample predictions
├── predict.py       — inference on custom images or test demos
└── requirements.txt — Python dependencies
```

---

## Setup

```bash
# 1. Clone / navigate to the project folder
cd CodeAlpha_HandwrittenCharacterRecognition

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Train
```bash
python train.py                        # default: 30 epochs, batch 128, lr 0.001
python train.py --epochs 50 --lr 5e-4  # custom settings
```

Training outputs:
- `saved_model/best_model.keras` — best checkpoint (by val_accuracy)
- `saved_model/final_model.keras` — model at end of training
- `training_history.png` — accuracy & loss curves
- `logs/` — TensorBoard logs

### Evaluate
```bash
python evaluate.py
```

Outputs:
- Per-class precision, recall, F1 (classification report)
- `confusion_matrix.png`
- `sample_predictions.png`

### Predict

On a custom image:
```bash
python predict.py --image path/to/letter.png
```

Demo on random test samples:
```bash
python predict.py --demo 20
```

---

## Model Architecture

```
Input (28×28×1)
│
├── Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
├── Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
├── Conv2D(128) → BN → MaxPool → Dropout(0.25)
│
├── Flatten
├── Dense(512) → BN → Dropout(0.5)
└── Dense(26, softmax)
```

---

## Author

Internship Task — CodeAlpha Machine Learning Track
