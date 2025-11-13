"""Create a dummy Keras model and matching label map for quick testing.

This script builds a tiny CNN matching the expected input shape (28x28x1)
and the number of classes used by the app's DEFAULT_CLASS_MAP. It saves the
model at `models/airwrite_cnn.h5` and a `models/label_map.json` file so the
app can run predictions for testing (note: this dummy model is untrained and
will produce arbitrary predictions but is useful to verify the full app
pipeline).

Run:
    python scripts\create_dummy_model.py

After running, start the app normally.
"""
import json
import os
from pathlib import Path

import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import save_model


DEFAULT_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/()="


def build_dummy_model(num_classes: int):
    inp = Input(shape=(28, 28, 1))
    x = Conv2D(8, (3, 3), activation="relu", padding="same")(inp)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inp, out)
    return model


def main():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    num_classes = len(DEFAULT_CHARS)
    print(f"Building dummy model with {num_classes} classes...")
    model = build_dummy_model(num_classes)

    model_path = models_dir / "airwrite_cnn.h5"
    label_path = models_dir / "label_map.json"

    # Save model (untrained)
    print(f"Saving dummy model to {model_path}")
    save_model(model, str(model_path), include_optimizer=False)

    # Save label map
    label_map = {str(i): ch for i, ch in enumerate(DEFAULT_CHARS)}
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Saved label map to {label_path}")


if __name__ == "__main__":
    main()
