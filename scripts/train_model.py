import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist  # For demo; replace with EMNIST for letters

# Configuration
MODEL_PATH = os.path.join("models", "airwrite_cnn.h5")
LABEL_MAP_PATH = os.path.join("models", "label_map.json")
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 43  # Adjust based on your classes (e.g., 0-9, A-Z, +-*/()=)

# Load and preprocess data (demo with MNIST; replace with EMNIST for full alphabet)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Save model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Save label map (example for 0-9; extend for full set)
label_map = {i: str(i) for i in range(10)}
with open(LABEL_MAP_PATH, 'w') as f:
    json.dump(label_map, f)
print(f"Label map saved to {LABEL_MAP_PATH}")