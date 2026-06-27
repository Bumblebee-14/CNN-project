"""
CNN Image Classifier — Fashion-MNIST
=====================================
Trains a Convolutional Neural Network to classify clothing images into
10 categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt,
Sneaker, Bag, Ankle boot).

Run with:  python train_fashion_mnist.py
Requires:  tensorflow, numpy, matplotlib, scikit-learn

This will save:
  - fashion_mnist_cnn.h5   (the trained model, used by the Flask app)
  - training_history.png  (accuracy/loss curves)
  - confusion_matrix.png  (evaluation visual)
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
IMG_SIZE = 28          # Fashion-MNIST images are already 28x28 grayscale
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 15
MODEL_PATH = "fashion_mnist_cnn.h5"

print("TensorFlow version:", tf.__version__)

# ---------------------------------------------------------------------------
# 2. LOAD + PREPROCESS DATA
#    (Step 1 of the assignment: resize / normalize images)
# ---------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# Normalize pixel values from [0, 255] -> [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1) since CNNs expect a
# channel axis (1 = grayscale, 3 would be RGB)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Carve out a validation set from training data
val_split = int(0.9 * len(x_train))
x_val, y_val = x_train[val_split:], y_train[val_split:]
x_train, y_train = x_train[:val_split], y_train[:val_split]

print(f"Final -> train: {x_train.shape}, val: {x_val.shape}, test: {x_test.shape}")

# Light data augmentation helps generalization on real-world-style photos
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
])

# ---------------------------------------------------------------------------
# 3. BUILD THE CNN
#    (Step 2: Conv2D, MaxPooling, Dense layers)
# ---------------------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    data_augmentation,

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),               # 28x28 -> 14x14

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),               # 14x14 -> 7x7

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),               # 7x7 -> 3x3

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2
    ),
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)

# ---------------------------------------------------------------------------
# 5. EVALUATE ON TEST SET
#    (Step 3: Evaluate accuracy on test dataset)
# ---------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss:     {test_loss:.4f}")

y_pred = np.argmax(model.predict(x_test), axis=1)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Fashion-MNIST CNN")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"], label="train")
axes[0].plot(history.history["val_accuracy"], label="val")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"], label="train")
axes[1].plot(history.history["val_loss"], label="val")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
print("Saved training_history.png")

# ---------------------------------------------------------------------------
# 6. SAVE MODEL  (used directly by app.py)
# ---------------------------------------------------------------------------
model.save(MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
