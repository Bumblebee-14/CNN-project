"""
CNN Image Classifier — Cats vs Dogs (Kaggle dataset)
=====================================================
Same pipeline as train_fashion_mnist.py, adapted for real-world RGB
photographs loaded from disk instead of a built-in dataset.

SETUP (do this first):
  1. Download the dataset from Kaggle:
     https://www.kaggle.com/c/dogs-vs-cats/data
     (or the cleaner "Cat and Dog" dataset by tongpython, which is
      already split into train/test folders)
  2. Unzip so you have a folder structure like:

       data/
         train/
           cats/   (cat.0.jpg, cat.1.jpg, ...)
           dogs/   (dog.0.jpg, dog.1.jpg, ...)
         test/
           cats/
           dogs/

     If your download is one big folder with files named cat.N.jpg /
     dog.N.jpg instead, run the `organize_kaggle_folder()` helper below
     once to sort them into class subfolders (Keras needs that structure).

  3. pip install tensorflow pillow scikit-learn matplotlib seaborn
  4. python train_cats_dogs.py

Saves: cats_dogs_cnn.h5, training_history.png, confusion_matrix.png
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "data"              # expects data/train/{cats,dogs}, data/test/{cats,dogs}
IMG_SIZE = 128                  # resize every photo to 128x128
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "cats_dogs_cnn.h5"
CLASS_NAMES = ["cats", "dogs"]


def organize_kaggle_folder(raw_dir="train_raw", out_dir="data/train"):
    """
    One-time helper: if you downloaded the raw Kaggle zip where all
    images sit in a single folder named like 'cat.123.jpg' / 'dog.45.jpg',
    this sorts them into class subfolders that Keras can read.
    Skip this function if your data is already organized into folders.
    """
    for cls in ["cats", "dogs"]:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    for fname in os.listdir(raw_dir):
        if fname.startswith("cat"):
            shutil.copy(os.path.join(raw_dir, fname), os.path.join(out_dir, "cats", fname))
        elif fname.startswith("dog"):
            shutil.copy(os.path.join(raw_dir, fname), os.path.join(out_dir, "dogs", fname))
    print(f"Organized images into {out_dir}/cats and {out_dir}/dogs")


# ---------------------------------------------------------------------------
# 1. LOAD + PREPROCESS  (resize, normalize, batch)
# ---------------------------------------------------------------------------
def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    # Normalize pixels to [0, 1]. Done as a layer so it's baked into
    # the saved model (the Flask app won't need to remember to do this).
    normalization = layers.Rescaling(1.0 / 255)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (normalization(x), y)).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y)).cache().prefetch(AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalization(x), y)).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# 2. BUILD CNN  (Conv2D, MaxPooling, Dense)
# ---------------------------------------------------------------------------
def build_model():
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        data_augmentation,

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),   # binary: cat=0, dog=1
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 3. TRAIN + EVALUATE
# ---------------------------------------------------------------------------
def main():
    print("TensorFlow version:", tf.__version__)

    if not os.path.isdir(os.path.join(DATA_DIR, "train")):
        print(
            f"\n[!] Could not find {DATA_DIR}/train. Download the dataset from \n"
            "    https://www.kaggle.com/c/dogs-vs-cats/data and arrange it as:\n"
            "    data/train/cats, data/train/dogs, data/test/cats, data/test/dogs\n"
            "    (see the module docstring at the top of this file for details)."
        )
        return

    train_ds, val_ds, test_ds = build_datasets()
    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend((preds > 0.5).astype(int).flatten())
        y_true.extend(labels.numpy().astype(int).flatten())

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Cats vs Dogs CNN")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
