"""
Flask Web App — Image Classifier
=================================
User uploads an image -> CNN predicts the category -> result shown in browser.

Works with EITHER trained model in this project:
  - fashion_mnist_cnn.h5  (10 clothing classes, grayscale 28x28)
  - cats_dogs_cnn.h5      (cat/dog binary, RGB 128x128)

It auto-detects whichever .h5 file is present in this folder and adapts
preprocessing accordingly.

Run with:
  python app.py
Then open http://127.0.0.1:5000 in your browser.

Requires: flask, tensorflow, pillow, numpy
"""

import os
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

# ---------------------------------------------------------------------------
# MODEL SETUP — auto-detect which model is available
# ---------------------------------------------------------------------------
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
CATSDOGS_CLASSES = ["Cat", "Dog"]

FASHION_MODEL_PATH = os.path.join(APP_DIR, "fashion_mnist_cnn.h5")
CATSDOGS_MODEL_PATH = os.path.join(APP_DIR, "cats_dogs_cnn.h5")

if os.path.exists(FASHION_MODEL_PATH):
    MODEL_TYPE = "fashion"
    model = load_model(FASHION_MODEL_PATH)
    CLASS_NAMES = FASHION_CLASSES
    IMG_SIZE = 28
elif os.path.exists(CATSDOGS_MODEL_PATH):
    MODEL_TYPE = "catsdogs"
    model = load_model(CATSDOGS_MODEL_PATH)
    CLASS_NAMES = CATSDOGS_CLASSES
    IMG_SIZE = 128
else:
    raise FileNotFoundError(
        "No trained model found. Run train_fashion_mnist.py or "
        "train_cats_dogs.py first to produce a .h5 file in this folder."
    )

print(f"Loaded model type: {MODEL_TYPE}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """Resize + normalize an uploaded image to match what the model expects."""
    img = Image.open(image_path)

    if MODEL_TYPE == "fashion":
        img = img.convert("L")                      # grayscale
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img).astype("float32") / 255.0
        arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    else:  # catsdogs
        img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img).astype("float32") / 255.0
        arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    return arr


def predict(image_path):
    arr = preprocess_image(image_path)
    preds = model.predict(arr, verbose=0)

    if MODEL_TYPE == "fashion":
        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx]) * 100
        label = CLASS_NAMES[idx]
        # also return top-3 for a nicer UI
        top_idx = np.argsort(preds[0])[::-1][:3]
        top3 = [(CLASS_NAMES[i], float(preds[0][i]) * 100) for i in top_idx]
    else:  # catsdogs, sigmoid output
        score = float(preds[0][0])
        idx = int(score > 0.5)
        label = CLASS_NAMES[idx]
        confidence = (score if idx == 1 else 1 - score) * 100
        top3 = [
            (CLASS_NAMES[1], score * 100),
            (CLASS_NAMES[0], (1 - score) * 100),
        ]

    return label, confidence, top3


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None

    if request.method == "POST":
        if "file" not in request.files:
            result = {"error": "No file part in the request."}
        else:
            file = request.files["file"]
            if file.filename == "":
                result = {"error": "No file selected."}
            elif not allowed_file(file.filename):
                result = {"error": "Unsupported file type. Use png/jpg/jpeg/bmp."}
            else:
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(save_path)

                label, confidence, top3 = predict(save_path)
                result = {
                    "label": label,
                    "confidence": f"{confidence:.2f}",
                    "top3": [(name, f"{conf:.2f}") for name, conf in top3],
                }
                image_url = url_for("static", filename=f"uploads/{filename}")

    return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    app.run(debug=True)
