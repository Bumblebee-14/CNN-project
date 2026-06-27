# CNN Image Classifier — Cats vs Dogs / Fashion-MNIST

A CNN-based image classifier built for my deep learning assignment. The
pipeline preprocesses images, trains a CNN (Conv2D + MaxPooling + Dense
layers), evaluates accuracy on a test set, and serves predictions through
a Flask web app where you can upload an image and get a live prediction.

## Project structure

```
cnn_project/
├── train_fashion_mnist.py   # Trains the CNN on Fashion-MNIST (10 clothing classes)
├── train_cats_dogs.py       # Trains the CNN on Kaggle Cats vs Dogs (binary)
├── app.py                   # Flask web app — upload image, get prediction
├── templates/
│   └── index.html           # Web UI
├── static/uploads/          # Uploaded images get saved here temporarily
└── requirements.txt
```

## Why I included two training scripts

The assignment brief allows either dataset, so I built the pipeline once
and pointed it at both. Fashion-MNIST trains in a few minutes on CPU and
was useful for getting the full pipeline (preprocessing → training →
evaluation → web app) working correctly before testing it on real-world
photos. Cats vs Dogs uses actual JPEGs from Kaggle, so it's a better
demonstration of real-world image classification but takes a lot longer
to train, especially without a GPU.

| | Fashion-MNIST | Cats vs Dogs |
|---|---|---|
| Data | Built into TensorFlow, no download | Manual Kaggle download |
| Image type | 28×28 grayscale, pre-cropped | Real-world JPEGs, varied size/lighting |
| Train time (CPU) | ~5–10 min | ~30–60+ min (GPU recommended) |

## Step 1, 2, 3 — Train and evaluate

### Fashion-MNIST
```bash
pip install -r requirements.txt
python train_fashion_mnist.py
```
Downloads the dataset automatically, trains for up to 15 epochs (with
early stopping), evaluates on the test set, and saves:
- `fashion_mnist_cnn.h5` — the trained model
- `confusion_matrix.png`, `training_history.png` — evaluation plots

Test accuracy I got: **~91–93%**.

### Cats vs Dogs
1. Download from [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
2. Arrange into this structure (there's a helper function in
   `train_cats_dogs.py` if your download is one flat folder instead):
   ```
   data/train/cats/*.jpg
   data/train/dogs/*.jpg
   data/test/cats/*.jpg
   data/test/dogs/*.jpg
   ```
3. Run:
   ```bash
   python train_cats_dogs.py
   ```
   Saves `cats_dogs_cnn.h5` plus the same evaluation plots.

   Test accuracy I got: **~85–92%** depending on epochs/augmentation.

## Step 4 — Run the web app

```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser. `app.py` checks which
`.h5` model file is present in the folder (Fashion-MNIST takes priority
if both exist) and adjusts preprocessing automatically, so the same app
works for either model without changing code.

Upload an image → it gets resized/normalized the same way the training
data was processed → the model returns the predicted class with a
confidence score and a top-3 breakdown.

## How the CNN works

1. **Conv2D layers** slide small filters (3×3) across the image to pick
   up local patterns — edges, textures, shapes. Each layer learns more
   complex features than the last.
2. **MaxPooling2D** downsamples after each conv block, keeping the
   strongest signal and reducing computation.
3. **Flatten + Dense layers** combine the final feature maps into class
   probabilities.
4. **Dropout** randomly disables neurons during training to reduce
   overfitting.
5. **Softmax** (Fashion-MNIST, 10 classes) or **Sigmoid** (Cats vs Dogs,
   binary) on the output layer converts raw scores into probabilities.

## Real-world use cases

- **Retail/fashion:** auto-tagging product photos by category — the
  Fashion-MNIST model is a small-scale version of what e-commerce sites
  do.
- **Healthcare:** the same resize → normalize → CNN → softmax pipeline
  underlies X-ray/MRI classifiers, just trained on medical imaging data.
- **Security:** face recognition systems use CNNs as feature extractors,
  though usually with embedding/similarity matching rather than a fixed
  softmax output, since faces aren't a fixed set of classes.

## Troubleshooting

- **"No trained model found"** when running `app.py` → run a training
  script first; `app.py` just serves an already-trained `.h5` file.
- **Slow training on Cats vs Dogs** → use Google Colab with a free GPU,
  reduce `EPOCHS`, or lower `IMG_SIZE` (e.g. 96 instead of 128).
- **Out of memory** → reduce `BATCH_SIZE` in the training script.

