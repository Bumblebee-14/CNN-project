# CNN Image Classifier — Cats vs Dogs / Fashion-MNIST

A complete pipeline: preprocess images → train a CNN → evaluate accuracy →
serve predictions through a Flask web app.

## Project structure

```
cnn_project/
├── train_fashion_mnist.py   # Trains CNN on Fashion-MNIST (10 clothing classes)
├── train_cats_dogs.py       # Trains CNN on Kaggle Cats vs Dogs (binary)
├── app.py                   # Flask web app — upload image, get prediction
├── templates/
│   └── index.html           # Web UI (sci-fi dark/cyan theme)
├── static/uploads/          # Uploaded images get saved here temporarily
└── requirements.txt
```

## Why two training scripts?

Both satisfy the assignment brief, but they trade off differently:

| | Fashion-MNIST | Cats vs Dogs |
|---|---|---|
| Data | Built into TensorFlow, no download | Manual Kaggle download required |
| Image type | 28×28 grayscale, pre-cropped | Real-world JPEGs, varied size/lighting |
| Train time (CPU) | ~5–10 min | ~30–60+ min (GPU strongly recommended) |
| Good for | Fast iteration, guaranteed to work first try | The literal "Cats vs Dogs" brief, more impressive demo |

**Recommendation:** start with Fashion-MNIST to get the whole pipeline working
end-to-end quickly, then swap to Cats vs Dogs once everything runs, ideally
on Google Colab with a free GPU (Runtime → Change runtime type → GPU).

## Step 1 & 2 & 3 — Train and evaluate

### Option A: Fashion-MNIST (recommended first)
```bash
pip install -r requirements.txt
python train_fashion_mnist.py
```
This downloads the dataset automatically, trains for up to 15 epochs (with
early stopping), evaluates on the test set, and saves:
- `fashion_mnist_cnn.h5` — the trained model
- `confusion_matrix.png`, `training_history.png` — evaluation visuals

Typical test accuracy: **~91–93%**.

### Option B: Cats vs Dogs
1. Download from [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
2. Arrange into this structure (see docstring in `train_cats_dogs.py` for a
   helper function if your download is one flat folder):
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

   Typical test accuracy: **~85–92%** depending on epochs/augmentation.

## Step 4 — Run the web app

```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser. `app.py` auto-detects
whichever `.h5` model file is present in the folder (Fashion-MNIST takes
priority if both exist) and adjusts preprocessing automatically — same
Flask app works for either model, no code changes needed.

Upload any image → the model resizes/normalizes it the same way training
data was processed → returns the predicted class with a confidence score
and a top-3 breakdown.

## How the CNN architecture works (for your report/viva)

Both models follow the same logic, just scaled differently:

1. **Conv2D layers** slide small filters (3×3) across the image to detect
   local patterns — edges, textures, shapes. Each layer learns more complex
   features than the last (edges → fur/fabric texture → object parts).
2. **MaxPooling2D** downsamples after each conv block, keeping the strongest
   signal and reducing computation while adding translation invariance.
3. **Flatten + Dense layers** take the final feature maps and combine them
   into class probabilities.
4. **Dropout** randomly disables neurons during training to prevent
   overfitting.
5. **Softmax** (Fashion, 10 classes) or **Sigmoid** (Cats vs Dogs, binary)
   on the output layer converts raw scores into probabilities.

## Real-world use case framing (for your submission)

- **Retail / fashion detection:** automatically tagging product photos by
  category (the Fashion-MNIST model is a toy version of this — e-commerce
  platforms use the same CNN→classification pattern at much larger scale).
- **Healthcare:** the identical pipeline (resize → normalize → CNN →
  softmax) underlies X-ray/MRI classifiers, just trained on medical imaging
  datasets with far more rigorous validation.
- **Security:** face recognition systems use CNNs as feature extractors,
  though they typically use embedding/similarity approaches rather than
  fixed-class softmax since the "classes" (faces) aren't known in advance.

## Troubleshooting

- **"No trained model found"** when running `app.py` → run a training
  script first; `app.py` just serves an already-trained `.h5` file.
- **Slow training on Cats vs Dogs** → use Google Colab with GPU, or cut
  `EPOCHS` down, or use a smaller `IMG_SIZE` (e.g. 96 instead of 128).
- **Out of memory** → reduce `BATCH_SIZE` in the training script.
