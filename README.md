# Bradbury

A handwritten-digit classifier built with Python and NumPy. It flattens 28 × 28 MNIST images and uses a single linear layer with softmax to score the ten digits.

## Quick start

From the repository root:

```sh
python -m pip install -r requirements.txt
python Scripts/predict_digit.py
python Scripts/evaluate_accuracy.py
```

The included baseline scores **91.23%** on the 10,000 MNIST test images. The prediction script picks a random test image and prints its label and the ten class probabilities.

## Train a model

| Script | Training method |
| --- | --- |
| `Scripts/train_sgd.py` | Sample-by-sample SGD with random initialization |
| `Scripts/train_sgd_zero_init.py` | Sample-by-sample SGD with zero initialization |
| `Scripts/train_minibatch.py` | Mini-batch SGD with batches of 64 |

Run a trainer with Python. Each trainer reads the bundled `Scripts/mnist.npz`, resumes from a compatible local `Scripts/weights.json` when available, and saves its trained weights there. Prediction, evaluation, and animation prefer that checkpoint and fall back to the bundled baseline.

## Animation

[Digit prediction demo](demos/digit-classifier.mp4) · [MNIST overview](demos/bradbury-mnist.mp4)

Install Manim Community separately to render the scene.

```sh
python -m pip install manim
python -m manim -pqh Scripts/animate_prediction.py DigitClassifier
```

The scene shows the input image, the class probabilities, and the prediction. Set `BRADBURY_SAMPLE` to a zero-based test image index to replay a specific example.
