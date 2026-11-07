# Bradbury

Bradbury is a Python and NumPy project for classifying handwritten digits from the MNIST dataset. It uses a single-layer softmax classifier and trains it with stochastic gradient descent.

## Setup

From the repository root, install NumPy and switch to the `Scripts` directory so the scripts can find the included dataset and weights.

```sh
python -m pip install numpy
cd Scripts
```

## Run

```sh
python tester.py
```

The tester selects a random MNIST test image and prints its label, the model's class probabilities, and its prediction.

```sh
python main.py
```

The training script loads `weights.json` if present, trains for three epochs on `mnist.npz`, and saves the updated parameters to `weights.json`.
