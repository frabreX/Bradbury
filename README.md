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

## Animated tester

From the repository root, install Manim Community and render the prediction:

```sh
python -m pip install manim
python -m manim -pqh Scripts/tester_manim.py DigitClassifier
```

The animation shows the input pixels, the model's ten softmax probabilities, and how the predicted digit compares with the MNIST label. Manim saves the MP4 under `media/videos/`.

Each run picks a random MNIST test sample and shows its index below the image. To replay a specific sample, set the zero-based `BRADBURY_SAMPLE` environment variable before running Manim. In PowerShell, use `$env:BRADBURY_SAMPLE = "8"` and clear it afterward with `Remove-Item Env:BRADBURY_SAMPLE`. In Bash, prefix the render command with `BRADBURY_SAMPLE=8`.

To watch the scene as it renders in an OpenGL window, use Manim Community 0.22 or later:

```sh
python -m manim --renderer=opengl --live-preview -qm Scripts/tester_manim.py DigitClassifier
```
