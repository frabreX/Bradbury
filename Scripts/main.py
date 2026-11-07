import numpy as np
import os
import json


def softmax(Z_0):
    softmax = np.exp(Z_0) / np.sum(np.exp(Z_0))
    return softmax

def cross_entropy(softmax_o, y_true):
    loss = y_true * np.log(softmax_o)
    return -np.sum(loss)


#Saving the weights
path = "weights.json"

mnist_path = "mnist.npz"

if os.path.exists(path):
    print("Weights have already been generated")

    with open(path, "r") as f:
        arrays = json.load(f)
        W1 = np.array(arrays["W1"])
        b1 = np.array(arrays["b1"])

else:
    W1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)

    arrays = {
        "W1": W1.tolist(),
        "b1": b1.tolist(),
    }

    with open(path, "w") as f:
        json.dump(arrays, f)

#Input and output data

with (np.load(mnist_path, allow_pickle=False) as data):
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]


# main cycle
i = 0
epochs = 10

for epoch in range(epochs):
    for i in range(len(x_train)):

        x = x_train[i].reshape(784, 1) / 255
        label = y_train[i]
        y_true = np.zeros((10, 1))
        y_true[label, 0] = 1


        # x = x_train[i].reshape(784, 1) / 255
        # y_true = np.zeros((10, 1))
        # y_true[y_train[i], 0] = 1

        # Output Layer
        Z_h = W1 @ x - b1
        softmax_o = softmax(Z_h)
        L = cross_entropy(softmax_o, y_true)

        # Backpropagation
        alpha = softmax_o - y_true                       # (10, 1)
        dW1 = alpha @ x.T                        # (10, 784)
        db1 = -alpha                             # (10, 1)

        #Gradient descent
        n = 0.0001 # Learning rate
        W1 = W1 - n * dW1
        b1 = b1 - n * db1

    print(f"The epoch {epoch+1} has finished")

    test_accuracy = True

    if test_accuracy is True:
        correct = 0
        for image, test_label in zip(x_test, y_test):
            x = image.reshape(784, 1) / 255
            most_probable = np.argmax(W1 @ x - b1)
            if test_label == most_probable:
                correct += 1
        print(f"The accuracy of the model is {correct / len(y_test):.2%}")

arrays = {
    "W1": W1.tolist(),
    "b1": b1.tolist(),
}

with open(path, "w") as f:
    json.dump(arrays, f)

