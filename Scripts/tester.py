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
i = np.random.randint(len(x_test))

x = x_test[i].reshape(784, 1) / 255
label = y_test[i]
y_true = np.zeros((10, 1))
y_true[label, 0] = 1

# Output Layer
Z_h = W1 @ x - b1
softmax_o = softmax(Z_h)
L = cross_entropy(softmax_o, y_test)



print(f"The input cypher was {label}")
print(f"The probability distribution is {softmax_o}")
most_probable = np.argmax(softmax_o)
print(f"The cypher predicted is {most_probable}")


