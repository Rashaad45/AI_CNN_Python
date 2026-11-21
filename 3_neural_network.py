"""
PROJECT 3: NEURAL NETWORK FROM SCRATCH (numpy only)
---------------------------------------------------
Network: 4 inputs -> 8 hidden (ReLU) -> 8 hidden (ReLU) -> 1 output (linear)
Loss:    Mean Squared Error  L = (1/m) * sum( (y_hat - y)^2 )
Training: backpropagation + gradient descent

Shapes used in this file (m = number of samples):
  X        : (m, 4)
  W[l]     : (neurons_in, neurons_out)
  b[l]     : (1, neurons_out)
  z, a     : (m, neurons)

Run:  python 3_neural_network.py
Needs: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


# -------------------------------------------------
# Activation function and its derivative
# -------------------------------------------------
def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    # 1 where z > 0, otherwise 0
    return (z > 0).astype(float)


# -------------------------------------------------
# The neural network
# -------------------------------------------------
class NeuralNetwork:

    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        layer_sizes example: [4, 8, 8, 1]
        """
        self.lr = learning_rate
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            # He initialisation: good starting scale for ReLU networks
            W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """
        Forward pass. Saves every z and a, because backprop needs them.
          z = a_prev @ W + b
          a = relu(z)        (hidden layers)
          a = z              (last layer, no activation -> regression)
        """
        self.z_list = []
        self.a_list = [X]                     # a[0] is the input itself

        number_of_layers = len(self.weights)
        for i in range(number_of_layers):
            z = self.a_list[-1] @ self.weights[i] + self.biases[i]
            self.z_list.append(z)

            if i == number_of_layers - 1:     # output layer
                a = z
            else:                             # hidden layer
                a = relu(z)
            self.a_list.append(a)

        return self.a_list[-1]

    def compute_loss(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)

    def backward(self, y):
        """
        Backpropagation using the chain rule.
          Output layer:  delta = dL/dz = (2/m) * (y_hat - y)
          Hidden layer:  delta = (delta_next @ W_next^T) * relu'(z)
          Gradients:     dW = a_prev^T @ delta
                         db = sum of delta over samples
        """
        m = y.shape[0]
        y_hat = self.a_list[-1]
        number_of_layers = len(self.weights)

        grads_W = [None] * number_of_layers
        grads_b = [None] * number_of_layers

        # Error at the output layer
        delta = (2 / m) * (y_hat - y)

        # Walk backwards from the last layer to the first
        for i in reversed(range(number_of_layers)):
            a_prev = self.a_list[i]
            grads_W[i] = a_prev.T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                # pass the error back to the previous layer
                delta = (delta @ self.weights[i].T) * relu_derivative(self.z_list[i - 1])

        return grads_W, grads_b

    def update(self, grads_W, grads_b):
        """Gradient descent:  W = W - lr * dL/dW"""
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.lr * grads_W[i]
            self.biases[i] = self.biases[i] - self.lr * grads_b[i]

    def train(self, X, y, epochs=1000, print_every=100):
        loss_history = []
        for epoch in range(epochs):
            y_hat = self.forward(X)
            loss = self.compute_loss(y_hat, y)
            grads_W, grads_b = self.backward(y)
            self.update(grads_W, grads_b)

            loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"  epoch {epoch:5d}   loss = {loss:.6f}")
        return loss_history

    def predict(self, X):
        return self.forward(X)


# -------------------------------------------------
# Gradient check: compare backprop with a numerical estimate.
# If backprop is coded correctly, the two numbers are almost equal.
# -------------------------------------------------
def gradient_check(net, X, y, eps=1e-5):
    net.forward(X)
    grads_W, _ = net.backward(y)

    layer, row, col = 0, 0, 0                 # check one weight
    original = net.weights[layer][row, col]

    net.weights[layer][row, col] = original + eps
    loss_plus = net.compute_loss(net.forward(X), y)
    net.weights[layer][row, col] = original - eps
    loss_minus = net.compute_loss(net.forward(X), y)
    net.weights[layer][row, col] = original   # put it back

    numerical = (loss_plus - loss_minus) / (2 * eps)
    analytical = grads_W[layer][row, col]
    print("Gradient check")
    print(f"  backprop gradient : {analytical:.8f}")
    print(f"  numerical gradient: {numerical:.8f}")


# -------------------------------------------------
# Make a dataset: y depends on the 4 inputs in a non-linear way
# -------------------------------------------------
m = 500
X = np.random.uniform(-1, 1, (m, 4))
y = (np.sin(2 * X[:, 0]) + X[:, 1] ** 2 - 0.5 * X[:, 2] * X[:, 3]).reshape(-1, 1)

# Split into training (80%) and test (20%) data
split = int(0.8 * m)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------------------------
# Build, check and train
# -------------------------------------------------
net = NeuralNetwork([4, 8, 8, 1], learning_rate=0.05)
gradient_check(net, X_train, y_train)

print("\nTraining...")
losses = net.train(X_train, y_train, epochs=3000, print_every=500)

train_loss = net.compute_loss(net.predict(X_train), y_train)
test_loss = net.compute_loss(net.predict(X_test), y_test)
baseline = np.mean((y_test - y_train.mean()) ** 2)   # always guess the average

print(f"\nFinal train MSE : {train_loss:.5f}")
print(f"Final test MSE  : {test_loss:.5f}")
print(f"Baseline MSE    : {baseline:.5f}  (just predicting the mean)")

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(losses, color="tab:cyan")
axes[0].set_yscale("log")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss (MSE)")
axes[0].set_title("Training loss")

y_pred_test = net.predict(X_test)
axes[1].scatter(y_test, y_pred_test, s=12)
low, high = y_test.min(), y_test.max()
axes[1].plot([low, high], [low, high], "r--", label="Perfect prediction")
axes[1].set_xlabel("True y")
axes[1].set_ylabel("Predicted y")
axes[1].set_title("Test set: predicted vs true")
axes[1].legend()

plt.tight_layout()
plt.savefig("3_neural_network.png", dpi=120)
print("\nPlot saved as 3_neural_network.png")
plt.show()
