"""
PROJECT 1: LINEAR REGRESSION FROM ZERO
--------------------------------------
Goal: find the straight line  y = b0 + b1 * x  that best fits some data.

We solve it TWO ways and check that both give the same answer:
  1) Normal equation   : beta = (X^T X)^-1 X^T y      (exact, one line)
  2) Gradient descent  : beta = beta - lr * gradient  (step by step)

Run:  python 1_linear_regression.py
Needs: numpy, matplotlib   (pip install numpy matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Step 1: Make some fake data
# The "true" line is y = 3 + 2.5x, plus some random noise.
# -------------------------------------------------
np.random.seed(0)                     # same random numbers every run
n = 100                               # number of data points

x = np.random.rand(n) * 10            # 100 numbers between 0 and 10
noise = np.random.randn(n) * 2        # random noise
y = 3 + 2.5 * x + noise               # target values

# -------------------------------------------------
# Step 2: Build the design matrix X
# We add a column of 1s so the model can learn the intercept b0.
# Each row looks like: [1, x_i]
# -------------------------------------------------
ones = np.ones(n)
X = np.column_stack([ones, x])        # shape (100, 2)

# -------------------------------------------------
# Method 1: Normal equation
# beta = (X^T X)^-1 X^T y
# -------------------------------------------------
XtX = X.T @ X                         # (2, 2) matrix
Xty = X.T @ y                         # (2,) vector
beta_exact = np.linalg.inv(XtX) @ Xty

print("Normal equation result")
print("  intercept b0 =", round(beta_exact[0], 4))
print("  slope     b1 =", round(beta_exact[1], 4))

# -------------------------------------------------
# Method 2: Gradient descent
# Cost (mean squared error):  J = (1/n) * sum( (y - X beta)^2 )
# Gradient:                   dJ/dbeta = -(2/n) * X^T (y - X beta)
# Update rule:                beta = beta - lr * gradient
# -------------------------------------------------
def mse(X, y, beta):
    """Mean squared error between predictions and real values."""
    predictions = X @ beta
    errors = y - predictions
    return np.mean(errors ** 2)


beta = np.zeros(2)                    # start with b0 = 0, b1 = 0
learning_rate = 0.01
epochs = 2000
cost_history = []

for epoch in range(epochs):
    predictions = X @ beta
    errors = y - predictions
    gradient = -(2 / n) * (X.T @ errors)
    beta = beta - learning_rate * gradient

    cost_history.append(mse(X, y, beta))

    if epoch % 400 == 0:
        print(f"  epoch {epoch:4d}  cost = {cost_history[-1]:.4f}")

print("\nGradient descent result")
print("  intercept b0 =", round(beta[0], 4))
print("  slope     b1 =", round(beta[1], 4))

# -------------------------------------------------
# Step 3: Compare the two methods
# -------------------------------------------------
difference = np.abs(beta - beta_exact).max()
print("\nBiggest difference between the two methods:", f"{difference:.6f}")
print("Final MSE (normal equation):", round(mse(X, y, beta_exact), 4))

# R^2 score: 1.0 = perfect fit, 0.0 = no better than predicting the mean
y_pred = X @ beta_exact
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot
print("R^2 score:", round(r2, 4))

# -------------------------------------------------
# Step 4: Plot the results
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: data and both fitted lines
x_line = np.linspace(0, 10, 100)
axes[0].scatter(x, y, color="tab:cyan", label="Data points")
axes[0].plot(x_line, beta_exact[0] + beta_exact[1] * x_line,
             color="red", linewidth=2, label="Normal equation")
axes[0].plot(x_line, beta[0] + beta[1] * x_line,
             color="black", linestyle="--", label="Gradient descent")
axes[0].set_xlabel("feature x")
axes[0].set_ylabel("target y")
axes[0].set_title("Linear regression fit")
axes[0].legend()

# Right plot: how the cost dropped during gradient descent
axes[1].plot(cost_history)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE cost")
axes[1].set_title("Gradient descent: cost vs epochs")
axes[1].set_yscale("log")

plt.tight_layout()
plt.savefig("1_linear_regression.png", dpi=120)
print("\nPlot saved as 1_linear_regression.png")
plt.show()
