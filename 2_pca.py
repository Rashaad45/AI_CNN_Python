"""
PROJECT 2: PRINCIPAL COMPONENT ANALYSIS (PCA)
---------------------------------------------
Goal: find the directions in which the data spreads out the most,
then squash the data onto the most important directions.

Steps:
  1) Center the data          (subtract the mean)
  2) Covariance matrix        C = (1/(n-1)) * Xc^T Xc
  3) Eigen-decomposition      C v = lambda v
  4) Sort eigenvalues big -> small
  5) Project                  Z = Xc @ W_k   (keep top k eigenvectors)

Run:  python 2_pca.py
Needs: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# -------------------------------------------------
# Part A: 2D example you can SEE
# -------------------------------------------------
# Make 500 points that are stretched along a diagonal line.
n = 500
mean = [0, 0]
cov_true = [[3.0, 2.0],
            [2.0, 2.0]]
X = np.random.multivariate_normal(mean, cov_true, n)   # shape (500, 2)

# Step 1: Center the data
X_mean = X.mean(axis=0)
X_centered = X - X_mean

# Step 2: Covariance matrix (2 x 2)
# Same as np.cov(X_centered.T)
C = (X_centered.T @ X_centered) / (n - 1)
print("Covariance matrix:")
print(np.round(C, 3))

# Step 3: Eigen-decomposition
# eigh is used because the covariance matrix is symmetric.
eigenvalues, eigenvectors = np.linalg.eigh(C)

# Step 4: Sort from largest to smallest eigenvalue
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]      # each COLUMN is one eigenvector

print("\nEigenvalues (variance along each component):", np.round(eigenvalues, 3))
print("Eigenvectors (columns):")
print(np.round(eigenvectors, 3))

# How much of the total spread does each component explain?
explained = eigenvalues / eigenvalues.sum()
for i in range(len(explained)):
    print(f"  PC{i+1} explains {explained[i]*100:.1f}% of the variance")

# Step 5: Project onto the first principal component (2D -> 1D)
k = 1
W_k = eigenvectors[:, :k]                  # shape (2, 1)
Z = X_centered @ W_k                       # shape (500, 1)

# Rebuild the points from 1 number each to see what we lost
X_rebuilt = Z @ W_k.T + X_mean
reconstruction_error = np.mean((X - X_rebuilt) ** 2)
print(f"\nCompressed 2 numbers -> {k} number per point")
print("Mean reconstruction error:", round(reconstruction_error, 4))

# -------------------------------------------------
# Check our answer against SVD (another way to do PCA)
# Xc = U S V^T  ->  columns of V are the principal components
# -------------------------------------------------
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
eigenvalues_svd = S ** 2 / (n - 1)
print("\nEigenvalues from SVD:", np.round(eigenvalues_svd, 3), "(should match above)")

# -------------------------------------------------
# Part B: Higher-dimensional example (5D -> 2D)
# 5 features, but really only 2 "hidden" factors drive them.
# -------------------------------------------------
hidden = np.random.randn(300, 2)                       # 2 true factors
mixing = np.random.randn(2, 5)                         # mix into 5 features
X5 = hidden @ mixing + 0.1 * np.random.randn(300, 5)   # small noise

X5_centered = X5 - X5.mean(axis=0)
C5 = np.cov(X5_centered.T)
vals5, vecs5 = np.linalg.eigh(C5)
order5 = np.argsort(vals5)[::-1]
vals5 = vals5[order5]
vecs5 = vecs5[:, order5]

explained5 = vals5 / vals5.sum()
print("\n5D data - variance explained by each component:")
for i in range(5):
    print(f"  PC{i+1}: {explained5[i]*100:5.1f}%")
print(f"Top 2 components keep {explained5[:2].sum()*100:.1f}% of the information")

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: original data with principal component arrows
axes[0].scatter(X[:, 0], X[:, 1], s=8, alpha=0.5, color="tab:cyan", label="Original data")
for i in range(2):
    # arrow length = 2 * standard deviation along that direction
    direction = eigenvectors[:, i] * 2 * np.sqrt(eigenvalues[i])
    axes[0].annotate("", xy=X_mean + direction, xytext=X_mean,
                     arrowprops=dict(color="red", width=2.5, headwidth=10))
    axes[0].text(*(X_mean + direction * 1.1), f"PC{i+1}", color="red", fontsize=12)
axes[0].set_title("Data and principal components")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")
axes[0].axis("equal")
axes[0].legend()

# Plot 2: original vs data rebuilt from 1 component
axes[1].scatter(X[:, 0], X[:, 1], s=8, alpha=0.3, color="tab:cyan", label="Original")
axes[1].scatter(X_rebuilt[:, 0], X_rebuilt[:, 1], s=8, color="red", label="From PC1 only")
axes[1].set_title("Projection onto PC1 (2D -> 1D)")
axes[1].axis("equal")
axes[1].legend()

# Plot 3: scree plot for the 5D example
axes[2].bar(range(1, 6), explained5 * 100, color="tab:blue")
axes[2].plot(range(1, 6), np.cumsum(explained5) * 100, "ro-", label="Cumulative")
axes[2].set_xlabel("Principal component")
axes[2].set_ylabel("% variance explained")
axes[2].set_title("Scree plot (5D data)")
axes[2].legend()

plt.tight_layout()
plt.savefig("2_pca.png", dpi=120)
print("\nPlot saved as 2_pca.png")
plt.show()
