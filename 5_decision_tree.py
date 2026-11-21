"""
PROJECT 5: DECISION TREE FROM SCRATCH
-------------------------------------
At every step the tree asks the yes/no question (feature <= threshold)
that makes the two groups as "pure" as possible.

Impurity measures:
  Entropy   H(S) = - sum( p_c * log2(p_c) )
  Gini      G(S) = 1 - sum( p_c^2 )
Information gain:
  IG = impurity(parent) - [ (n_left/n) * impurity(left) + (n_right/n) * impurity(right) ]

Part A: small "comedy show" dataset (13 rows) - print the tree
Part B: bigger noisy dataset - watch the tree OVERFIT as it gets deeper

Run:  python 5_decision_tree.py
Needs: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Impurity functions
# -------------------------------------------------
def entropy(y):
    counts = np.bincount(y)
    p = counts / len(y)
    total = 0.0
    for pi in p:
        if pi > 0:                     # log(0) is undefined, so skip zeros
            total -= pi * np.log2(pi)
    return total


def gini(y):
    counts = np.bincount(y)
    p = counts / len(y)
    return 1 - np.sum(p ** 2)


def information_gain(y, left_mask, impurity_function):
    y_left = y[left_mask]
    y_right = y[~left_mask]
    n = len(y)
    w_left = len(y_left) / n
    w_right = len(y_right) / n
    children = w_left * impurity_function(y_left) + w_right * impurity_function(y_right)
    return impurity_function(y) - children


# -------------------------------------------------
# A node of the tree
# -------------------------------------------------
class Node:
    def __init__(self):
        self.feature = None      # which column to ask about
        self.threshold = None    # ask: is value <= threshold ?
        self.left = None         # answer True
        self.right = None        # answer False
        self.prediction = None   # only set for leaf nodes
        self.impurity = 0.0
        self.samples = 0
        self.value = None        # class counts in this node

    def is_leaf(self):
        return self.prediction is not None


# -------------------------------------------------
# The decision tree
# -------------------------------------------------
class DecisionTree:

    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        if criterion == "gini":
            self.impurity_function = gini
        else:
            self.impurity_function = entropy
        self.criterion = criterion

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.root = self._grow(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        """Try every feature and every threshold; keep the one with the biggest gain."""
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]
        for feature in range(n_features):
            values = np.unique(X[:, feature])
            # Candidate thresholds = midpoints between neighbouring values
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                left_mask = X[:, feature] <= threshold
                gain = information_gain(y, left_mask, self.impurity_function)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _grow(self, X, y, depth):
        node = Node()
        node.samples = len(y)
        node.value = np.bincount(y, minlength=self.n_classes)
        node.impurity = self.impurity_function(y)

        # Stopping rules -> make a leaf
        pure = node.impurity == 0
        too_deep = self.max_depth is not None and depth >= self.max_depth
        too_small = len(y) < self.min_samples_split

        if pure or too_deep or too_small:
            node.prediction = int(np.argmax(node.value))
            return node

        feature, threshold, gain = self._best_split(X, y)
        if feature is None:               # no split improves anything
            node.prediction = int(np.argmax(node.value))
            return node

        node.feature = feature
        node.threshold = threshold
        left_mask = X[:, feature] <= threshold
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _predict_one(self, x, node):
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def print_tree(self, feature_names, class_names, node=None, indent=""):
        if node is None:
            node = self.root
        info = f"{self.criterion}={node.impurity:.3f}, samples={node.samples}, value={node.value.tolist()}"
        if node.is_leaf():
            print(f"{indent}-> predict {class_names[node.prediction]}   ({info})")
            return
        print(f"{indent}[{feature_names[node.feature]} <= {node.threshold}]   ({info})")
        print(f"{indent}  True:")
        self.print_tree(feature_names, class_names, node.left, indent + "    ")
        print(f"{indent}  False:")
        self.print_tree(feature_names, class_names, node.right, indent + "    ")


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# =================================================
# PART A: Comedy show dataset (should I go to the show?)
# Columns: Age, Experience, Rank, Nationality (UK=0, USA=1, N=2)
# Label:   0 = NO, 1 = YES
# =================================================
X_show = np.array([
    [36, 10, 9, 0],
    [42, 12, 4, 1],
    [23, 4, 6, 2],
    [52, 4, 4, 1],
    [43, 21, 8, 1],
    [44, 14, 5, 0],
    [66, 3, 7, 2],
    [35, 14, 9, 0],
    [52, 13, 7, 2],
    [35, 5, 9, 2],
    [24, 3, 5, 1],
    [18, 3, 7, 0],
    [45, 9, 9, 0],
])
y_show = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1])

feature_names = ["Age", "Experience", "Rank", "Nationality"]
class_names = ["NO", "YES"]

print("Hand calculation check on the full dataset:")
print(f"  entropy(root) = {entropy(y_show):.4f}")
print(f"  gini(root)    = {gini(y_show):.4f}")
mask = X_show[:, 2] <= 6.5
print(f"  information gain of 'Rank <= 6.5' (entropy) = "
      f"{information_gain(y_show, mask, entropy):.4f}")

tree = DecisionTree(criterion="gini")
tree.fit(X_show, y_show)
print("\nLearned tree (gini):")
tree.print_tree(feature_names, class_names)
print("Training accuracy:", accuracy(y_show, tree.predict(X_show)))

new_comedian = np.array([[40, 10, 7, 1]])
print("\nNew comedian: Age 40, Experience 10, Rank 7, USA ->",
      class_names[tree.predict(new_comedian)[0]])

# =================================================
# PART B: Overfitting experiment
# Two circles-ish classes with 15% of labels flipped (noise).
# =================================================
np.random.seed(3)
m = 600
X2 = np.random.uniform(-1, 1, (m, 2))
y2 = (X2[:, 0] ** 2 + X2[:, 1] ** 2 < 0.5).astype(int)
flip = np.random.rand(m) < 0.15
y2[flip] = 1 - y2[flip]

split = int(0.7 * m)
X_train, X_test = X2[:split], X2[split:]
y_train, y_test = y2[:split], y2[split:]

depths = range(1, 16)
train_acc = []
test_acc = []
for d in depths:
    t = DecisionTree(max_depth=d, criterion="entropy").fit(X_train, y_train)
    train_acc.append(accuracy(y_train, t.predict(X_train)))
    test_acc.append(accuracy(y_test, t.predict(X_test)))

print("\nDepth  train_acc  test_acc")
for d, tr, te in zip(depths, train_acc, test_acc):
    print(f"{d:5d}  {tr:9.3f}  {te:8.3f}")
best_depth = list(depths)[int(np.argmax(test_acc))]
print("Best depth on test data:", best_depth)

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Impurity curves for 2 classes
p = np.linspace(0.001, 0.999, 200)
axes[0].plot(p, -p * np.log2(p) - (1 - p) * np.log2(1 - p), label="Entropy")
axes[0].plot(p, 1 - p ** 2 - (1 - p) ** 2, label="Gini")
axes[0].set_xlabel("p (fraction of class 1)")
axes[0].set_ylabel("Impurity")
axes[0].set_title("Impurity is highest at a 50/50 mix")
axes[0].legend()

# Overfitting curve
axes[1].plot(list(depths), train_acc, "o-", label="Train")
axes[1].plot(list(depths), test_acc, "o-", label="Test")
axes[1].set_xlabel("max_depth")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Deeper tree -> overfitting")
axes[1].legend()

# Decision regions for the best depth
best_tree = DecisionTree(max_depth=best_depth, criterion="entropy").fit(X_train, y_train)
grid = np.linspace(-1, 1, 200)
gx, gy = np.meshgrid(grid, grid)
grid_points = np.column_stack([gx.ravel(), gy.ravel()])
regions = best_tree.predict(grid_points).reshape(gx.shape)
axes[2].contourf(gx, gy, regions, alpha=0.3, cmap="coolwarm")
axes[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", s=12, edgecolor="k")
axes[2].set_title(f"Decision regions (depth {best_depth})")

plt.tight_layout()
plt.savefig("5_decision_tree.png", dpi=120)
print("\nPlot saved as 5_decision_tree.png")
plt.show()
