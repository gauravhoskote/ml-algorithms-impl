import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y_true = iris.target

# Run DBSCAN
dbscan = DBSCAN(eps=0.6, min_samples=5)  # tune eps & min_samples for better clustering
y_pred = dbscan.fit_predict(X)

# Plot comparison
plt.figure(figsize=(12, 5))

# Actual labels
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", marker="o")
plt.title("Actual Iris Species")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

# DBSCAN clusters
plt.subplot(1, 2, 2)
# Noise points are labeled as -1 by DBSCAN
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="viridis", marker="o")
plt.title("DBSCAN Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.tight_layout()
plt.show()
