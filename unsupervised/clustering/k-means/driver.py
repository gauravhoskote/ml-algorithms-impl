import numpy as np
from sklearn.datasets import load_iris
from scipy.stats import mode
import matplotlib.pyplot as plt
from KmeansCustom import KMeansCustom

# Step 1: Load Iris dataset
iris = load_iris()
X = iris.data
y_true = iris.target  # true labels (0=setosa, 1=versicolor, 2=virginica)

# Step 2: Run clustering
kmeans = KMeansCustom(K=3)
kmeans.fit(X)
y_pred = kmeans.labels_

# Step 3: Align clusters to true labels
labels_map = {}
for cluster in range(3):
    mask = (y_pred == cluster)
    if np.any(mask):
        labels_map[cluster] = mode(y_true[mask], keepdims=True)[0][0]

# remap predicted labels
y_pred_aligned = np.array([labels_map[label] for label in y_pred])

# Step 4: Evaluate accuracy
accuracy = np.mean(y_pred_aligned == y_true)
print("Cluster-to-label mapping:", labels_map)
print("Clustering Accuracy:", accuracy)

# Step 5: Visualization
plt.figure(figsize=(10,4))

# Plot predicted clusters
plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c=y_pred, cmap="viridis")
plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1], c="red", marker="x", s=200)
plt.title("Predicted Clusters")

# Plot true labels
plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c=y_true, cmap="viridis")
plt.title("True Iris Species")

plt.show()