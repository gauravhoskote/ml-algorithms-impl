import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Load dataset
iris = load_iris()
X = iris.data

# Perform hierarchical clustering (average linkage)
Z = linkage(X, method="average")

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode="lastp", p=12, show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram (Iris dataset)")
plt.xlabel("Cluster index")
plt.ylabel("Distance")
plt.show()

# Get flat cluster assignments (choose 3 clusters for Iris)
labels = fcluster(Z, t=3, criterion="maxclust")
print(labels[:20])  # show first 20 labels
