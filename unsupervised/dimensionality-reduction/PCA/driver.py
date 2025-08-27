import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA-transformed data
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=50)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")
plt.colorbar(label="Actual Labels")
plt.show()