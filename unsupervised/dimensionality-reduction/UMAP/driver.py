import matplotlib.pyplot as plt
from sklearn import datasets
import umap.umap_ as umap

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Apply UMAP (reduce to 2D)
umap_model = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X)

# Plot UMAP results
plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="viridis", s=50)
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.title("UMAP on Iris Dataset")
plt.colorbar(label="Actual Labels")
plt.show()