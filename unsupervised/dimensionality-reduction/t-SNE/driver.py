import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Apply t-SNE (reduce to 2D)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot t-SNE results
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="viridis", s=50)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE on Iris Dataset")
plt.colorbar(label="Actual Labels")
plt.show()