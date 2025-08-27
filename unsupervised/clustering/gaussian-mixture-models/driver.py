import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Fit GMM with 3 components (since Iris has 3 classes)
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# Reduce dimensions for visualization (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot GMM clustering vs actual labels
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Predicted clusters
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=40)
axes[0].set_title("GMM Clustering (Predicted)")

# Actual labels
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=40)
axes[1].set_title("Actual Labels")

plt.show()