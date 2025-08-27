import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Apply LDA (reduce to 2D)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Plot LDA results
plt.figure(figsize=(8, 6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap="viridis", s=50)
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.title("LDA on Iris Dataset")
plt.colorbar(label="Actual Labels")
plt.show()