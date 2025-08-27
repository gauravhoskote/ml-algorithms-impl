import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Generate synthetic 2D classification dataset
X, y = make_classification(
    n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42
)

# Fit kNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # k=5
knn.fit(X, y)
y_pred = knn.predict(X)

# Accuracy
print("kNN Classification Accuracy:", accuracy_score(y, y_pred))

# Confusion matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)

# Plot decision boundary
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, edgecolor='k', s=50)
plt.title("kNN Classification Decision Boundary (k=5)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# Generate regression dataset
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Fit kNN regressor
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_reg, y_reg)
y_pred_reg = knn_reg.predict(X_reg)

# Metrics
print("kNN Regression MSE:", mean_squared_error(y_reg, y_pred_reg))

# Plot actual vs predicted
plt.figure(figsize=(8,6))
plt.scatter(X_reg, y_reg, color='blue', label='Actual')
plt.scatter(X_reg, y_pred_reg, color='red', label='Predicted')
plt.title("kNN Regression (k=5)")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()