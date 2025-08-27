import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate 2D binary classification dataset
X, y = make_classification(
    n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42
)

# Linear SVM
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X, y)
y_pred = linear_svm.predict(X)

# Accuracy
print("Linear SVM Accuracy:", accuracy_score(y, y_pred))

# Plot decision boundary
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = linear_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, edgecolor='k', s=50)
plt.title("Linear SVM Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# RBF Kernel
rbf_svm = SVC(kernel='rbf', C=1.0, gamma=0.5)
rbf_svm.fit(X, y)

# Polynomial Kernel
poly_svm = SVC(kernel='poly', degree=3, C=1.0)
poly_svm.fit(X, y)

# Function to plot decision boundary
def plot_svm(svm_model, X, y, title):
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_svm(rbf_svm, X, y, "SVM with RBF Kernel")
plot_svm(poly_svm, X, y, "SVM with Polynomial Kernel (degree=3)")

