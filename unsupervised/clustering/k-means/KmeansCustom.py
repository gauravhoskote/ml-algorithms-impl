import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class KMeansCustom:
    def __init__(self, K=3, max_iters=100):
        self.K = K
        self.max_iters = max_iters

    def fit(self, X):
        n_samples, n_features = X.shape

        random_idx = np.random.choice(n_samples, self.K, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)

            # compute new centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.K)])

            # check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.labels_ = labels

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
