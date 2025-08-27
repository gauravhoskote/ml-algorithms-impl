import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Scale features
X = StandardScaler().fit_transform(X)

# Build autoencoder
input_dim = X.shape[1]
encoding_dim = 2  # reduce to 2D for visualization

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(8, activation='relu')(input_layer)
encoded = layers.Dense(encoding_dim, activation='linear')(encoded)

decoded = layers.Dense(8, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = models.Model(input_layer, decoded)
encoder = models.Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder
autoencoder.fit(X, X, epochs=100, batch_size=16, verbose=0)

# Encode data
X_encoded = encoder.predict(X)

# Plot encoded (2D representation)
plt.figure(figsize=(8, 6))
plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=y, cmap="viridis", s=50)
plt.xlabel("Encoded Dimension 1")
plt.ylabel("Encoded Dimension 2")
plt.title("Autoencoder (2D Latent Representation of Iris)")
plt.colorbar(label="Actual Labels")
plt.show()
