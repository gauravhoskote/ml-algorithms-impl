import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# Generate synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=5, noise=20, random_state=42)

# Initialize models
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Fit models
ridge.fit(X, y)
lasso.fit(X, y)
elastic.fit(X, y)

# Predictions
y_ridge = ridge.predict(X)
y_lasso = lasso.predict(X)
y_elastic = elastic.predict(X)

# Print metrics
print("Ridge MSE:", mean_squared_error(y, y_ridge))
print("Lasso MSE:", mean_squared_error(y, y_lasso))
print("ElasticNet MSE:", mean_squared_error(y, y_elastic))

# Compare coefficients
plt.figure(figsize=(10,6))
plt.plot(ridge.coef_, marker='o', label='Ridge')
plt.plot(lasso.coef_, marker='x', label='Lasso')
plt.plot(elastic.coef_, marker='s', label='ElasticNet')
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title("Ridge vs Lasso vs ElasticNet Coefficients")
plt.legend()
plt.show()
