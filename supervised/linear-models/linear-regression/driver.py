import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Fit Linear Regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Print metrics
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y, y_pred))
print("R^2 Score:", r2_score(y, y_pred))

# Plot actual vs predicted
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression Example")
plt.legend()
plt.show()
