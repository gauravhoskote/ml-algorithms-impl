from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Generate regression dataset
X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=25, random_state=42)

# Bagging Regressor
bag_reg = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=50,
    max_samples=0.8,
    bootstrap=True,
    random_state=42
)

bag_reg.fit(X_reg, y_reg)
y_pred_reg = bag_reg.predict(X_reg)

print("Bagging Regressor MSE:", mean_squared_error(y_reg, y_pred_reg))

# Plot regression fit
plt.figure(figsize=(8,6))
plt.scatter(X_reg, y_reg, color='blue', label='Actual')
plt.scatter(X_reg, y_pred_reg, color='red', label='Predicted')
plt.title("Bagging Regressor (Decision Trees)")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()
