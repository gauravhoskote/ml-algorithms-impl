import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Fit Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf_clf.fit(X, y)
y_pred = rf_clf.predict(X)

# Accuracy
print("Random Forest Classification Accuracy:", accuracy_score(y, y_pred))

# Feature importance
feature_importance = pd.Series(rf_clf.feature_importances_, index=iris.feature_names)
print("\nFeature Importance:\n", feature_importance)

# Plot feature importance
feature_importance.sort_values().plot(kind='barh', figsize=(8,5))
plt.title("Random Forest Feature Importance (Classification)")
plt.show()