import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Split train into (train + blend-validation)
X_train, X_blend, y_train, y_blend = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Base models
model1 = DecisionTreeClassifier(max_depth=5, random_state=42)
model2 = SVC(probability=True, kernel="rbf", random_state=42)

# Train base models on training set
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# Predict probabilities on blend-validation set
blend_pred1 = model1.predict_proba(X_blend)[:, 1]
blend_pred2 = model2.predict_proba(X_blend)[:, 1]

# Stack predictions as features for meta-model
blend_features = np.column_stack((blend_pred1, blend_pred2))

# Meta-model (Logistic Regression)
meta_model = LogisticRegression()
meta_model.fit(blend_features, y_blend)

# Final prediction: use test set
test_pred1 = model1.predict_proba(X_test)[:, 1]
test_pred2 = model2.predict_proba(X_test)[:, 1]
test_features = np.column_stack((test_pred1, test_pred2))

y_pred = meta_model.predict(test_features)

print("Blending Accuracy:", accuracy_score(y_test, y_pred))
