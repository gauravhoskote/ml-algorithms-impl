# Voting Classifier Example
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Feature scaling (helps LogisticRegression & KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Base learners
clf1 = LogisticRegression(max_iter=5000, random_state=42)
clf2 = KNeighborsClassifier(n_neighbors=5)
clf3 = DecisionTreeClassifier(max_depth=4, random_state=42)

# Voting Classifier (Hard voting)
voting_hard = VotingClassifier(
    estimators=[("lr", clf1), ("knn", clf2), ("dt", clf3)],
    voting="hard"
)
voting_hard.fit(X_train, y_train)
y_pred_hard = voting_hard.predict(X_test)

# Voting Classifier (Soft voting) - requires predict_proba
voting_soft = VotingClassifier(
    estimators=[("lr", clf1), ("knn", clf2), ("dt", clf3)],
    voting="soft"
)
voting_soft.fit(X_train, y_train)
y_pred_soft = voting_soft.predict(X_test)

# Accuracy
print("Hard Voting Accuracy:", accuracy_score(y_test, y_pred_hard))
print("Soft Voting Accuracy:", accuracy_score(y_test, y_pred_soft))
