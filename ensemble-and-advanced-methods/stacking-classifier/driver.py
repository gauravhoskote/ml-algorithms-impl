from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Base models
base_estimators = [
    ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("svm", SVC(probability=True, kernel="rbf", random_state=42))
]

# Meta model (stacking final estimator)
stack_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(),
    passthrough=True
)

# Train & predict
stack_model.fit(X_train, y_train)
y_pred = stack_model.predict(X_test)

print("Stacking Classifier Accuracy:", accuracy_score(y_test, y_pred))
