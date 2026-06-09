# ============================================================
# A1 - make_hastie_10_2
# Boosting + Bagging + SVM using GridSearchCV param_grid
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_hastie_10_2
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 1. Generate dataset
# ============================================================

X, y = make_hastie_10_2(
    n_samples=3000,
    random_state=42
)

print("Dataset shape:", X.shape)
print("Classes:", np.unique(y))


# ============================================================
# 2. Train-test split
# First 2000 = train
# Remaining 1000 = test
# ============================================================

X_train = X[:2000]
y_train = y[:2000]

X_test = X[2000:]
y_test = y[2000:]

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ============================================================
# 3. Evaluation function
# ============================================================

def evaluate_model(model_name, model):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("Training Accuracy:", round(train_acc, 4))
    print("Test Accuracy    :", round(test_acc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    return {
        "Model": model_name,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc
    }


results = []


# ============================================================
# 4. Gradient Boosting with param_grid
# Decision stump = max_depth = 1
# ============================================================

gb_model = GradientBoostingClassifier(
    random_state=42
)

gb_param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.5, 1.0],
    "max_depth": [1]
}

gb_grid = GridSearchCV(
    estimator=gb_model,
    param_grid=gb_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

gb_grid.fit(X_train, y_train)

print("\n" + "=" * 70)
print("Gradient Boosting GridSearchCV")
print("=" * 70)

print("Best Parameters:", gb_grid.best_params_)
print("Best CV Accuracy:", round(gb_grid.best_score_, 4))

best_gb = gb_grid.best_estimator_

results.append(
    evaluate_model(
        "Best Gradient Boosting Classifier",
        best_gb
    )
)


# ============================================================
# 5. Bagging Classifier with param_grid
# ============================================================

try:
    bagging_model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        random_state=42
    )

    bagging_param_grid = {
        "n_estimators": [50, 100, 200],
        "estimator__max_depth": [1, 2, 3, None],
        "max_samples": [0.5, 0.7, 1.0]
    }

except TypeError:
    # For older sklearn versions
    bagging_model = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(random_state=42),
        random_state=42
    )

    bagging_param_grid = {
        "n_estimators": [50, 100, 200],
        "base_estimator__max_depth": [1, 2, 3, None],
        "max_samples": [0.5, 0.7, 1.0]
    }


bagging_grid = GridSearchCV(
    estimator=bagging_model,
    param_grid=bagging_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

bagging_grid.fit(X_train, y_train)

print("\n" + "=" * 70)
print("Bagging GridSearchCV")
print("=" * 70)

print("Best Parameters:", bagging_grid.best_params_)
print("Best CV Accuracy:", round(bagging_grid.best_score_, 4))

best_bagging = bagging_grid.best_estimator_

results.append(
    evaluate_model(
        "Best Bagging Classifier",
        best_bagging
    )
)


# ============================================================
# 6. SVM with param_grid
# ============================================================

svm_model = SVC(
    random_state=42
)

svm_param_grid = {
    "kernel": ["rbf"],
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1, 1]
}

svm_grid = GridSearchCV(
    estimator=svm_model,
    param_grid=svm_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

print("\n" + "=" * 70)
print("SVM GridSearchCV")
print("=" * 70)

print("Best Parameters:", svm_grid.best_params_)
print("Best CV Accuracy:", round(svm_grid.best_score_, 4))

best_svm = svm_grid.best_estimator_

results.append(
    evaluate_model(
        "Best SVM RBF Classifier",
        best_svm
    )
)


# ============================================================
# 7. Final comparison
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(results_df.sort_values(by="Test Accuracy", ascending=False))

best_model = results_df.sort_values(by="Test Accuracy", ascending=False).iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Best Model:", best_model["Model"])
print("Best Test Accuracy:", round(best_model["Test Accuracy"], 4))


# ============================================================
# 8. Plot comparison
# ============================================================

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Test Accuracy"])
plt.ylabel("Test Accuracy")
plt.title("Model Comparison After GridSearchCV")
plt.xticks(rotation=30, ha="right")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 9. Final conclusion
# ============================================================

print("""
Final Conclusion:

GridSearchCV was used to tune hyperparameters for Gradient Boosting, Bagging,
and SVM.

For Gradient Boosting, n_estimators and learning_rate were tuned while keeping
max_depth = 1 to use decision stumps.

For Bagging, n_estimators, max_samples, and tree depth were tuned.

For SVM, C and gamma were tuned using the RBF kernel.

The best model is selected based on highest test accuracy.
""")