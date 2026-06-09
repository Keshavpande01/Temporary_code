# ============================================================
# OJ Dataset - SVM Classification Practical
# ============================================================



import numpy as np
import pandas as pd

from ISLP import load_data

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------
# Step 1: Load OJ dataset
# ------------------------------------------------------------

OJ = load_data("OJ")

print(OJ.head())
print(OJ.shape)
print(OJ.info())

# ------------------------------------------------------------
# Step 2: Define target and features
# ------------------------------------------------------------

# Target variable
y = OJ["Purchase"]

# Feature variables
X = OJ.drop(columns=["Purchase"])

# Convert categorical variables into dummy variables
X = pd.get_dummies(X, drop_first=True)

print("Feature shape after dummy encoding:", X.shape)

# ------------------------------------------------------------
# (a) Train-test split
# Training data = 800 observations
# Testing data = remaining observations
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=800,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ============================================================
# (b) Linear Support Vector Classifier with C = 0.01
# ============================================================

linear_svc_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", LinearSVC(C=0.01, max_iter=10000, random_state=42))
])

linear_svc_model.fit(X_train, y_train)

# Predictions
y_train_pred = linear_svc_model.predict(X_train)
y_test_pred = linear_svc_model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nLinearSVC with C = 0.01")
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy :", test_accuracy)

# ============================================================
# (c) GridSearchCV to choose optimal C
# Values: 0.01, 1, 10
# ============================================================

param_grid = {
    "svc__C": [0.01, 1, 10]
}

grid_search = GridSearchCV(
    estimator=linear_svc_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

print("\nGridSearchCV Results")
print("Best C:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# ============================================================
# (d) Training and test error using new optimal C
# ============================================================

best_linear_model = grid_search.best_estimator_

y_train_best_pred = best_linear_model.predict(X_train)
y_test_best_pred = best_linear_model.predict(X_test)

best_train_accuracy = accuracy_score(y_train, y_train_best_pred)
best_test_accuracy = accuracy_score(y_test, y_test_best_pred)

best_train_error = 1 - best_train_accuracy
best_test_error = 1 - best_test_accuracy

print("\nBest LinearSVC Model")
print("Training Accuracy:", best_train_accuracy)
print("Testing Accuracy :", best_test_accuracy)
print("Training Error   :", best_train_error)
print("Testing Error    :", best_test_error)

# ============================================================
# (e) Support Vector Machine with Radial Kernel
# Default gamma
# ============================================================

radial_svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", gamma="scale", random_state=42))
])

radial_svm_model.fit(X_train, y_train)

y_train_radial_pred = radial_svm_model.predict(X_train)
y_test_radial_pred = radial_svm_model.predict(X_test)

radial_train_accuracy = accuracy_score(y_train, y_train_radial_pred)
radial_test_accuracy = accuracy_score(y_test, y_test_radial_pred)

print("\nRadial Kernel SVM")
print("Training Accuracy:", radial_train_accuracy)
print("Testing Accuracy :", radial_test_accuracy)

# ============================================================
# Final Comparison
# ============================================================

results = pd.DataFrame({
    "Model": [
        "LinearSVC C=0.01",
        "Best LinearSVC using GridSearchCV",
        "Radial Kernel SVM"
    ],
    "Train Accuracy": [
        train_accuracy,
        best_train_accuracy,
        radial_train_accuracy
    ],
    "Test Accuracy": [
        test_accuracy,
        best_test_accuracy,
        radial_test_accuracy
    ],
    "Train Error": [
        1 - train_accuracy,
        best_train_error,
        1 - radial_train_accuracy
    ],
    "Test Error": [
        1 - test_accuracy,
        best_test_error,
        1 - radial_test_accuracy
    ]
})

print("\nFinal Comparison")
print(results)

best_model = results.loc[results["Test Accuracy"].idxmax()]

print("\nBest approach based on test accuracy:")
print(best_model)