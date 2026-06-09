# ============================================================
# Simulated Two-Class Nonlinear Dataset
# SVM with Polynomial Kernel and RBF Kernel
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ============================================================
# 1. Generate simulated nonlinear data
# ============================================================
# 100 training samples
# 100 test samples
# noise = 0.3

X_train, y_train = make_moons(
    n_samples=100,
    noise=0.3,
    random_state=1
)

X_test, y_test = make_moons(
    n_samples=100,
    noise=0.3,
    random_state=2
)

print("Training data shape:", X_train.shape)
print("Testing data shape :", X_test.shape)


# ============================================================
# 2. Convert training data into DataFrame for easy plotting
# ============================================================

df_train = pd.DataFrame({
    "x1": X_train[:, 0],
    "x2": X_train[:, 1],
    "y": y_train
})

df_test = pd.DataFrame({
    "x1": X_test[:, 0],
    "x2": X_test[:, 1],
    "y": y_test
})


# ============================================================
# 3. Plot training data
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    df_train[df_train["y"] == 0]["x1"],
    df_train[df_train["y"] == 0]["x2"],
    label="Class 0"
)

plt.scatter(
    df_train[df_train["y"] == 1]["x1"],
    df_train[df_train["y"] == 1]["x2"],
    label="Class 1"
)

plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.title("Training Data - Nonlinear Two-Class Dataset")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 4. Function to plot decision boundary
# ============================================================

def plot_svm_decision_boundary(model, X, y, title, show_support_vectors=True):
    # Grid range
    x1_min, x1_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    x2_min, x2_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8

    x1_grid = np.linspace(x1_min, x1_max, 300)
    x2_grid = np.linspace(x2_min, x2_max, 300)

    xx1, xx2 = np.meshgrid(x1_grid, x2_grid)

    X_grid = np.c_[xx1.ravel(), xx2.ravel()]

    # Predicted class on grid
    decision_boundary = model.predict(X_grid)
    decision_boundary = decision_boundary.reshape(xx1.shape)

    # Decision function for margin/contour
    decision_function = model.decision_function(X_grid)
    decision_function = decision_function.reshape(xx1.shape)

    plt.figure(figsize=(8, 6))

    # Plot decision regions
    plt.contourf(
        xx1,
        xx2,
        decision_boundary,
        alpha=0.25
    )

    # Plot decision boundary and margins
    plt.contour(
        xx1,
        xx2,
        decision_function,
        levels=[-1, 0, 1],
        linestyles=["--", "-", "--"]
    )

    # Plot data points
    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        label="Class 0",
        edgecolor="k"
    )

    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        label="Class 1",
        edgecolor="k"
    )

    # Show support vectors
    if show_support_vectors:
        plt.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=180,
            facecolors="none",
            edgecolors="black",
            linewidths=2,
            label="Support Vectors"
        )

    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# 5. Polynomial Kernel SVM
# ============================================================
# degree > 1 as asked in question

poly_svm = SVC(
    kernel="poly",
    degree=3,
    C=1,
    gamma="scale",
    coef0=1,
    random_state=1
)

poly_svm.fit(X_train, y_train)

y_train_pred_poly = poly_svm.predict(X_train)
y_test_pred_poly = poly_svm.predict(X_test)

poly_train_accuracy = accuracy_score(y_train, y_train_pred_poly)
poly_test_accuracy = accuracy_score(y_test, y_test_pred_poly)

print("\n" + "=" * 70)
print("Polynomial Kernel SVM")
print("=" * 70)

print("Training Accuracy:", round(poly_train_accuracy, 4))
print("Test Accuracy    :", round(poly_test_accuracy, 4))
print("Number of Support Vectors:", poly_svm.n_support_)

plot_svm_decision_boundary(
    poly_svm,
    X_train,
    y_train,
    "Polynomial Kernel SVM Decision Boundary",
    show_support_vectors=True
)


# ============================================================
# 6. Radial Kernel SVM / RBF Kernel SVM
# ============================================================

rbf_svm = SVC(
    kernel="rbf",
    C=1,
    gamma="scale",
    random_state=1
)

rbf_svm.fit(X_train, y_train)

y_train_pred_rbf = rbf_svm.predict(X_train)
y_test_pred_rbf = rbf_svm.predict(X_test)

rbf_train_accuracy = accuracy_score(y_train, y_train_pred_rbf)
rbf_test_accuracy = accuracy_score(y_test, y_test_pred_rbf)

print("\n" + "=" * 70)
print("Radial / RBF Kernel SVM")
print("=" * 70)

print("Training Accuracy:", round(rbf_train_accuracy, 4))
print("Test Accuracy    :", round(rbf_test_accuracy, 4))
print("Number of Support Vectors:", rbf_svm.n_support_)

plot_svm_decision_boundary(
    rbf_svm,
    X_train,
    y_train,
    "Radial Basis Function / RBF Kernel SVM Decision Boundary",
    show_support_vectors=True
)


# ============================================================
# 7. Compare both models
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Polynomial Kernel SVM",
        "RBF Kernel SVM"
    ],
    "Training Accuracy": [
        poly_train_accuracy,
        rbf_train_accuracy
    ],
    "Test Accuracy": [
        poly_test_accuracy,
        rbf_test_accuracy
    ]
})

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(comparison)

best_model = comparison.loc[comparison["Test Accuracy"].idxmax()]

print("\nBest model based on test accuracy:")
print(best_model)


# ============================================================
# 8. Plot test data with best model decision boundary
# ============================================================

if best_model["Model"] == "Polynomial Kernel SVM":
    final_model = poly_svm
else:
    final_model = rbf_svm

plot_svm_decision_boundary(
    final_model,
    X_test,
    y_test,
    "Best Model Decision Boundary on Test Data",
    show_support_vectors=False
)


# ============================================================
# 9. Final conclusion
# ============================================================

print("""
Final Comment:

The make_moons dataset creates a two-class problem with nonlinear separation.

A polynomial kernel SVM and an RBF kernel SVM were trained on the data.
Both kernels can create nonlinear decision boundaries.

The better model is selected based on test accuracy, not only training accuracy.
If training accuracy is very high but test accuracy is low, then the model is overfitting.

Usually, for moon-shaped nonlinear data, the RBF kernel performs well because it
can create flexible curved decision boundaries.
""")