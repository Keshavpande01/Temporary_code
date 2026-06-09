import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate 200 samples total (100 for train, 100 for test)
X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Convert to DataFrame for easier plotting with sns
train_df = pd.DataFrame(X_train, columns=['x1', 'x2'])
train_df['y'] = y_train


def plot_clf(model, df, grid_range, show_contours=False, show_support_vectors=False):
    # Get grid of values
    x1 = grid_range
    x2 = grid_range
    xx1, xx2 = np.meshgrid(x1, x2, sparse=False)
    Xgrid = np.stack((xx1.flatten(), xx2.flatten())).T

    # Get decision boundary values
    decision_boundary = model.predict(Xgrid)
    decision_boundary_grid = decision_boundary.reshape(len(x2), len(x1))

    # Get decision function values (for contours)
    decision_function = model.decision_function(Xgrid)
    decision_function_grid = decision_function.reshape(len(x2), len(x1))

    plt.figure(figsize=(8, 8))
    if show_contours:
        plt.contourf(x1, x2, decision_function_grid, alpha=0.3)
    plt.contour(x1, x2, decision_boundary_grid, colors='k', linewidths=1)

    sns.scatterplot(x='x1', y='x2', hue='y', data=df)

    if show_support_vectors:
        sv = model.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], color='red', marker='+', s=100, label='Support Vectors')

    plt.show()

# Setup for plots
grid = np.linspace(-2, 3, 100)

# 1. Linear Kernel (Should perform poorly)
lin_svc = SVC(kernel='linear').fit(X_train, y_train)
print(f"Linear Train Accuracy: {accuracy_score(y_train, lin_svc.predict(X_train))}")
plot_clf(lin_svc, train_df, grid, show_contours=True)

# 2. Radial (RBF) Kernel (Should perform significantly better)
rbf_svc = SVC(kernel='rbf', gamma=1).fit(X_train, y_train)
print(f"RBF Train Accuracy: {accuracy_score(y_train, rbf_svc.predict(X_train))}")
plot_clf(rbf_svc, train_df, grid, show_contours=True)

