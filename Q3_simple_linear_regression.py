# ============================================================
# Simple Linear Regression on Simulated Data
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# Set seed for reproducibility
# ============================================================

rng = np.random.default_rng(seed=1)


# ============================================================
# (a) Create vector X with 200 observations from N(0, 1)
# ============================================================

X = rng.normal(loc=0, scale=1, size=200)


# ============================================================
# (b) Create noise vector e from N(0, 0.25)
# ============================================================
# Important:
# Variance = 0.25
# Standard deviation = sqrt(0.25) = 0.5

e = rng.normal(loc=0, scale=np.sqrt(0.25), size=200)


# ============================================================
# (c) Generate y according to:
# y = -1.1 + 0.6X + e
# ============================================================

y = -1.1 + 0.6 * X + e

# Length of y = 200
# theta_0 = -1.1
# theta_1 = 0.6

print("Length of X:", len(X))
print("Length of e:", len(e))
print("Length of y:", len(y))

print("\nTrue theta_0 / intercept:", -1.1)
print("True theta_1 / slope:", 0.6)


# ============================================================
# (d) Scatter plot between X and y
# ============================================================

plt.figure(figsize=(7, 5))
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Scatter Plot of X vs y")
plt.grid(True)
plt.show()

# Comment:
# The scatter plot shows an approximately linear positive relationship
# between X and y. As X increases, y also tends to increase.
# The points do not lie exactly on a straight line because random noise e
# has been added.


# ============================================================
# (e) Fit least squares linear regression model
# using 70-30 train-test split
# ============================================================

# sklearn expects X as 2D, so reshape X
X_reshaped = X.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped,
    y,
    test_size=0.30,
    random_state=1
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# ============================================================
# Print model results
# ============================================================

print("\nEstimated intercept theta_0:")
print(model.intercept_)

print("\nEstimated slope theta_1:")
print(model.coef_[0])

print("\nMean Squared Error:")
print(mean_squared_error(y_test, y_pred))

print("\nR2 Score:")
print(r2_score(y_test, y_pred))


# ============================================================
# Scatter plot of X_test and y_test
# Regression line in red color
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(X_test, y_test, label="Test Data")

# Sort X_test for smooth regression line
sorted_index = np.argsort(X_test.ravel())
X_test_sorted = X_test[sorted_index]
y_pred_sorted = y_pred[sorted_index]

plt.plot(
    X_test_sorted,
    y_pred_sorted,
    color="red",
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("X_test")
plt.ylabel("y_test")
plt.title("Linear Regression on Test Data")
plt.legend()
plt.grid(True)
plt.show()