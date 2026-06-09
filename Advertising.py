# ============================================================
# Income2 Dataset - Multiple Regression
# Income ~ Education + Seniority
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ISLP import load_data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. Load dataset
# ============================================================

try:
    Income2 = load_data("Income2")
except Exception:
    Income2 = pd.read_csv("Income2.csv")

print("\nFirst 5 rows:")
print(Income2.head())

print("\nShape:")
print(Income2.shape)

print("\nColumns:")
print(Income2.columns)

print("\nMissing values:")
print(Income2.isnull().sum())

print("\nSummary statistics:")
print(Income2.describe())


# ============================================================
# 2. Define X and y
# ============================================================

X = Income2[["Education", "Seniority"]]
y = Income2["Income"]


# ============================================================
# 3. Train-test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# ============================================================
# 4. Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. Evaluation function
# ============================================================

def evaluate_model(model_name, model, X_train_data, X_test_data):
    model.fit(X_train_data, y_train)

    y_pred = model.predict(X_test_data)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    return {
        "Model": model_name,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 6. Models
# ============================================================

results = []

results.append(
    evaluate_model(
        "Multiple Linear Regression",
        LinearRegression(),
        X_train_scaled,
        X_test_scaled
    )
)

results.append(
    evaluate_model(
        "Ridge Regression",
        Ridge(alpha=1.0),
        X_train_scaled,
        X_test_scaled
    )
)

results.append(
    evaluate_model(
        "Lasso Regression",
        Lasso(alpha=0.01, max_iter=10000),
        X_train_scaled,
        X_test_scaled
    )
)

results.append(
    evaluate_model(
        "SVR RBF",
        SVR(kernel="rbf", C=10, gamma="scale"),
        X_train_scaled,
        X_test_scaled
    )
)

results.append(
    evaluate_model(
        "Decision Tree Regressor",
        DecisionTreeRegressor(max_depth=4, random_state=42),
        X_train,
        X_test
    )
)

results.append(
    evaluate_model(
        "Random Forest Regressor",
        RandomForestRegressor(n_estimators=200, random_state=42),
        X_train,
        X_test
    )
)

results.append(
    evaluate_model(
        "Gradient Boosting Regressor",
        GradientBoostingRegressor(random_state=42),
        X_train,
        X_test
    )
)


# ============================================================
# 7. Polynomial Regression
# ============================================================

poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("linear", LinearRegression())
])

results.append(
    evaluate_model(
        "Polynomial Regression Degree 2",
        poly_model,
        X_train,
        X_test
    )
)


# ============================================================
# 8. Final comparison
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("Final Model Comparison")
print("=" * 70)

print(results_df.sort_values(by="RMSE"))


# ============================================================
# 9. 2D Scatter Plots
# ============================================================

plt.figure(figsize=(7, 5))
plt.scatter(Income2["Education"], Income2["Income"])
plt.xlabel("Education")
plt.ylabel("Income")
plt.title("Income vs Education")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(Income2["Seniority"], Income2["Income"])
plt.xlabel("Seniority")
plt.ylabel("Income")
plt.title("Income vs Seniority")
plt.grid(True)
plt.show()


# ============================================================
# 10. 3D Plot
# ============================================================

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    Income2["Education"],
    Income2["Seniority"],
    Income2["Income"]
)

ax.set_xlabel("Education")
ax.set_ylabel("Seniority")
ax.set_zlabel("Income")
ax.set_title("Income2: Education + Seniority vs Income")

plt.show()