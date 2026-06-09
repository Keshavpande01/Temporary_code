# ============================================================
# Auto Dataset - Regression Practical
# Target: mpg
# Models:
# Linear Regression, Polynomial Regression, Ridge, Lasso,
# SVR, Decision Tree, Random Forest, Gradient Boosting
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. Load dataset
# ============================================================

try:
    from ISLP import load_data
    df = load_data("Auto")
except Exception:
    df = pd.read_csv("Auto.csv")

for col in ["Unnamed: 0", "ID", "id", "index", "name"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# horsepower may contain "?"
if "horsepower" in df.columns:
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")

df = df.dropna()


# ============================================================
# 2. Basic EDA
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

print("\nCorrelation with mpg:")
print(df.corr(numeric_only=True)["mpg"].sort_values(ascending=False))


# ============================================================
# 3. Basic plots
# ============================================================

for col in ["horsepower", "weight", "displacement"]:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[col], df["mpg"], alpha=0.7)
        plt.xlabel(col)
        plt.ylabel("mpg")
        plt.title(f"mpg vs {col}")
        plt.grid(True)
        plt.show()


# ============================================================
# 4. Define X and y
# ============================================================

X = df.drop(columns=["mpg"])
y = df["mpg"]

# origin is categorical in Auto dataset
if "origin" in X.columns:
    X["origin"] = X["origin"].astype("category")

X = pd.get_dummies(X, drop_first=True)

print("\nFeature shape after encoding:")
print(X.shape)

print("\nFeature columns:")
print(X.columns)


# ============================================================
# 5. Train-test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. Evaluation function
# ============================================================

def evaluate_model(model_name, model, X_train_data, X_test_data):
    model.fit(X_train_data, y_train)

    y_train_pred = model.predict(X_train_data)
    y_test_pred = model.predict(X_test_data)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)
    print("Train RMSE:", round(train_rmse, 4))
    print("Test RMSE :", round(test_rmse, 4))
    print("Train R2  :", round(train_r2, 4))
    print("Test R2   :", round(test_r2, 4))

    return {
        "Model": model_name,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train R2": train_r2,
        "Test R2": test_r2
    }


# ============================================================
# 7. Simple Linear Regression: mpg ~ horsepower
# ============================================================

X_hp = df[["horsepower"]]
y_hp = df["mpg"]

X_hp_train, X_hp_test, y_hp_train, y_hp_test = train_test_split(
    X_hp,
    y_hp,
    test_size=0.30,
    random_state=42
)

simple_lr = LinearRegression()
simple_lr.fit(X_hp_train, y_hp_train)

y_hp_pred = simple_lr.predict(X_hp_test)

print("\n" + "=" * 70)
print("Simple Linear Regression: mpg ~ horsepower")
print("=" * 70)
print("Intercept:", round(simple_lr.intercept_, 4))
print("Slope:", round(simple_lr.coef_[0], 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_hp_test, y_hp_pred)), 4))
print("R2:", round(r2_score(y_hp_test, y_hp_pred), 4))

plt.figure(figsize=(7, 5))
plt.scatter(X_hp_test, y_hp_test, label="Test Data")
plt.plot(X_hp_test, y_hp_pred, color="red", label="Regression Line")
plt.xlabel("horsepower")
plt.ylabel("mpg")
plt.title("Simple Linear Regression: mpg vs horsepower")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 8. Normal regression models
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
        DecisionTreeRegressor(random_state=42),
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
# 9. Polynomial Regression: mpg ~ horsepower + horsepower^2
# ============================================================

poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("linear", LinearRegression())
])

poly_model.fit(X_hp_train, y_hp_train)

y_poly_pred = poly_model.predict(X_hp_test)

print("\n" + "=" * 70)
print("Polynomial Regression: mpg ~ horsepower + horsepower^2")
print("=" * 70)
print("RMSE:", round(np.sqrt(mean_squared_error(y_hp_test, y_poly_pred)), 4))
print("R2:", round(r2_score(y_hp_test, y_poly_pred), 4))


# ============================================================
# 10. GridSearchCV for important models
# ============================================================

cv = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

grid_models = {
    "Ridge GridSearch": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge())
        ]),
        {
            "model__alpha": [0.01, 0.1, 1, 10, 100]
        }
    ),

    "Lasso GridSearch": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(max_iter=10000))
        ]),
        {
            "model__alpha": [0.001, 0.01, 0.1, 1, 10]
        }
    ),

    "SVR GridSearch": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR())
        ]),
        {
            "model__kernel": ["rbf"],
            "model__C": [1, 10, 100],
            "model__gamma": ["scale", 0.01, 0.1, 1]
        }
    ),

    "Random Forest GridSearch": (
        RandomForestRegressor(random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10]
        }
    ),

    "Gradient Boosting GridSearch": (
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [1, 2, 3]
        }
    )
}

grid_results = []

for name, item in grid_models.items():
    model, param_grid = item

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print("Best Parameters:")
    print(grid.best_params_)
    print("Best CV RMSE:", round(-grid.best_score_, 4))
    print("Test RMSE  :", round(rmse, 4))
    print("Test R2    :", round(r2, 4))

    grid_results.append({
        "Model": name,
        "Best CV RMSE": -grid.best_score_,
        "Test RMSE": rmse,
        "Test R2": r2,
        "Best Parameters": grid.best_params_
    })


# ============================================================
# 11. Final comparison
# ============================================================

results_df = pd.DataFrame(results)
grid_results_df = pd.DataFrame(grid_results)

print("\n" + "=" * 70)
print("NORMAL MODEL COMPARISON")
print("=" * 70)
print(results_df.sort_values(by="Test RMSE"))

print("\n" + "=" * 70)
print("GRIDSEARCH MODEL COMPARISON")
print("=" * 70)
print(grid_results_df.sort_values(by="Test RMSE"))

best_row = grid_results_df.sort_values(by="Test RMSE").iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL AFTER GRIDSEARCH")
print("=" * 70)
print("Best Model:", best_row["Model"])
print("Best Parameters:", best_row["Best Parameters"])
print("Test RMSE:", round(best_row["Test RMSE"], 4))
print("Test R2:", round(best_row["Test R2"], 4))


# ============================================================
# 12. Final conclusion
# ============================================================

print("""
Final Conclusion:

Auto is a regression dataset because mpg is continuous.

Target variable:
mpg

Important predictors:
horsepower, weight, displacement, cylinders, acceleration, year, origin

Models were evaluated using RMSE and R2 score.

RMSE measures prediction error.
R2 measures how much variation in mpg is explained by the model.

The best model is selected based on lowest test RMSE and highest test R2 score.
""")