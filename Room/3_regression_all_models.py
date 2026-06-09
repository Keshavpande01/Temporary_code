# ============================================================
# 3_regression_all_models.py
# Regression Models for Any CSV Dataset
#
# Models:
# Linear Regression
# Polynomial Regression
# Ridge Regression
# Lasso Regression
# SVR
# Decision Tree Regressor
# Random Forest Regressor
# Gradient Boosting Regressor
#
# Change only:
# file_name
# target_col
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. Load Dataset
# ============================================================

file_name = "your_dataset.csv"   # CHANGE THIS
target_col = "target"            # CHANGE THIS

df = pd.read_csv(file_name)


# ============================================================
# 2. Basic Cleaning
# ============================================================

for col in ["Unnamed: 0", "ID", "id", "index", "name"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Useful for Auto.csv where horsepower may contain "?"
if "horsepower" in df.columns:
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")


# ============================================================
# 3. Basic EDA
# ============================================================

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe(include="all"))


# ============================================================
# 4. Separate X and y
# ============================================================

df = df.dropna()

X = df.drop(columns=[target_col])
y = df[target_col].astype(float)


# ============================================================
# 5. Handle Missing Values and Encode Features
# ============================================================

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

X = pd.get_dummies(X, drop_first=True)

print("\n" + "=" * 70)
print("FEATURE MATRIX")
print("=" * 70)
print("Feature shape:", X.shape)
print("Target:", target_col)


# ============================================================
# 6. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# ============================================================
# 7. Evaluation Function
# ============================================================

def evaluate_regression(model_name, model, X_train_data, X_test_data):
    model.fit(X_train_data, y_train)

    y_train_pred = model.predict(X_train_data)
    y_test_pred = model.predict(X_test_data)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("Train RMSE:", round(train_rmse, 4))
    print("Test RMSE :", round(test_rmse, 4))
    print("Train MAE :", round(train_mae, 4))
    print("Test MAE  :", round(test_mae, 4))
    print("Train R2  :", round(train_r2, 4))
    print("Test R2   :", round(test_r2, 4))

    return {
        "Model": model_name,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train R2": train_r2,
        "Test R2": test_r2
    }


# ============================================================
# 8. Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 9. Normal Regression Models
# ============================================================

results = []

models = [
    (
        "Linear Regression",
        LinearRegression(),
        X_train_scaled,
        X_test_scaled
    ),

    (
        "Ridge Regression",
        Ridge(alpha=1.0),
        X_train_scaled,
        X_test_scaled
    ),

    (
        "Lasso Regression",
        Lasso(alpha=0.01, max_iter=10000),
        X_train_scaled,
        X_test_scaled
    ),

    (
        "SVR RBF",
        SVR(kernel="rbf", C=10, gamma="scale"),
        X_train_scaled,
        X_test_scaled
    ),

    (
        "Decision Tree Regressor",
        DecisionTreeRegressor(random_state=42),
        X_train,
        X_test
    ),

    (
        "Random Forest Regressor",
        RandomForestRegressor(n_estimators=200, random_state=42),
        X_train,
        X_test
    ),

    (
        "Gradient Boosting Regressor",
        GradientBoostingRegressor(random_state=42),
        X_train,
        X_test
    )
]

for model_name, model, Xtr, Xte in models:
    results.append(
        evaluate_regression(
            model_name,
            model,
            Xtr,
            Xte
        )
    )


# ============================================================
# 10. Polynomial Regression Degree 2
# ============================================================

poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

results.append(
    evaluate_regression(
        "Polynomial Regression Degree 2",
        poly_model,
        X_train,
        X_test
    )
)


# ============================================================
# 11. Normal Model Comparison
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("NORMAL MODEL COMPARISON")
print("=" * 100)

print(results_df.sort_values(by="Test RMSE"))


# ============================================================
# 12. GridSearchCV for Main Regression Models
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
            "model__gamma": ["scale", "auto", 0.01, 0.1, 1]
        }
    ),

    "Decision Tree GridSearch": (
        DecisionTreeRegressor(random_state=42),
        {
            "max_depth": [2, 3, 5, 7, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5]
        }
    ),

    "Random Forest GridSearch": (
        RandomForestRegressor(random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5]
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

for model_name, (model, param_grid) in grid_models.items():

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

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("Best Parameters:")
    print(grid.best_params_)

    print("\nBest CV RMSE:", round(-grid.best_score_, 4))
    print("Test RMSE  :", round(rmse, 4))
    print("Test MAE   :", round(mae, 4))
    print("Test R2    :", round(r2, 4))

    grid_results.append({
        "Model": model_name,
        "Best CV RMSE": -grid.best_score_,
        "Test RMSE": rmse,
        "Test MAE": mae,
        "Test R2": r2,
        "Best Parameters": grid.best_params_
    })


# ============================================================
# 13. GridSearch Model Comparison
# ============================================================

grid_results_df = pd.DataFrame(grid_results)

grid_results_df = grid_results_df.sort_values(
    by="Test RMSE"
)

print("\n" + "=" * 100)
print("GRIDSEARCH MODEL COMPARISON")
print("=" * 100)

print(grid_results_df)


# ============================================================
# 14. Best Model
# ============================================================

best_row = grid_results_df.iloc[0]

print("\n" + "=" * 100)
print("BEST MODEL AFTER GRIDSEARCH")
print("=" * 100)

print("Best Model:", best_row["Model"])
print("Best Parameters:", best_row["Best Parameters"])
print("Best CV RMSE:", round(best_row["Best CV RMSE"], 4))
print("Test RMSE:", round(best_row["Test RMSE"], 4))
print("Test MAE:", round(best_row["Test MAE"], 4))
print("Test R2:", round(best_row["Test R2"], 4))


# ============================================================
# 15. Final Comment
# ============================================================

print("""
Final Comment:

This is a regression problem because the target variable is continuous.

Models were evaluated using:
1. MSE
2. RMSE
3. MAE
4. R2 score

GridSearchCV was used to tune the main regression models.

The best model is selected based on the lowest test RMSE.
A good model should also have a high test R2 score.
""")
