# ============================================================
# 7_loocv.py
# LOOCV Template
#
# Supports:
# 1. Classification LOOCV using Logistic Regression
# 2. Regression LOOCV using Linear Regression
# 3. Regression LOOCV using Polynomial Regression
#
# Change only:
# file_name
# target_col
# problem_type
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression


# ============================================================
# 1. User Settings
# ============================================================

file_name = "your_dataset.csv"        # CHANGE THIS
target_col = "target"                 # CHANGE THIS

# Use:
# problem_type = "classification"
# or
# problem_type = "regression"

problem_type = "classification"       # CHANGE THIS


# ============================================================
# 2. Load Dataset
# ============================================================

df = pd.read_csv(file_name)


# ============================================================
# 3. Basic Cleaning
# ============================================================

for col in ["Unnamed: 0", "ID", "id", "index", "name"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Useful for Auto.csv
if "horsepower" in df.columns:
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")


# ============================================================
# 4. Basic EDA
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

print("\nTarget column:", target_col)


# ============================================================
# 5. Separate X and y
# ============================================================

df = df.dropna()

# Important for Weekly/Smarket:
# Today directly decides Direction, so remove Today if present.

drop_cols = []

if "Today" in df.columns:
    drop_cols.append("Today")

X = df.drop(columns=[target_col] + drop_cols)
y = df[target_col]


# ============================================================
# 6. Optional Dataset-Specific Feature Selection
# ============================================================
# For Weekly/Smarket common LOOCV question, use Lag1 and Lag2 only.
# Uncomment these lines if needed:
#
# X = df[["Lag1", "Lag2"]]
# y = df[target_col]
#
# For Auto polynomial LOOCV, use horsepower only:
#
# X = df[["horsepower"]]
# y = df[target_col]


# ============================================================
# 7. Handle Missing Values and Encoding
# ============================================================

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

X = pd.get_dummies(X, drop_first=True)

print("\nFeature shape:")
print(X.shape)


# ============================================================
# 8. LOOCV Setup
# ============================================================

loo = LeaveOneOut()


# ============================================================
# 9A. Classification LOOCV
# ============================================================

if problem_type == "classification":

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nTarget encoding:")
    for original_class, encoded_value in zip(
        label_encoder.classes_,
        range(len(label_encoder.classes_))
    ):
        print(original_class, "->", encoded_value)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ])

    scores = cross_val_score(
        model,
        X,
        y_encoded,
        cv=loo,
        scoring="accuracy"
    )

    loocv_accuracy = scores.mean()
    loocv_error = 1 - loocv_accuracy

    print("\n" + "=" * 70)
    print("LOOCV CLASSIFICATION RESULT")
    print("=" * 70)

    print("Model: Logistic Regression")
    print("LOOCV Accuracy:", round(loocv_accuracy, 4))
    print("LOOCV Error   :", round(loocv_error, 4))


# ============================================================
# 9B. Regression LOOCV
# ============================================================

elif problem_type == "regression":

    y = y.astype(float)

    # ------------------------------------------------------------
    # Linear Regression LOOCV
    # ------------------------------------------------------------

    linear_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    linear_mse_scores = -cross_val_score(
        linear_model,
        X,
        y,
        cv=loo,
        scoring="neg_mean_squared_error"
    )

    linear_mse = linear_mse_scores.mean()
    linear_rmse = np.sqrt(linear_mse)

    print("\n" + "=" * 70)
    print("LOOCV REGRESSION RESULT - LINEAR REGRESSION")
    print("=" * 70)

    print("Linear Regression LOOCV MSE :", round(linear_mse, 4))
    print("Linear Regression LOOCV RMSE:", round(linear_rmse, 4))


    # ------------------------------------------------------------
    # Polynomial Regression Degree 2 LOOCV
    # ------------------------------------------------------------

    poly2_model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    poly2_mse_scores = -cross_val_score(
        poly2_model,
        X,
        y,
        cv=loo,
        scoring="neg_mean_squared_error"
    )

    poly2_mse = poly2_mse_scores.mean()
    poly2_rmse = np.sqrt(poly2_mse)

    print("\n" + "=" * 70)
    print("LOOCV REGRESSION RESULT - POLYNOMIAL DEGREE 2")
    print("=" * 70)

    print("Polynomial Degree 2 LOOCV MSE :", round(poly2_mse, 4))
    print("Polynomial Degree 2 LOOCV RMSE:", round(poly2_rmse, 4))


    # ------------------------------------------------------------
    # Polynomial Regression Degree 3 LOOCV
    # ------------------------------------------------------------

    poly3_model = Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    poly3_mse_scores = -cross_val_score(
        poly3_model,
        X,
        y,
        cv=loo,
        scoring="neg_mean_squared_error"
    )

    poly3_mse = poly3_mse_scores.mean()
    poly3_rmse = np.sqrt(poly3_mse)

    print("\n" + "=" * 70)
    print("LOOCV REGRESSION RESULT - POLYNOMIAL DEGREE 3")
    print("=" * 70)

    print("Polynomial Degree 3 LOOCV MSE :", round(poly3_mse, 4))
    print("Polynomial Degree 3 LOOCV RMSE:", round(poly3_rmse, 4))


    # ------------------------------------------------------------
    # Final Comparison
    # ------------------------------------------------------------

    comparison = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Polynomial Degree 2",
            "Polynomial Degree 3"
        ],
        "LOOCV MSE": [
            linear_mse,
            poly2_mse,
            poly3_mse
        ],
        "LOOCV RMSE": [
            linear_rmse,
            poly2_rmse,
            poly3_rmse
        ]
    })

    comparison = comparison.sort_values(by="LOOCV RMSE")

    print("\n" + "=" * 70)
    print("LOOCV REGRESSION COMPARISON")
    print("=" * 70)

    print(comparison)

    best_model = comparison.iloc[0]

    print("\nBest model:")
    print(best_model["Model"])
    print("Best LOOCV RMSE:", round(best_model["LOOCV RMSE"], 4))


# ============================================================
# 10. Invalid Problem Type
# ============================================================

else:
    raise ValueError("problem_type must be either 'classification' or 'regression'.")


# ============================================================
# 11. Final Comment
# ============================================================

print("""
Final Comment:

LOOCV means Leave-One-Out Cross Validation.

If there are n observations, LOOCV trains the model n times.
Each time, one observation is used as test data and the remaining n-1
observations are used as training data.

Classification:
Use LOOCV accuracy and LOOCV error.

Regression:
Use LOOCV MSE and LOOCV RMSE.

LOOCV is best for small or medium datasets.
Avoid LOOCV for very large datasets because it is slow.
""")
