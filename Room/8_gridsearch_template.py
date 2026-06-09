# ============================================================
# 8_gridsearch_template.py
# General GridSearchCV Template
#
# Works for:
# 1. Classification
# 2. Regression
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

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


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

# Useful for Weekly/Smarket
if "Today" in df.columns:
    df = df.drop(columns=["Today"])

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
# 5. Handle Missing Values
# ============================================================

df = df.dropna()

X = df.drop(columns=[target_col])
y = df[target_col]

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

X = pd.get_dummies(X, drop_first=True)

print("\nFeature shape after encoding:")
print(X.shape)


# ============================================================
# 6A. Classification GridSearchCV
# ============================================================

if problem_type == "classification":

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    print("\nTarget encoding:")
    for original_class, encoded_value in zip(
        label_encoder.classes_,
        range(len(label_encoder.classes_))
    ):
        print(original_class, "->", encoded_value)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    scoring = "accuracy"

    models = {

        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000))
            ]),
            {
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__penalty": ["l1", "l2"],
                "model__solver": ["liblinear"]
            }
        ),

        "Linear SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="linear"))
            ]),
            {
                "model__C": [0.01, 0.1, 1, 10, 100]
            }
        ),

        "RBF SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf"))
            ]),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", "auto", 0.01, 0.1, 1]
            }
        ),

        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "criterion": ["gini", "entropy"],
                "max_depth": [2, 3, 5, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5]
            }
        ),

        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, None],
                "min_samples_split": [2, 5, 10]
            }
        ),

        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [1, 2, 3]
            }
        )
    }

    results = []

    for model_name, (model, param_grid) in models.items():

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 70)
        print(model_name)
        print("=" * 70)

        print("Best Parameters:")
        print(grid.best_params_)

        print("\nBest CV Accuracy:", round(grid.best_score_, 4))
        print("Test Accuracy   :", round(acc, 4))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        results.append({
            "Model": model_name,
            "Best CV Accuracy": grid.best_score_,
            "Test Accuracy": acc,
            "Best Parameters": grid.best_params_
        })

    results_df = pd.DataFrame(results).sort_values(
        by="Test Accuracy",
        ascending=False
    )

    print("\n" + "=" * 100)
    print("FINAL CLASSIFICATION GRIDSEARCH COMPARISON")
    print("=" * 100)

    print(results_df)

    best_row = results_df.iloc[0]

    print("\nBEST MODEL:")
    print(best_row["Model"])
    print("BEST PARAMETERS:")
    print(best_row["Best Parameters"])
    print("BEST TEST ACCURACY:", round(best_row["Test Accuracy"], 4))


# ============================================================
# 6B. Regression GridSearchCV
# ============================================================

elif problem_type == "regression":

    y = y.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    cv = KFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    scoring = "neg_root_mean_squared_error"

    models = {

        "Ridge Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge())
            ]),
            {
                "model__alpha": [0.01, 0.1, 1, 10, 100]
            }
        ),

        "Lasso Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(max_iter=10000))
            ]),
            {
                "model__alpha": [0.001, 0.01, 0.1, 1, 10]
            }
        ),

        "SVR RBF": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf"))
            ]),
            {
                "model__C": [1, 10, 100],
                "model__gamma": ["scale", "auto", 0.01, 0.1, 1]
            }
        ),

        "Decision Tree Regressor": (
            DecisionTreeRegressor(random_state=42),
            {
                "max_depth": [2, 3, 5, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5]
            }
        ),

        "Random Forest Regressor": (
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, None],
                "min_samples_split": [2, 5, 10]
            }
        ),

        "Gradient Boosting Regressor": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [1, 2, 3]
            }
        )
    }

    results = []

    for model_name, (model, param_grid) in models.items():

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
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

        results.append({
            "Model": model_name,
            "Best CV RMSE": -grid.best_score_,
            "Test RMSE": rmse,
            "Test MAE": mae,
            "Test R2": r2,
            "Best Parameters": grid.best_params_
        })

    results_df = pd.DataFrame(results).sort_values(
        by="Test RMSE"
    )

    print("\n" + "=" * 100)
    print("FINAL REGRESSION GRIDSEARCH COMPARISON")
    print("=" * 100)

    print(results_df)

    best_row = results_df.iloc[0]

    print("\nBEST MODEL:")
    print(best_row["Model"])
    print("BEST PARAMETERS:")
    print(best_row["Best Parameters"])
    print("TEST RMSE:", round(best_row["Test RMSE"], 4))
    print("TEST R2:", round(best_row["Test R2"], 4))


# ============================================================
# 7. Invalid Problem Type
# ============================================================

else:
    raise ValueError("problem_type must be either 'classification' or 'regression'.")


# ============================================================
# 8. Final Comment
# ============================================================

print("""
Final Comment:

GridSearchCV tries all combinations of hyperparameters in the param_grid.

For classification:
Best model is selected using highest test accuracy.

For regression:
Best model is selected using lowest test RMSE.

Important:
SVM uses C and gamma.
Ridge/Lasso use alpha.
Tree models use max_depth, min_samples_split and min_samples_leaf.
Random Forest and Boosting use n_estimators.
""")
