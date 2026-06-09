# ============================================================
# Universal ML Pipeline
# Automatically Detect Target Column
# Detect Regression or Classification Automatically
# If Classification: Encode Target Labels into Numbers
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)

from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_name = "Wage.csv"     # Change only this line

df = pd.read_csv(file_name)

print("=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)
print(df.head())

print("\nShape before cleaning:", df.shape)


# ============================================================
# 2. REMOVE UNNECESSARY INDEX COLUMNS
# ============================================================

for col in ["Unnamed: 0", "ID", "id", "index"]:
    if col in df.columns:
        df = df.drop(columns=[col])

print("\nShape after removing index columns:", df.shape)


# ============================================================
# 3. BASIC EDA
# ============================================================

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)
print(df.columns)

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)
print(df.dtypes)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
print(df.isnull().sum())

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)
print("Duplicate rows:", df.duplicated().sum())

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(df.describe(include="all"))


# ============================================================
# 4. AUTO TARGET COLUMN DETECTION
# ============================================================

possible_targets = [
    # Classification datasets
    "Purchase",        # Caravan, OJ
    "Direction",       # Weekly, Smarket
    "AHD",             # Heart
    "target",
    "Target",
    "class",
    "Class",
    "label",
    "Label",
    "status",
    "Status",
    "Disease",
    "disease",
    "output",
    "Outcome",
    "Species",
    "species",

    # Regression datasets
    "medv",            # Boston
    "mpg",             # Auto
    "wage",            # Wage
    "Balance",         # Credit
    "balance",
    "bikers",          # Bikeshare
    "time",
    "salary",
    "price",
    "Price"
]

target_col = None

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    print("\nColumns available in dataset:")
    print(df.columns)

    raise ValueError(
        "Target column not detected automatically. "
        "Please manually set target_col from the printed column names."
    )

print("\n" + "=" * 70)
print("AUTO DETECTED TARGET COLUMN")
print("=" * 70)
print("Detected target column:", target_col)


# ============================================================
# 5. TARGET ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TARGET ANALYSIS")
print("=" * 70)

print("Target column:", target_col)
print("Target dtype:", df[target_col].dtype)
print("Number of unique target values:", df[target_col].nunique())

print("\nUnique target values:")
print(df[target_col].unique()[:20])

print("\nTarget value counts:")
print(df[target_col].value_counts())


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[target_col])
y = df[target_col]


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("\n" + "=" * 70)
print("FEATURE TYPES")
print("=" * 70)
print("Numeric columns:")
print(numeric_cols)

print("\nCategorical columns:")
print(categorical_cols)

# Fill missing values in numeric columns
for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

# Fill missing values in categorical columns
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# Fill missing values in target
if y.isnull().sum() > 0:
    if y.dtype in ["int64", "float64"]:
        y = y.fillna(y.median())
    else:
        y = y.fillna(y.mode()[0])


# ============================================================
# 8. DETECT PROBLEM TYPE
# ============================================================
# Rule:
# Numeric target with many unique values = regression
# Text target or few unique values = classification

unique_count = y.nunique()

if y.dtype in ["int64", "float64"] and unique_count > 10:
    problem_type = "regression"
else:
    problem_type = "classification"

print("\n" + "=" * 70)
print("DETECTED PROBLEM TYPE")
print("=" * 70)
print("Problem type:", problem_type.upper())


# ============================================================
# 9. IF CLASSIFICATION, ENCODE TARGET INTO NUMBERS
# ============================================================

label_encoder = None

if problem_type == "classification":

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nClassification target encoded successfully.")

    print("\nOriginal classes:")
    print(label_encoder.classes_)

    print("\nEncoded classes:")
    for original_class, encoded_value in zip(
        label_encoder.classes_,
        range(len(label_encoder.classes_))
    ):
        print(original_class, "->", encoded_value)

    y = y_encoded

else:
    y = y.astype(float)


# ============================================================
# 10. ENCODE CATEGORICAL FEATURE COLUMNS
# ============================================================

X_encoded = pd.get_dummies(X, drop_first=True)

print("\n" + "=" * 70)
print("FEATURE MATRIX AFTER ENCODING")
print("=" * 70)

print("Feature shape:", X_encoded.shape)

print("\nFeature columns:")
print(X_encoded.columns)


# ============================================================
# 11. TRAIN-TEST SPLIT
# ============================================================

if problem_type == "classification":
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.30,
        random_state=42
    )

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ============================================================
# 12. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 13A. REGRESSION MODELS
# ============================================================

if problem_type == "regression":

    print("\n" + "=" * 70)
    print("RUNNING REGRESSION MODELS")
    print("=" * 70)

    regression_models = {
        "Linear Regression": (
            LinearRegression(),
            X_train_scaled,
            X_test_scaled
        ),

        "Ridge Regression": (
            Ridge(alpha=1.0),
            X_train_scaled,
            X_test_scaled
        ),

        "Lasso Regression": (
            Lasso(alpha=0.01, max_iter=10000),
            X_train_scaled,
            X_test_scaled
        ),

        "SVR RBF": (
            SVR(kernel="rbf", C=10, gamma="scale"),
            X_train_scaled,
            X_test_scaled
        ),

        "Decision Tree Regressor": (
            DecisionTreeRegressor(random_state=42),
            X_train,
            X_test
        ),

        "Random Forest Regressor": (
            RandomForestRegressor(n_estimators=200, random_state=42),
            X_train,
            X_test
        ),

        "Gradient Boosting Regressor": (
            GradientBoostingRegressor(random_state=42),
            X_train,
            X_test
        )
    }

    regression_results = []

    for model_name, item in regression_models.items():

        model, Xtr, Xte = item

        model.fit(Xtr, y_train)

        y_pred = model.predict(Xte)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("\n" + "=" * 70)
        print(model_name)
        print("=" * 70)

        print("MSE :", round(mse, 4))
        print("RMSE:", round(rmse, 4))
        print("R2  :", round(r2, 4))

        regression_results.append({
            "Model": model_name,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

    results_df = pd.DataFrame(regression_results)

    print("\n" + "=" * 70)
    print("FINAL REGRESSION MODEL COMPARISON")
    print("=" * 70)

    print(results_df.sort_values(by="RMSE"))

    best_model = results_df.sort_values(by="RMSE").iloc[0]

    print("\nBest Regression Model:")
    print(best_model)


# ============================================================
# 13B. CLASSIFICATION MODELS
# ============================================================

else:

    print("\n" + "=" * 70)
    print("RUNNING CLASSIFICATION MODELS")
    print("=" * 70)

    classification_models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000),
            X_train_scaled,
            X_test_scaled
        ),

        "Naive Bayes": (
            GaussianNB(),
            X_train_scaled,
            X_test_scaled
        ),

        "SVM RBF": (
            SVC(kernel="rbf", C=1, gamma="scale"),
            X_train_scaled,
            X_test_scaled
        ),

        "Decision Tree Classifier": (
            DecisionTreeClassifier(random_state=42),
            X_train,
            X_test
        ),

        "Random Forest Classifier": (
            RandomForestClassifier(n_estimators=200, random_state=42),
            X_train,
            X_test
        ),

        "Gradient Boosting Classifier": (
            GradientBoostingClassifier(random_state=42),
            X_train,
            X_test
        )
    }

    classification_results = []

    for model_name, item in classification_models.items():

        model, Xtr, Xte = item

        model.fit(Xtr, y_train)

        y_pred = model.predict(Xte)

        accuracy = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 70)
        print(model_name)
        print("=" * 70)

        print("Accuracy:", round(accuracy, 4))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        classification_results.append({
            "Model": model_name,
            "Accuracy": accuracy
        })

    results_df = pd.DataFrame(classification_results)

    print("\n" + "=" * 70)
    print("FINAL CLASSIFICATION MODEL COMPARISON")
    print("=" * 70)

    print(results_df.sort_values(by="Accuracy", ascending=False))

    best_model = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]

    print("\nBest Classification Model:")
    print(best_model)


# ============================================================
# 14. FINAL COMMENT
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMMENT")
print("=" * 70)

if problem_type == "regression":
    print("""
The target variable is continuous, so this dataset is treated as a regression problem.

The models were evaluated using:
1. MSE
2. RMSE
3. R2 score

The best regression model is the one with the lowest RMSE and highest R2 score.
""")

else:
    print("""
The target variable is categorical/discrete, so this dataset is treated as a classification problem.

The target labels were converted into numerical values using LabelEncoder.

The models were evaluated using:
1. Accuracy
2. Confusion matrix
3. Classification report

The best classification model is the one with the highest accuracy.

Note:
If the dataset is imbalanced, also check precision, recall, F1-score and AUC.
""")