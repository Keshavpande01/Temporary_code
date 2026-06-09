# ============================================================
# Heart Disease Prediction using heart.csv
# Models:
# 1. Logistic Regression
# 2. Random Forest
# 3. XGBoost
#
# Evaluation:
# 10-Fold Cross Validation
# Accuracy mean and standard deviation
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "Heart.csv")

heart = pd.read_csv(file_path)

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)
print(heart.head())

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)
print(heart.shape)

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)
print(heart.columns)

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)
print(heart.dtypes)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
print(heart.isnull().sum())


# ============================================================
# 2. BASIC CLEANING
# ============================================================

# Remove unwanted index column if present
if "Unnamed: 0" in heart.columns:
    heart = heart.drop(columns=["Unnamed: 0"])

# Drop missing values
# Your previous dataset had missing values in Ca and Thal
heart = heart.dropna()

print("\n" + "=" * 70)
print("SHAPE AFTER CLEANING")
print("=" * 70)
print(heart.shape)


# ============================================================
# 3. TARGET COLUMN DETECTION
# ============================================================
# Some heart.csv files have target column as:
# target, AHD, HeartDisease, output, Disease

possible_targets = [
    "target", "Target",
    "AHD",
    "HeartDisease", "heart_disease",
    "output", "Disease"
]

target_column = None

for col in possible_targets:
    if col in heart.columns:
        target_column = col
        break

if target_column is None:
    raise ValueError("Target column not found. Check column names using print(heart.columns).")

print("\nTarget column used:", target_column)


# ============================================================
# 4. TARGET ENCODING
# ============================================================
# If target is Yes/No, convert it into 1/0

if heart[target_column].dtype == "object":
    heart[target_column] = heart[target_column].map({
        "No": 0,
        "Yes": 1,
        "no": 0,
        "yes": 1,
        "Normal": 0,
        "Disease": 1
    })

# If mapping created NaN due to different spelling, show error
if heart[target_column].isnull().sum() > 0:
    print("\nUnique target values:")
    print(heart[target_column].unique())
    raise ValueError("Target encoding failed. Check target values.")

print("\n" + "=" * 70)
print("TARGET VALUE COUNTS")
print("=" * 70)
print(heart[target_column].value_counts())


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(heart.describe())

print("\n" + "=" * 70)
print("CORRELATION WITH TARGET")
print("=" * 70)

numeric_cols_for_corr = heart.select_dtypes(include=["int64", "float64"]).columns

correlation_with_target = heart[numeric_cols_for_corr].corr()[target_column].sort_values(ascending=False)

print(correlation_with_target)


# ------------------------------------------------------------
# Target distribution plot
# ------------------------------------------------------------

plt.figure(figsize=(6, 4))
heart[target_column].value_counts().plot(kind="bar")
plt.xlabel("Heart Disease Class")
plt.ylabel("Count")
plt.title("Target Distribution")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# Correlation heatmap using matplotlib
# ------------------------------------------------------------

corr_matrix = heart[numeric_cols_for_corr].corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Histograms of numeric features
# ------------------------------------------------------------

numeric_cols = heart.select_dtypes(include=["int64", "float64"]).columns.tolist()

if target_column in numeric_cols:
    numeric_cols.remove(target_column)

heart[numeric_cols].hist(figsize=(14, 10), bins=20)
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.show()


# ============================================================
# 6. FEATURES AND TARGET
# ============================================================

X = heart.drop(columns=[target_column])
y = heart[target_column].astype(int)

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\n" + "=" * 70)
print("FEATURE TYPES")
print("=" * 70)
print("Numeric features:")
print(numeric_features)
print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 7. PREPROCESSING PIPELINE
# ============================================================
# Numeric features -> StandardScaler
# Categorical features -> OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)


# ============================================================
# 8. 10-FOLD CROSS VALIDATION SETUP
# ============================================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


def evaluate_model(model_name, model):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)
    print("Cross-validation scores:")
    print(scores)
    print("Mean Accuracy:", round(scores.mean(), 4))
    print("Standard Deviation:", round(scores.std(), 4))

    return {
        "Model": model_name,
        "Mean Accuracy": scores.mean(),
        "Std Accuracy": scores.std()
    }


results = []


# ============================================================
# 9. LOGISTIC REGRESSION MODEL
# ============================================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=5000))
    ]
)

results.append(
    evaluate_model(
        "Logistic Regression",
        logistic_model
    )
)


# ============================================================
# 10. RANDOM FOREST MODEL
# ============================================================

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42
        ))
    ]
)

results.append(
    evaluate_model(
        "Random Forest",
        random_forest_model
    )
)


# ============================================================
# 11. XGBOOST MODEL
# ============================================================

try:
    from xgboost import XGBClassifier

    xgboost_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            ))
        ]
    )

    results.append(
        evaluate_model(
            "XGBoost",
            xgboost_model
        )
    )

except Exception as e:
    print("\n" + "=" * 70)
    print("XGBoost not available")
    print("=" * 70)
    print("Install it using:")
    print("pip install xgboost")
    print("Error:", e)


# ============================================================
# 12. FINAL MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(results_df.sort_values(by="Mean Accuracy", ascending=False))


# ============================================================
# 13. FINAL EXAM COMMENT
# ============================================================

best_model = results_df.sort_values(by="Mean Accuracy", ascending=False).iloc[0]

print("\n" + "=" * 70)
print("FINAL CONCLUSION")
print("=" * 70)

print(f"""
Heart disease prediction was performed using heart.csv.

EDA was performed by checking:
1. Dataset shape
2. Column names and data types
3. Missing values
4. Target distribution
5. Summary statistics
6. Correlation with target
7. Histograms
8. Correlation heatmap

Feature engineering:
1. Missing values were removed.
2. Categorical variables were one-hot encoded.
3. Numerical variables were standardized.

Three machine learning models were developed:
1. Logistic Regression
2. Random Forest
3. XGBoost

All models were evaluated using 10-fold cross-validation with accuracy score.

Best Model:
{best_model["Model"]}

Mean Accuracy:
{best_model["Mean Accuracy"]:.4f}

Standard Deviation:
{best_model["Std Accuracy"]:.4f}
""")
