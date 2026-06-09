# ============================================================
# 2_classification_all_models.py
# Classification Models for Any CSV Dataset
#
# Models:
# Logistic Regression
# Naive Bayes
# Linear SVM
# RBF SVM
# KNN
# Decision Tree
# Random Forest
# Gradient Boosting
#
# Change only:
# file_name
# target_col
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


# ============================================================
# 1. Load Dataset
# ============================================================

file_name = "your_dataset.csv"     # CHANGE THIS
target_col = "target"              # CHANGE THIS

df = pd.read_csv(file_name)


# ============================================================
# 2. Basic Cleaning
# ============================================================

for col in ["Unnamed: 0", "ID", "id", "index"]:
    if col in df.columns:
        df = df.drop(columns=[col])


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

print("\nTarget counts:")
print(df[target_col].value_counts())


# ============================================================
# 4. Separate X and y
# ============================================================

X = df.drop(columns=[target_col])
y = df[target_col]


# ============================================================
# 5. Handle Missing Values
# ============================================================

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

if y.isnull().sum() > 0:
    y = y.fillna(y.mode()[0])


# ============================================================
# 6. Encode Features and Target
# ============================================================

X = pd.get_dummies(X, drop_first=True)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\n" + "=" * 70)
print("TARGET ENCODING")
print("=" * 70)

for original_class, encoded_value in zip(
    label_encoder.classes_,
    range(len(label_encoder.classes_))
):
    print(original_class, "->", encoded_value)

print("\nFeature shape after encoding:")
print(X.shape)


# ============================================================
# 7. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# 8. Cross Validation
# ============================================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


# ============================================================
# 9. Models and Parameter Grids
# ============================================================

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

    "Naive Bayes": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", GaussianNB())
        ]),
        {}
    ),

    "Linear SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="linear", probability=True))
        ]),
        {
            "model__C": [0.01, 0.1, 1, 10, 100]
        }
    ),

    "RBF SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True))
        ]),
        {
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", "auto", 0.01, 0.1, 1]
        }
    ),

    "KNN": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [2, 3, 4, 5, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5]
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


# ============================================================
# 10. Metric Function
# ============================================================

def get_metrics(y_test, y_pred, y_prob=None):
    accuracy = accuracy_score(y_test, y_pred)

    if len(np.unique(y_test)) == 2:
        precision = precision_score(y_test, y_pred, zero_division=0)
        sensitivity = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        cm = confusion_matrix(y_test, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        else:
            specificity = np.nan

        roc_auc = np.nan
        if y_prob is not None:
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

    else:
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        sensitivity = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        specificity = np.nan
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        roc_auc = np.nan

    return accuracy, precision, sensitivity, specificity, f1, roc_auc


# ============================================================
# 11. Train Models Using GridSearchCV
# ============================================================

results = []

for model_name, (model, param_grid) in models.items():

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    y_prob = None
    if len(np.unique(y)) == 2 and hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy, precision, sensitivity, specificity, f1, roc_auc = get_metrics(
        y_test,
        y_pred,
        y_prob
    )

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("Best Parameters:")
    print(grid.best_params_)

    print("\nBest CV Accuracy:", round(grid.best_score_, 4))
    print("Test Accuracy   :", round(accuracy, 4))
    print("Precision       :", round(precision, 4))
    print("Sensitivity     :", round(sensitivity, 4))
    print("Specificity     :", round(specificity, 4) if not np.isnan(specificity) else "NA")
    print("F1-score        :", round(f1, 4))
    print("AUC             :", round(roc_auc, 4) if not np.isnan(roc_auc) else "NA")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    results.append({
        "Model": model_name,
        "Best CV Accuracy": grid.best_score_,
        "Test Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1-score": f1,
        "AUC": roc_auc,
        "Best Parameters": grid.best_params_
    })


# ============================================================
# 12. Final Model Comparison
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Test Accuracy",
    ascending=False
)

print("\n" + "=" * 100)
print("FINAL MODEL COMPARISON")
print("=" * 100)

print(results_df)


# ============================================================
# 13. Best Model
# ============================================================

best_row = results_df.iloc[0]

print("\n" + "=" * 100)
print("BEST MODEL")
print("=" * 100)

print("Best Model:", best_row["Model"])
print("Best Test Accuracy:", round(best_row["Test Accuracy"], 4))
print("Best Parameters:", best_row["Best Parameters"])


# ============================================================
# 14. Final Comment
# ============================================================

print("""
Final Comment:

This is a classification problem because the target variable is categorical.

The target labels were converted into numerical values using LabelEncoder.

Models were tuned using GridSearchCV with 10-fold cross-validation.

The best model was selected based on highest test accuracy.

For binary classification, accuracy, precision, sensitivity, specificity,
F1-score and AUC are reported.

For multiclass classification, weighted precision, recall and F1-score are used.
""")
