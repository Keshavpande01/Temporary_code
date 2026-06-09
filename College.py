# ============================================================
# College.csv - Classification Practical
# Target: Private
# Models:
# Logistic Regression, Naive Bayes, SVM, KNN,
# Decision Tree, Random Forest, Gradient Boosting
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# 1. Load dataset
# ============================================================

try:
    from ISLP import load_data
    df = load_data("College")
except Exception:
    df = pd.read_csv("College.csv")


# ============================================================
# 2. Basic cleaning
# ============================================================

# Remove unnecessary index/id columns if present
for col in ["Unnamed: 0", "ID", "id", "index"]:
    if col in df.columns:
        df = df.drop(columns=[col])


# ============================================================
# 3. Basic EDA
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nTarget counts:")
print(df["Private"].value_counts())

print("\nSummary statistics:")
print(df.describe())


# ============================================================
# 4. Target encoding
# ============================================================
# Private:
# No  = 0
# Yes = 1

df["Private"] = df["Private"].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# 5. Define X and y
# ============================================================

X = df.drop(columns=["Private"])
y = df["Private"]

# Encode categorical feature columns if any
X = pd.get_dummies(X, drop_first=True)

print("\nFeature shape after encoding:")
print(X.shape)

print("\nFeature columns:")
print(X.columns)


# ============================================================
# 6. Train-test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# 7. Cross-validation setup
# ============================================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


# ============================================================
# 8. Models and parameter grids
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
            "model__gamma": ["scale", 0.001, 0.01, 0.1, 1]
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
            "min_samples_split": [2, 5, 10]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [1, 2, 3]
        }
    )
}


# ============================================================
# 9. Evaluation function
# ============================================================

def calculate_metrics(y_test, y_pred, y_prob=None):
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    sensitivity = recall_score(y_test, y_pred, zero_division=0)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = np.nan

    if y_prob is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

    return accuracy, precision, sensitivity, specificity, f1, roc_auc, cm


# ============================================================
# 10. GridSearchCV + evaluation
# ============================================================

results = []
best_models = {}

for name, (model, param_grid) in models.items():

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)

    y_prob = None
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy, precision, sensitivity, specificity, f1, roc_auc, cm = calculate_metrics(
        y_test,
        y_pred,
        y_prob
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Best Parameters:")
    print(grid.best_params_)

    print("\nBest CV Accuracy:", round(grid.best_score_, 4))
    print("Test Accuracy   :", round(accuracy, 4))
    print("Precision       :", round(precision, 4))
    print("Sensitivity     :", round(sensitivity, 4))
    print("Specificity     :", round(specificity, 4))
    print("F1-score        :", round(f1, 4))
    print("AUC             :", round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Public", "Private"]))

    results.append({
        "Model": name,
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
# 11. Final comparison
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
# 12. Best model
# ============================================================

best_row = results_df.iloc[0]

print("\n" + "=" * 100)
print("BEST MODEL")
print("=" * 100)

print("Best Model:", best_row["Model"])
print("Best Test Accuracy:", round(best_row["Test Accuracy"], 4))
print("Best Parameters:", best_row["Best Parameters"])


# ============================================================
# 13. ROC curve for best model
# ============================================================

best_model_name = best_row["Model"]
best_model = best_models[best_model_name]

if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{best_model_name} AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Sensitivity")
    plt.title("ROC Curve for Best Model")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# 14. Final conclusion
# ============================================================

print("""
Final Conclusion:

College.csv is mainly a classification dataset when Private is used as target.

Target variable:
Private

Classes:
No  = Public college
Yes = Private college

Classification models were trained and tuned using GridSearchCV with
10-fold cross-validation.

The models were evaluated using accuracy, precision, sensitivity, specificity,
F1-score, confusion matrix, ROC curve and AUC.

The best model was selected based on highest test accuracy.
""")