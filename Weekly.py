# ============================================================
# Weekly Dataset - Function Based Classification Pipeline
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# ============================================================
# 1. Load dataset
# ============================================================

def load_weekly_data():
    try:
        from ISLP import load_data
        df = load_data("Weekly")
    except Exception:
        df = pd.read_csv("Weekly.csv")

    return df


# ============================================================
# 2. EDA
# ============================================================

def basic_eda(df):
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDirection counts:")
    print(df["Direction"].value_counts())

    print("\nSummary statistics:")
    print(df.describe())


# ============================================================
# 3. Preprocessing
# ============================================================

def preprocess_weekly(df):
    df = df.copy()

    df["Direction"] = df["Direction"].map({
        "Down": 0,
        "Up": 1
    })

    X = df[["Year", "Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume"]]
    y = df["Direction"]

    return X, y, df


# ============================================================
# 4. Train-test split
# ============================================================

def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )


# ============================================================
# 5. Models and parameter grids
# ============================================================

def get_models():
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
                "max_depth": [2, 3, 4, 5, None],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"]
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

    return models


# ============================================================
# 6. Calculate metrics
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
# 7. Train and evaluate all models
# ============================================================

def train_evaluate_models(X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    models = get_models()
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
        print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

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

    results_df = pd.DataFrame(results)
    return results_df, best_models


# ============================================================
# 8. Final comparison
# ============================================================

def final_comparison(results_df):
    results_df = results_df.sort_values(
        by="Test Accuracy",
        ascending=False
    )

    print("\n" + "=" * 100)
    print("FINAL MODEL COMPARISON")
    print("=" * 100)

    print(results_df)

    best_row = results_df.iloc[0]

    print("\n" + "=" * 100)
    print("BEST MODEL")
    print("=" * 100)

    print("Best Model:", best_row["Model"])
    print("Best Test Accuracy:", round(best_row["Test Accuracy"], 4))
    print("Best Parameters:", best_row["Best Parameters"])

    return best_row


# ============================================================
# 9. LOOCV
# ============================================================

def loocv_weekly(df):
    X_loocv = df[["Lag1", "Lag2"]]
    y_loocv = df["Direction"]

    loo = LeaveOneOut()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ])

    scores = cross_val_score(
        model,
        X_loocv,
        y_loocv,
        cv=loo,
        scoring="accuracy"
    )

    print("\n" + "=" * 100)
    print("LOOCV USING LOGISTIC REGRESSION WITH LAG1 AND LAG2")
    print("=" * 100)

    print("LOOCV Accuracy:", round(scores.mean(), 4))
    print("LOOCV Error   :", round(1 - scores.mean(), 4))


# ============================================================
# 10. ROC curve for best model
# ============================================================

def plot_best_roc(best_model_name, best_models, X_test, y_test):
    best_model = best_models[best_model_name]

    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"{best_model_name} AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Best Model")
        plt.legend()
        plt.grid(True)
        plt.show()


# ============================================================
# 11. Main function
# ============================================================

def main():
    df = load_weekly_data()

    basic_eda(df)

    X, y, df = preprocess_weekly(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    results_df, best_models = train_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    best_row = final_comparison(results_df)

    loocv_weekly(df)

    plot_best_roc(
        best_row["Model"],
        best_models,
        X_test,
        y_test
    )

    print("\nFinal Conclusion:")
    print("""
Weekly is a binary classification dataset.
Target variable is Direction.

Direction was encoded as:
Down = 0
Up   = 1

GridSearchCV was used to tune all models.
The best model was selected using highest test accuracy.

Today was not used because it causes data leakage.
""")


# ============================================================
# Run program
# ============================================================

main()