# ============================================================
# BrainCancer.csv Complete ML Practical
# Classification + Regression + PCA + Clustering
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram


# ============================================================
# 1. LOAD DATASET
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "BrainCancer.csv")

BrainCancer = pd.read_csv(file_path)

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)
print(BrainCancer.head())

print("\n" + "=" * 70)
print("SHAPE")
print("=" * 70)
print(BrainCancer.shape)

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)
print(BrainCancer.columns)

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)
print(BrainCancer.dtypes)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
print(BrainCancer.isnull().sum())


# ============================================================
# 2. BASIC CLEANING
# ============================================================

# Remove unwanted index column if present
if "Unnamed: 0" in BrainCancer.columns:
    BrainCancer = BrainCancer.drop(columns=["Unnamed: 0"])

# Drop missing values for simple exam practical
BrainCancer = BrainCancer.dropna()

print("\nShape after cleaning:")
print(BrainCancer.shape)

print("\nMissing values after cleaning:")
print(BrainCancer.isnull().sum())


# ============================================================
# 3. BASIC EDA
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(BrainCancer.describe())

# Show value counts for categorical columns
categorical_cols = BrainCancer.select_dtypes(include=["object", "category"]).columns.tolist()

print("\n" + "=" * 70)
print("CATEGORICAL VALUE COUNTS")
print("=" * 70)

for col in categorical_cols:
    print("\nColumn:", col)
    print(BrainCancer[col].value_counts())


# ============================================================
# 4. CLASSIFICATION TASK
# Target = status
# status usually means event/death status
# ============================================================

if "status" in BrainCancer.columns:

    print("\n" + "=" * 70)
    print("CLASSIFICATION TASK: Predict status")
    print("=" * 70)

    classification_data = BrainCancer.copy()

    # If status is text, convert to numeric
    if classification_data["status"].dtype == "object":
        print("\nOriginal status values:")
        print(classification_data["status"].unique())

        classification_data["status"] = classification_data["status"].map({
            "Alive": 0,
            "Dead": 1,
            "alive": 0,
            "dead": 1,
            "No": 0,
            "Yes": 1,
            "no": 0,
            "yes": 1
        })

    # If mapping failed, try categorical codes
    if classification_data["status"].isnull().sum() > 0:
        classification_data["status"] = pd.Categorical(BrainCancer["status"]).codes

    print("\nStatus value counts:")
    print(classification_data["status"].value_counts())

    X_class = classification_data.drop(columns=["status"])

    # Important:
    # Do not use time to predict status if you want a fair baseline classification.
    # But for exam, you can decide. Here we remove time to avoid leakage.
    if "time" in X_class.columns:
        X_class = X_class.drop(columns=["time"])

    y_class = classification_data["status"].astype(int)

    # Encode categorical variables
    X_class = pd.get_dummies(X_class, drop_first=True)

    print("\nClassification feature shape:")
    print(X_class.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X_class,
        y_class,
        test_size=0.30,
        random_state=42,
        stratify=y_class
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def evaluate_classifier(model_name, model, X_train_data, X_test_data):
        model.fit(X_train_data, y_train)

        y_pred = model.predict(X_test_data)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        sensitivity = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        cm = confusion_matrix(y_test, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        else:
            specificity = np.nan

        print("\n" + "=" * 70)
        print(model_name)
        print("=" * 70)

        print("Accuracy    :", round(acc, 4))
        print("Precision   :", round(precision, 4))
        print("Sensitivity :", round(sensitivity, 4))
        print("Specificity :", round(specificity, 4))
        print("F1-score    :", round(f1, 4))

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        roc_auc = np.nan

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_data)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            print("AUC:", round(roc_auc, 4))

            plt.figure(figsize=(7, 5))
            plt.plot(fpr, tpr, label=f"{model_name} AUC = {roc_auc:.4f}")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate / Sensitivity")
            plt.title("ROC Curve")
            plt.legend()
            plt.grid(True)
            plt.show()

        return {
            "Model": model_name,
            "Accuracy": acc,
            "Precision": precision,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "F1-score": f1,
            "AUC": roc_auc
        }

    classification_results = []

    classification_results.append(
        evaluate_classifier(
            "Logistic Regression",
            LogisticRegression(max_iter=5000),
            X_train_scaled,
            X_test_scaled
        )
    )

    classification_results.append(
        evaluate_classifier(
            "Naive Bayes",
            GaussianNB(),
            X_train_scaled,
            X_test_scaled
        )
    )

    classification_results.append(
        evaluate_classifier(
            "SVM RBF Classifier",
            SVC(kernel="rbf", C=1, gamma="scale", probability=True),
            X_train_scaled,
            X_test_scaled
        )
    )

    classification_results.append(
        evaluate_classifier(
            "Decision Tree Classifier",
            DecisionTreeClassifier(max_depth=4, random_state=42),
            X_train,
            X_test
        )
    )

    classification_results.append(
        evaluate_classifier(
            "Random Forest Classifier",
            RandomForestClassifier(n_estimators=200, random_state=42),
            X_train,
            X_test
        )
    )

    classification_results.append(
        evaluate_classifier(
            "Gradient Boosting Classifier",
            GradientBoostingClassifier(random_state=42),
            X_train,
            X_test
        )
    )

    try:
        from xgboost import XGBClassifier

        classification_results.append(
            evaluate_classifier(
                "XGBoost Classifier",
                XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    eval_metric="logloss",
                    random_state=42
                ),
                X_train,
                X_test
            )
        )

    except Exception as e:
        print("\nXGBoost not available.")
        print("Install using: pip install xgboost")
        print("Error:", e)

    classification_results_df = pd.DataFrame(classification_results)

    print("\n" + "=" * 70)
    print("CLASSIFICATION MODEL COMPARISON")
    print("=" * 70)

    print(classification_results_df.sort_values(by="Accuracy", ascending=False))

else:
    print("\nNo status column found, classification task skipped.")


# ============================================================
# 5. REGRESSION TASK
# Target = time
# Predict survival/follow-up time
# ============================================================

if "time" in BrainCancer.columns:

    print("\n" + "=" * 70)
    print("REGRESSION TASK: Predict time")
    print("=" * 70)

    regression_data = BrainCancer.copy()

    X_reg = regression_data.drop(columns=["time"])
    y_reg = regression_data["time"]

    # Remove status if you want to predict survival time using baseline variables only
    if "status" in X_reg.columns:
        X_reg = X_reg.drop(columns=["status"])

    X_reg = pd.get_dummies(X_reg, drop_first=True)

    print("\nRegression feature shape:")
    print(X_reg.shape)

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg,
        y_reg,
        test_size=0.30,
        random_state=42
    )

    scaler_reg = StandardScaler()

    Xr_train_scaled = scaler_reg.fit_transform(Xr_train)
    Xr_test_scaled = scaler_reg.transform(Xr_test)

    def evaluate_regressor(model_name, model, X_train_data, X_test_data):
        model.fit(X_train_data, yr_train)

        y_train_pred = model.predict(X_train_data)
        y_test_pred = model.predict(X_test_data)

        train_rmse = np.sqrt(mean_squared_error(yr_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(yr_test, y_test_pred))

        train_r2 = r2_score(yr_train, y_train_pred)
        test_r2 = r2_score(yr_test, y_test_pred)

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

    regression_results = []

    regression_results.append(
        evaluate_regressor(
            "Linear Regression",
            LinearRegression(),
            Xr_train_scaled,
            Xr_test_scaled
        )
    )

    regression_results.append(
        evaluate_regressor(
            "Ridge Regression",
            Ridge(alpha=1.0),
            Xr_train_scaled,
            Xr_test_scaled
        )
    )

    regression_results.append(
        evaluate_regressor(
            "Lasso Regression",
            Lasso(alpha=0.01, max_iter=10000),
            Xr_train_scaled,
            Xr_test_scaled
        )
    )

    regression_results.append(
        evaluate_regressor(
            "SVR RBF Regressor",
            SVR(kernel="rbf", C=10, gamma="scale"),
            Xr_train_scaled,
            Xr_test_scaled
        )
    )

    regression_results.append(
        evaluate_regressor(
            "Decision Tree Regressor",
            DecisionTreeRegressor(max_depth=4, random_state=42),
            Xr_train,
            Xr_test
        )
    )

    regression_results.append(
        evaluate_regressor(
            "Random Forest Regressor",
            RandomForestRegressor(n_estimators=200, random_state=42),
            Xr_train,
            Xr_test
        )
    )

    regression_results.append(
        evaluate_regressor(
            "Gradient Boosting Regressor",
            GradientBoostingRegressor(random_state=42),
            Xr_train,
            Xr_test
        )
    )

    try:
        from xgboost import XGBRegressor

        regression_results.append(
            evaluate_regressor(
                "XGBoost Regressor",
                XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                ),
                Xr_train,
                Xr_test
            )
        )

    except Exception as e:
        print("\nXGBoost Regressor not available.")
        print("Install using: pip install xgboost")
        print("Error:", e)

    regression_results_df = pd.DataFrame(regression_results)

    print("\n" + "=" * 70)
    print("REGRESSION MODEL COMPARISON")
    print("=" * 70)

    print(regression_results_df.sort_values(by="Test RMSE"))

else:
    print("\nNo time column found, regression task skipped.")


# ============================================================
# 6. PCA + CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("PCA + CLUSTERING")
print("=" * 70)

# Use all variables except time/status for unsupervised analysis
X_unsup = BrainCancer.copy()

for col in ["time", "status"]:
    if col in X_unsup.columns:
        X_unsup = X_unsup.drop(columns=[col])

X_unsup = pd.get_dummies(X_unsup, drop_first=True)

X_unsup_scaled = StandardScaler().fit_transform(X_unsup)

pca = PCA()
X_pca = pca.fit_transform(X_unsup_scaled)

pve = pca.explained_variance_ratio_
cumulative_pve = np.cumsum(pve)

print("\nExplained variance ratio:")
print(pve)

print("\nCumulative explained variance:")
print(cumulative_pve)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pve) + 1), cumulative_pve, marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("BrainCancer PCA Cumulative Explained Variance")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("BrainCancer PCA Plot")
plt.grid(True)
plt.show()


# ============================================================
# 7. K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_unsup_scaled)

BrainCancer["KMeans_Cluster"] = kmeans_labels

print("\nK-Means cluster counts:")
print(BrainCancer["KMeans_Cluster"].value_counts())

if "time" in BrainCancer.columns:
    print("\nMean survival time by cluster:")
    print(BrainCancer.groupby("KMeans_Cluster")["time"].mean())

if "status" in BrainCancer.columns:
    print("\nStatus distribution by cluster:")
    print(pd.crosstab(BrainCancer["KMeans_Cluster"], BrainCancer["status"]))

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering on BrainCancer Dataset")
plt.grid(True)
plt.show()


# ============================================================
# 8. HIERARCHICAL CLUSTERING
# ============================================================

sample_size = min(50, X_unsup_scaled.shape[0])
X_sample = X_unsup_scaled[:sample_size]

hc = linkage(
    X_sample,
    method="complete",
    metric="euclidean"
)

plt.figure(figsize=(12, 6))
dendrogram(hc)
plt.title("Hierarchical Clustering Dendrogram - BrainCancer")
plt.xlabel("Observation Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("""
Implemented on braincancer.csv:

1. Data loading
2. Missing value checking and cleaning
3. Exploratory data analysis
4. Classification using status as target
5. Logistic Regression
6. Naive Bayes
7. SVM RBF Classifier
8. Decision Tree Classifier
9. Random Forest Classifier
10. Gradient Boosting Classifier
11. XGBoost Classifier, if installed
12. Regression using time as target
13. Linear Regression
14. Ridge Regression
15. Lasso Regression
16. SVR Regressor
17. Decision Tree Regressor
18. Random Forest Regressor
19. Gradient Boosting Regressor
20. XGBoost Regressor, if installed
21. PCA
22. K-Means clustering
23. Hierarchical clustering

BrainCancer is mainly a survival dataset.
With normal ML methods, status can be treated as a classification target
and time can be treated as a regression target.
""")