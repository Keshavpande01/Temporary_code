# ============================================================
# Weekly Dataset Complete ML Practical
# Classification + LOOCV + PCA + Clustering
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try ISLP first; if it fails, load local Weekly.csv
try:
    from ISLP import load_data
    Weekly = load_data("Weekly")
except Exception:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "Weekly.csv")
    Weekly = pd.read_csv(file_path)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
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

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram


# ============================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)
print(Weekly.head())

print("\nShape:")
print(Weekly.shape)

print("\nColumns:")
print(Weekly.columns)

print("\nMissing values:")
print(Weekly.isnull().sum())

print("\nDirection counts:")
print(Weekly["Direction"].value_counts())

print("\nSummary statistics:")
print(Weekly.describe())


# ============================================================
# 2. BASIC EDA
# ============================================================

plt.figure(figsize=(6, 4))
Weekly["Direction"].value_counts().plot(kind="bar")
plt.xlabel("Direction")
plt.ylabel("Count")
plt.title("Weekly Direction Distribution")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(Weekly["Lag1"], Weekly["Lag2"], c=(Weekly["Direction"] == "Up").astype(int))
plt.xlabel("Lag1")
plt.ylabel("Lag2")
plt.title("Lag1 vs Lag2 Colored by Direction")
plt.grid(True)
plt.show()


# ============================================================
# 3. TARGET ENCODING
# ============================================================
# Down = 0
# Up   = 1

Weekly["Direction_binary"] = Weekly["Direction"].map({
    "Down": 0,
    "Up": 1
})


# ============================================================
# 4. DEFINE FEATURES AND TARGET
# ============================================================
# Do not use Today as predictor because Today directly determines Direction.
# Using Today causes data leakage.

features = ["Year", "Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume"]

X = Weekly[features]
y = Weekly["Direction_binary"]

print("\nFeatures used:")
print(features)

print("\nTarget used: Direction_binary")


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. CLASSIFICATION EVALUATION FUNCTION
# ============================================================

def evaluate_classifier(model_name, model, X_train_data, X_test_data):
    model.fit(X_train_data, y_train)

    y_pred = model.predict(X_test_data)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    sensitivity = recall_score(y_test, y_pred, zero_division=0)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = np.nan

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("Confusion Matrix:")
    print(cm)

    print("\nAccuracy    :", round(accuracy, 4))
    print("Precision   :", round(precision, 4))
    print("Sensitivity :", round(sensitivity, 4))
    print("Specificity :", round(specificity, 4))
    print("F1-score    :", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Down", "Up"]))

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
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1-score": f1,
        "AUC": roc_auc
    }


classification_results = []


# ============================================================
# 7. LOGISTIC REGRESSION
# ============================================================

classification_results.append(
    evaluate_classifier(
        "Logistic Regression",
        LogisticRegression(max_iter=5000),
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 8. NAIVE BAYES
# ============================================================

classification_results.append(
    evaluate_classifier(
        "Gaussian Naive Bayes",
        GaussianNB(),
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 9. SVM LINEAR
# ============================================================

classification_results.append(
    evaluate_classifier(
        "Linear SVM",
        SVC(kernel="linear", C=1, probability=True),
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 10. SVM RBF
# ============================================================

classification_results.append(
    evaluate_classifier(
        "RBF Kernel SVM",
        SVC(kernel="rbf", C=1, gamma="scale", probability=True),
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 11. KNN
# ============================================================

classification_results.append(
    evaluate_classifier(
        "KNN Classifier",
        KNeighborsClassifier(n_neighbors=5),
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 12. DECISION TREE
# ============================================================

classification_results.append(
    evaluate_classifier(
        "Decision Tree Classifier",
        DecisionTreeClassifier(max_depth=4, random_state=42),
        X_train,
        X_test
    )
)


# ============================================================
# 13. RANDOM FOREST
# ============================================================

classification_results.append(
    evaluate_classifier(
        "Random Forest Classifier",
        RandomForestClassifier(n_estimators=200, random_state=42),
        X_train,
        X_test
    )
)


# ============================================================
# 14. GRADIENT BOOSTING
# ============================================================

classification_results.append(
    evaluate_classifier(
        "Gradient Boosting Classifier",
        GradientBoostingClassifier(random_state=42),
        X_train,
        X_test
    )
)


# ============================================================
# 15. XGBOOST
# ============================================================

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


# ============================================================
# 16. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(classification_results)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.sort_values(by="Accuracy", ascending=False))


# ============================================================
# 17. 10-FOLD CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("10-FOLD CROSS VALIDATION")
print("=" * 70)

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

cv_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),

    "Linear SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="linear", C=1))
    ]),

    "RBF SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", C=1, gamma="scale"))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

for name, model in cv_models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print("\nModel:", name)
    print("CV Scores:", scores)
    print("Mean Accuracy:", round(scores.mean(), 4))
    print("Std Accuracy :", round(scores.std(), 4))


# ============================================================
# 18. LOOCV USING LOGISTIC REGRESSION
# Original common Weekly question uses Lag1 and Lag2 only.
# ============================================================

print("\n" + "=" * 70)
print("LOOCV LOGISTIC REGRESSION USING LAG1 AND LAG2")
print("=" * 70)

X_loocv = Weekly[["Lag1", "Lag2"]]
y_loocv = Weekly["Direction_binary"]

n = len(Weekly)

correct_predictions = []

for i in range(n):
    X_test_i = X_loocv.iloc[[i]]
    y_test_i = y_loocv.iloc[i]

    X_train_i = X_loocv.drop(index=X_loocv.index[i])
    y_train_i = y_loocv.drop(index=y_loocv.index[i])

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("log_reg", LogisticRegression(max_iter=5000))
    ])

    model.fit(X_train_i, y_train_i)

    y_pred_i = model.predict(X_test_i)[0]

    if y_pred_i == y_test_i:
        correct_predictions.append(1)
    else:
        correct_predictions.append(0)

loocv_accuracy = np.mean(correct_predictions)
loocv_error = 1 - loocv_accuracy

print("LOOCV Accuracy:", round(loocv_accuracy, 4))
print("LOOCV Error   :", round(loocv_error, 4))


# ============================================================
# 19. PCA
# ============================================================

X_scaled_full = StandardScaler().fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled_full)

pve = pca.explained_variance_ratio_
cumulative_pve = np.cumsum(pve)

print("\n" + "=" * 70)
print("PCA RESULTS")
print("=" * 70)

print("Explained variance ratio:")
print(pve)

print("\nCumulative explained variance:")
print(cumulative_pve)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pve) + 1), cumulative_pve, marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Weekly PCA Cumulative Explained Variance")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Weekly PCA Plot Colored by Direction")
plt.grid(True)
plt.show()


# ============================================================
# 20. K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled_full)

Weekly["KMeans_Cluster"] = kmeans_labels

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)

print("K-Means cluster counts:")
print(Weekly["KMeans_Cluster"].value_counts())

print("\nCrosstab between Direction and KMeans Cluster:")
print(pd.crosstab(Weekly["Direction"], Weekly["KMeans_Cluster"]))

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering on Weekly Dataset")
plt.grid(True)
plt.show()


# ============================================================
# 21. HIERARCHICAL CLUSTERING
# ============================================================

sample_size = min(50, X_scaled_full.shape[0])
X_sample = X_scaled_full[:sample_size]

hc = linkage(
    X_sample,
    method="complete",
    metric="euclidean"
)

plt.figure(figsize=(12, 6))
dendrogram(hc)
plt.title("Hierarchical Clustering Dendrogram - First 50 Weekly Observations")
plt.xlabel("Observation Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("""
Implemented on Weekly dataset:

1. Data loading and EDA
2. Target encoding: Down = 0, Up = 1
3. Logistic Regression
4. Naive Bayes
5. Linear SVM
6. RBF SVM
7. KNN
8. Decision Tree Classifier
9. Random Forest Classifier
10. Gradient Boosting Classifier
11. XGBoost Classifier, if installed
12. Accuracy, Precision, Sensitivity, Specificity, F1-score, ROC and AUC
13. 10-fold Cross Validation
14. LOOCV using Logistic Regression with Lag1 and Lag2
15. PCA
16. K-Means clustering
17. Hierarchical clustering

Weekly is mainly a binary classification dataset because the target variable
Direction has two classes: Up and Down.

Important note:
Do not use Today as a predictor because Today directly determines Direction.
Using Today causes data leakage.
""")