# ============================================================
# OJ DATASET - COMPLETE ML PRACTICAL
# Topics:
# Data preprocessing
# Logistic Regression
# Regularization
# Naive Bayes / Bayesian learning / Generative model
# SVM Linear and RBF Kernel
# Decision Tree
# Bagging
# Random Forest
# AdaBoost
# Gradient Boosting
# XGBoost
# PCA
# K-Means Clustering
# Hierarchical Clustering
# Evaluation Metrics
# ROC Curve and AUC
# K-Fold Cross Validation and Model Selection
# ============================================================

# Install if needed:
# pip install ISLP scikit-learn pandas numpy matplotlib scipy xgboost

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ISLP import load_data

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    adjusted_rand_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage


# ============================================================
# 1. LOAD OJ DATASET
# ============================================================

OJ = load_data("OJ")

print("First 5 rows:")
print(OJ.head())

print("\nDataset shape:")
print(OJ.shape)

print("\nColumns:")
print(OJ.columns)

print("\nTarget variable counts:")
print(OJ["Purchase"].value_counts())


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

# Target variable
# CH = 1, MM = 0
y = OJ["Purchase"].map({"CH": 1, "MM": 0})

# Feature variables
X = OJ.drop(columns=["Purchase"])

# Convert categorical columns into dummy variables
X = pd.get_dummies(X, drop_first=True)

print("\nFeature shape after encoding:")
print(X.shape)

print("\nFeatures:")
print(X.columns)


# ============================================================
# 3. TRAIN-TEST SPLIT
# Training data = 800 observations
# Testing data = remaining observations
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=800,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ============================================================
# 4. FEATURE SCALING
# Important for:
# Logistic Regression
# SVM
# PCA
# K-Means
# Hierarchical clustering
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_scaled_full = scaler.fit_transform(X)


# ============================================================
# 5. FUNCTION FOR EVALUATION METRICS
# ============================================================

def evaluate_model(model_name, model, X_train_data, X_test_data):
    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    # Train model
    model.fit(X_train_data, y_train)

    # Predict class
    y_pred = model.predict(X_test_data)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)   # same as recall
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    print("\nMetrics:")
    print("Accuracy    :", round(accuracy, 4))
    print("Precision   :", round(precision, 4))
    print("Sensitivity :", round(sensitivity, 4))
    print("Specificity :", round(specificity, 4))
    print("F1-score    :", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["MM", "CH"]))

    # ROC and AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_data)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_data)
    else:
        y_score = None

    if y_score is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        print("AUC:", round(roc_auc, 4))

        return {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr
        }

    else:
        return {
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "auc": np.nan,
            "fpr": None,
            "tpr": None
        }


# ============================================================
# 6. MODELS TO IMPLEMENT
# ============================================================

results = []


# ------------------------------------------------------------
# Logistic Regression
# Regularization: L2 by default
# ------------------------------------------------------------

log_reg = LogisticRegression(max_iter=5000)

results.append(
    evaluate_model(
        "Logistic Regression with L2 Regularization",
        log_reg,
        X_train_scaled,
        X_test_scaled
    )
)


# ------------------------------------------------------------
# Logistic Regression with L1 Regularization
# ------------------------------------------------------------

log_reg_l1 = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=5000
)

results.append(
    evaluate_model(
        "Logistic Regression with L1 Regularization",
        log_reg_l1,
        X_train_scaled,
        X_test_scaled
    )
)


# ------------------------------------------------------------
# Bayesian Learning / Generative Model
# Naive Bayes is a generative classifier
# ------------------------------------------------------------

nb = GaussianNB()

results.append(
    evaluate_model(
        "Gaussian Naive Bayes / Bayesian Generative Model",
        nb,
        X_train_scaled,
        X_test_scaled
    )
)


# ------------------------------------------------------------
# Linear SVM
# ------------------------------------------------------------

linear_svm = SVC(kernel="linear", probability=True, random_state=42)

results.append(
    evaluate_model(
        "Support Vector Machine - Linear Kernel",
        linear_svm,
        X_train_scaled,
        X_test_scaled
    )
)


# ------------------------------------------------------------
# RBF Kernel SVM
# ------------------------------------------------------------

rbf_svm = SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42)

results.append(
    evaluate_model(
        "Support Vector Machine - RBF Kernel",
        rbf_svm,
        X_train_scaled,
        X_test_scaled
    )
)


# ------------------------------------------------------------
# Polynomial Kernel SVM
# Kernel Method Example
# ------------------------------------------------------------

poly_svm = SVC(kernel="poly", degree=3, C=1, probability=True, random_state=42)

results.append(
    evaluate_model(
        "Support Vector Machine - Polynomial Kernel",
        poly_svm,
        X_train_scaled,
        X_test_scaled
    )
)


# ------------------------------------------------------------
# Decision Tree Classifier
# ------------------------------------------------------------

dt = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

results.append(
    evaluate_model(
        "Decision Tree Classifier",
        dt,
        X_train,
        X_test
    )
)


# ------------------------------------------------------------
# Bagging Classifier
# ------------------------------------------------------------

try:
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100,
        random_state=42
    )
except TypeError:
    bagging = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100,
        random_state=42
    )

results.append(
    evaluate_model(
        "Bagging Classifier",
        bagging,
        X_train,
        X_test
    )
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

results.append(
    evaluate_model(
        "Random Forest Classifier",
        rf,
        X_train,
        X_test
    )
)


# ------------------------------------------------------------
# AdaBoost
# ------------------------------------------------------------

try:
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )
except TypeError:
    ada = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )

results.append(
    evaluate_model(
        "AdaBoost Classifier",
        ada,
        X_train,
        X_test
    )
)


# ------------------------------------------------------------
# Gradient Boosting
# ------------------------------------------------------------

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

results.append(
    evaluate_model(
        "Gradient Boosting Classifier",
        gb,
        X_train,
        X_test
    )
)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

try:
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        eval_metric="logloss",
        random_state=42
    )

    results.append(
        evaluate_model(
            "XGBoost Classifier",
            xgb,
            X_train,
            X_test
        )
    )

except Exception as e:
    print("\nXGBoost not available.")
    print("Install using: pip install xgboost")
    print("Error:", e)


# ============================================================
# 7. MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON TABLE")
print("=" * 70)

print(
    results_df[
        ["model", "accuracy", "precision", "sensitivity", "specificity", "f1", "auc"]
    ].sort_values(by="accuracy", ascending=False)
)


# ============================================================
# 8. ROC CURVES
# ============================================================

plt.figure(figsize=(9, 7))

for result in results:
    if result["fpr"] is not None:
        plt.plot(
            result["fpr"],
            result["tpr"],
            label=f'{result["model"]} AUC={result["auc"]:.3f}'
        )

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Sensitivity")
plt.title("ROC Curve Comparison")
plt.legend(fontsize=8)
plt.grid(True)
plt.show()


# ============================================================
# 9. K-FOLD CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("K-FOLD CROSS VALIDATION")
print("=" * 70)

cv_models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "RBF SVM": SVC(kernel="rbf", C=1, gamma="scale"),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

for name, model in cv_models.items():
    if name in ["Logistic Regression", "RBF SVM"]:
        scores = cross_val_score(model, X_scaled_full, y, cv=5, scoring="accuracy")
    else:
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print(name)
    print("CV Scores:", scores)
    print("Mean CV Accuracy:", round(scores.mean(), 4))
    print()


# ============================================================
# 10. MODEL SELECTION USING GRIDSEARCHCV
# Example: Tune RBF SVM
# ============================================================

print("\n" + "=" * 70)
print("GRID SEARCH FOR RBF SVM")
print("=" * 70)

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1, "scale"]
}

grid_svm = GridSearchCV(
    SVC(kernel="rbf", probability=True),
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid_svm.fit(X_train_scaled, y_train)

print("Best Parameters:", grid_svm.best_params_)
print("Best CV Accuracy:", round(grid_svm.best_score_, 4))

best_svm = grid_svm.best_estimator_

results.append(
    evaluate_model(
        "Best RBF SVM after Grid Search",
        best_svm,
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 11. PCA - DIMENSIONALITY REDUCTION
# ============================================================

print("\n" + "=" * 70)
print("PCA - PRINCIPAL COMPONENT ANALYSIS")
print("=" * 70)

pca = PCA()
pca.fit(X_train_scaled)

explained_variance = pca.explained_variance_ratio_

print("Explained variance ratio:")
print(explained_variance)

print("\nCumulative explained variance:")
print(np.cumsum(explained_variance))

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(explained_variance) + 1),
    np.cumsum(explained_variance),
    marker="o"
)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# PCA with 2 components for visualization
# ------------------------------------------------------------

pca_2 = PCA(n_components=2)

X_train_pca = pca_2.fit_transform(X_train_scaled)
X_test_pca = pca_2.transform(X_test_scaled)

print("\nShape after PCA with 2 components:")
print(X_train_pca.shape)

plt.figure(figsize=(7, 5))
plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train,
    alpha=0.7
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("OJ Dataset After PCA")
plt.grid(True)
plt.show()


# ============================================================
# 12. CLASSIFICATION AFTER PCA
# Example: Logistic Regression after PCA
# ============================================================

pca_model = PCA(n_components=5)

X_train_pca_5 = pca_model.fit_transform(X_train_scaled)
X_test_pca_5 = pca_model.transform(X_test_scaled)

log_reg_pca = LogisticRegression(max_iter=5000)

results.append(
    evaluate_model(
        "Logistic Regression after PCA",
        log_reg_pca,
        X_train_pca_5,
        X_test_pca_5
    )
)


# ============================================================
# 13. K-MEANS CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)

# Use PCA 2D data for clustering visualization
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

kmeans_labels = kmeans.fit_predict(X_scaled_full)

print("K-Means Cluster Counts:")
print(pd.Series(kmeans_labels).value_counts())

# Since true labels are available, we can compare clustering with true Purchase labels
ari_kmeans = adjusted_rand_score(y, kmeans_labels)

print("Adjusted Rand Index for K-Means:")
print(round(ari_kmeans, 4))

# Plot K-Means on PCA data
X_pca_full = pca_2.fit_transform(X_scaled_full)

plt.figure(figsize=(7, 5))
plt.scatter(
    X_pca_full[:, 0],
    X_pca_full[:, 1],
    c=kmeans_labels,
    alpha=0.7
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering on OJ Dataset")
plt.grid(True)
plt.show()


# ============================================================
# 14. ELBOW METHOD FOR K-MEANS
# ============================================================

wcss = []

for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled_full)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters K")
plt.ylabel("WCSS / Inertia")
plt.title("Elbow Method for K-Means")
plt.grid(True)
plt.show()


# ============================================================
# 15. HIERARCHICAL CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("HIERARCHICAL CLUSTERING")
print("=" * 70)

# Agglomerative clustering
hierarchical = AgglomerativeClustering(n_clusters=2, linkage="ward")

hierarchical_labels = hierarchical.fit_predict(X_scaled_full)

print("Hierarchical Cluster Counts:")
print(pd.Series(hierarchical_labels).value_counts())

ari_hierarchical = adjusted_rand_score(y, hierarchical_labels)

print("Adjusted Rand Index for Hierarchical Clustering:")
print(round(ari_hierarchical, 4))


# ------------------------------------------------------------
# Dendrogram
# Use small sample because full dendrogram is too crowded
# ------------------------------------------------------------

sample_size = 50
X_sample = X_scaled_full[:sample_size]

linked = linkage(X_sample, method="ward")

plt.figure(figsize=(12, 6))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram - First 50 Samples")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()


# ============================================================
# 16. DECISION TREE VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE VISUALIZATION")
print("=" * 70)

dt_vis = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_vis.fit(X_train, y_train)

plt.figure(figsize=(18, 8))
plot_tree(
    dt_vis,
    feature_names=X.columns,
    class_names=["MM", "CH"],
    filled=True,
    rounded=True
)
plt.title("Decision Tree Classifier on OJ Dataset")
plt.show()


# ============================================================
# 17. FEATURE IMPORTANCE FROM RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

rf_importance = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_importance.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_importance.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importance.head(15))

plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance["Feature"].head(15),
    feature_importance["Importance"].head(15)
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importances - Random Forest")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PRACTICAL SUMMARY")
print("=" * 70)

print("""
Implemented on OJ dataset:

1. Data preprocessing
2. Train-test split
3. Feature scaling
4. Logistic Regression
5. L1 and L2 Regularization
6. Naive Bayes / Bayesian Learning / Generative Model
7. Linear SVM
8. RBF Kernel SVM
9. Polynomial Kernel SVM
10. Decision Tree Classifier
11. Bagging Classifier
12. Random Forest
13. AdaBoost
14. Gradient Boosting
15. XGBoost, if installed
16. Evaluation metrics
17. ROC Curve
18. AUC
19. K-Fold Cross Validation
20. GridSearchCV Model Selection
21. PCA
22. PCA + Classification
23. K-Means Clustering
24. Elbow Method
25. Hierarchical Clustering
26. Dendrogram
27. Feature Importance
""")