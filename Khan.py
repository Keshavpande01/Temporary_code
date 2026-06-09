# ============================================================
# Khan Dataset from ISLP
# Classification + PCA + Clustering
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ISLP import load_data

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    adjusted_rand_score
)

from scipy.cluster.hierarchy import linkage, dendrogram


# ============================================================
# 1. LOAD KHAN DATASET
# ============================================================

Khan = load_data("Khan")

print("Type of Khan object:")
print(type(Khan))

print("\nKeys in Khan dataset:")
print(Khan.keys())


# ============================================================
# 2. EXTRACT TRAIN AND TEST DATA
# ============================================================

X_train = pd.DataFrame(Khan["xtrain"])
X_test = pd.DataFrame(Khan["xtest"])

y_train = pd.Series(np.ravel(Khan["ytrain"]))
y_test = pd.Series(np.ravel(Khan["ytest"]))

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining labels shape:")
print(y_train.shape)

print("\nTesting labels shape:")
print(y_test.shape)

print("\nTraining class counts:")
print(y_train.value_counts().sort_index())

print("\nTesting class counts:")
print(y_test.value_counts().sort_index())


# ============================================================
# 3. FEATURE SCALING
# ============================================================
# Gene expression datasets should be scaled before PCA, SVM, KNN.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 4. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, model, X_train_data, X_test_data):
    model.fit(X_train_data, y_train)

    y_train_pred = model.predict(X_train_data)
    y_test_pred = model.predict(X_test_data)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("Training Accuracy:", round(train_acc, 4))
    print("Test Accuracy    :", round(test_acc, 4))

    print("\nConfusion Matrix on Test Data:")
    print(confusion_matrix(y_test, y_test_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    return {
        "Model": model_name,
        "Training Accuracy": train_acc,
        "Test Accuracy": test_acc
    }


results = []


# ============================================================
# 5. CLASSIFICATION MODELS
# ============================================================

# Logistic Regression
results.append(
    evaluate_model(
        "Logistic Regression",
        LogisticRegression(max_iter=5000),
        X_train_scaled,
        X_test_scaled
    )
)


# Linear SVM
results.append(
    evaluate_model(
        "Linear SVM",
        SVC(kernel="linear", C=1),
        X_train_scaled,
        X_test_scaled
    )
)


# RBF SVM
results.append(
    evaluate_model(
        "RBF Kernel SVM",
        SVC(kernel="rbf", C=1, gamma="scale"),
        X_train_scaled,
        X_test_scaled
    )
)


# KNN
results.append(
    evaluate_model(
        "K-Nearest Neighbors",
        KNeighborsClassifier(n_neighbors=3),
        X_train_scaled,
        X_test_scaled
    )
)


# Naive Bayes
results.append(
    evaluate_model(
        "Gaussian Naive Bayes",
        GaussianNB(),
        X_train_scaled,
        X_test_scaled
    )
)


# Decision Tree
results.append(
    evaluate_model(
        "Decision Tree Classifier",
        DecisionTreeClassifier(random_state=42),
        X_train,
        X_test
    )
)


# Random Forest
results.append(
    evaluate_model(
        "Random Forest Classifier",
        RandomForestClassifier(n_estimators=200, random_state=42),
        X_train,
        X_test
    )
)


# ============================================================
# 6. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.sort_values(by="Test Accuracy", ascending=False))


# ============================================================
# 7. PCA ON KHAN DATASET
# ============================================================

# Combine train and test for visualization
X_all = pd.concat([X_train, X_test], axis=0)
y_all = pd.concat([y_train, y_test], axis=0)

X_all_scaled = StandardScaler().fit_transform(X_all)

pca = PCA()
X_pca = pca.fit_transform(X_all_scaled)

pve = pca.explained_variance_ratio_
cum_pve = np.cumsum(pve)

print("\n" + "=" * 70)
print("PCA RESULTS")
print("=" * 70)

print("Variance explained by first 10 PCs:")
for i in range(10):
    print(f"PC{i+1}: {pve[i]:.4f}")

print("\nCumulative variance explained by first 10 PCs:")
for i in range(10):
    print(f"PC1 to PC{i+1}: {cum_pve[i]:.4f}")


# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), cum_pve[:20], marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Khan Dataset PCA - Cumulative Explained Variance")
plt.grid(True)
plt.show()


# PCA scatter plot PC1 vs PC2
plt.figure(figsize=(8, 6))

for label in sorted(y_all.unique()):
    idx = y_all == label
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        label=f"Class {label}",
        alpha=0.8
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Khan Dataset PCA Plot: PC1 vs PC2")
plt.legend()
plt.grid(True)
plt.show()


# PCA scatter plot PC1 vs PC3
plt.figure(figsize=(8, 6))

for label in sorted(y_all.unique()):
    idx = y_all == label
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 2],
        label=f"Class {label}",
        alpha=0.8
    )

plt.xlabel("PC1")
plt.ylabel("PC3")
plt.title("Khan Dataset PCA Plot: PC1 vs PC3")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 8. K-MEANS CLUSTERING
# ============================================================
# Number of true classes usually = 4

K = len(np.unique(y_all))

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_all_scaled)

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)

print("K-Means cluster counts:")
print(pd.Series(kmeans_labels).value_counts().sort_index())

print("\nCrosstab between true class and K-Means cluster:")
print(pd.crosstab(y_all, kmeans_labels, rownames=["True Class"], colnames=["KMeans Cluster"]))

ari_kmeans = adjusted_rand_score(y_all, kmeans_labels)

print("\nAdjusted Rand Index for K-Means:")
print(round(ari_kmeans, 4))


plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    alpha=0.8
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering on Khan Dataset")
plt.grid(True)
plt.show()


# ============================================================
# 9. HIERARCHICAL CLUSTERING
# ============================================================

hc = linkage(
    X_all_scaled,
    method="complete",
    metric="euclidean"
)

plt.figure(figsize=(14, 6))

dendrogram(
    hc,
    labels=y_all.astype(str).values,
    leaf_rotation=90
)

plt.title("Hierarchical Clustering Dendrogram - Khan Dataset")
plt.xlabel("True Class Labels")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("""
Implemented on Khan dataset:

1. Loaded Khan dataset from ISLP
2. Extracted xtrain, xtest, ytrain, ytest
3. Standardized gene expression data
4. Logistic Regression
5. Linear SVM
6. RBF SVM
7. KNN
8. Gaussian Naive Bayes
9. Decision Tree Classifier
10. Random Forest Classifier
11. PCA
12. K-Means clustering
13. Hierarchical clustering
14. Confusion matrix and classification report
15. Model comparison using test accuracy

Khan is a high-dimensional multiclass classification dataset.
It is especially useful for PCA, SVM, KNN, Random Forest and clustering.
""")