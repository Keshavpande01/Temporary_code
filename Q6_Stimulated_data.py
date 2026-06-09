# ============================================================
# Simulated Data - PCA and K-Means Clustering
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from scipy.stats import mode


# ============================================================
# 1. Generate simulated data
# ============================================================
# 3 classes
# 40 observations per class
# Total = 120 observations

np.random.seed(1)

n_per_class = 40
n_features = 10

# Class 1 centered around 0
class_1 = np.random.normal(
    loc=0,
    scale=1,
    size=(n_per_class, n_features)
)

# Class 2 shifted positively
class_2 = np.random.normal(
    loc=3,
    scale=1,
    size=(n_per_class, n_features)
)

# Class 3 shifted negatively
class_3 = np.random.normal(
    loc=-3,
    scale=1,
    size=(n_per_class, n_features)
)

# Combine data
X = np.vstack([class_1, class_2, class_3])

# True class labels
true_labels = np.array(
    [0] * n_per_class +
    [1] * n_per_class +
    [2] * n_per_class
)

print("Shape of simulated data:")
print(X.shape)

print("\nTrue label counts:")
print(pd.Series(true_labels).value_counts())


# ============================================================
# 2. Standardize the data
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# 3. Perform PCA
# ============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nShape after PCA:")
print(X_pca.shape)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal variance explained by first two PCs:")
print(np.sum(pca.explained_variance_ratio_))


# ============================================================
# 4. Plot first two principal component score vectors
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    X_pca[true_labels == 0, 0],
    X_pca[true_labels == 0, 1],
    label="Class 1",
    alpha=0.8
)

plt.scatter(
    X_pca[true_labels == 1, 0],
    X_pca[true_labels == 1, 1],
    label="Class 2",
    alpha=0.8
)

plt.scatter(
    X_pca[true_labels == 2, 0],
    X_pca[true_labels == 2, 1],
    label="Class 3",
    alpha=0.8
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Plot of Simulated Data")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# Comment:
# If the three classes are separated in the PCA plot,
# then continue to K-Means.
# If not, increase the mean shift in class_2 and class_3.
# Example:
# class_2 loc = 5
# class_3 loc = -5
# ============================================================


# ============================================================
# 5. Perform K-Means clustering with K = 3
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=1,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

print("\nK-Means cluster counts:")
print(pd.Series(cluster_labels).value_counts())


# ============================================================
# 6. Compare K-Means clusters with true class labels
# ============================================================

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, cluster_labels))

ari = adjusted_rand_score(true_labels, cluster_labels)

print("\nAdjusted Rand Index:")
print(ari)


# ============================================================
# 7. Plot K-Means clusters on PCA plot
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    alpha=0.8
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters on PCA Plot")
plt.grid(True)
plt.show()


# ============================================================
# 8. Better comparison using crosstab
# ============================================================
# K-Means cluster numbers are arbitrary.
# Cluster 0 does not necessarily mean Class 0.
# Therefore, use crosstab to compare.

comparison_table = pd.crosstab(
    true_labels,
    cluster_labels,
    rownames=["True Class"],
    colnames=["K-Means Cluster"]
)

print("\nComparison Table:")
print(comparison_table)


# ============================================================
# 9. Final comment
# ============================================================

print("""
Final Comment:

A simulated dataset with 120 observations and 10 features was generated.
There were 3 true classes with 40 observations in each class.

PCA was performed and the first two principal components were plotted.
The PCA plot shows whether the three classes are visually separated.

K-Means clustering was then applied with K = 3.
The obtained cluster labels were compared with the true class labels using
a confusion matrix and Adjusted Rand Index.

Important:
K-Means cluster labels are arbitrary. Cluster label 0, 1, or 2 may not directly
match the true class labels 0, 1, or 2. Therefore, the comparison should be done
using a confusion matrix or crosstab instead of directly checking equality.
""")