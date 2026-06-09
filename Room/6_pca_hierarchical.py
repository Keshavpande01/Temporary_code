# ============================================================
# 6_pca_hierarchical.py
# PCA + Hierarchical Clustering for Any CSV Dataset
#
# Change only:
# file_name
# target_col
#
# If dataset has no target column:
# target_col = None
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.metrics import adjusted_rand_score


# ============================================================
# 1. Load Dataset
# ============================================================

file_name = "your_dataset.csv"   # CHANGE THIS
target_col = None                # CHANGE THIS, example: "Private", "Direction", "medv"


df = pd.read_csv(file_name)


# ============================================================
# 2. Basic Cleaning
# ============================================================

for col in ["Unnamed: 0", "ID", "id", "index", "name"]:
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


# ============================================================
# 4. Remove Target Column if Present
# ============================================================
# In unsupervised learning, target should not be used for clustering.
# We keep target only for comparison after clustering.

true_labels = None

if target_col is not None and target_col in df.columns:
    true_labels = df[target_col]
    X = df.drop(columns=[target_col])
    print("\nRemoved target column for clustering:", target_col)
else:
    X = df.copy()
    print("\nNo target column removed.")


# ============================================================
# 5. Handle Missing Values
# ============================================================

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])


# ============================================================
# 6. Encode Categorical Columns
# ============================================================

X = pd.get_dummies(X, drop_first=True)

print("\n" + "=" * 70)
print("FEATURE MATRIX AFTER PREPROCESSING")
print("=" * 70)
print("Feature shape:", X.shape)
print("Feature columns:")
print(X.columns)


# ============================================================
# 7. Standardize Data
# ============================================================
# Scaling is compulsory for PCA and hierarchical clustering.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# 8. PCA
# ============================================================

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

pve = pca.explained_variance_ratio_
cum_pve = np.cumsum(pve)

print("\n" + "=" * 70)
print("PCA RESULTS")
print("=" * 70)

print("\nExplained variance ratio:")
print(pve)

print("\nCumulative explained variance:")
print(cum_pve)


# ============================================================
# 9. PCA Cumulative Variance Plot
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(cum_pve) + 1),
    cum_pve,
    marker="o"
)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.show()


# ============================================================
# 10. PCA Scatter Plot Before Clustering
# ============================================================

plt.figure(figsize=(7, 5))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    alpha=0.8
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Plot Before Hierarchical Clustering")
plt.grid(True)
plt.show()


# ============================================================
# 11. Hierarchical Clustering
# ============================================================
# Methods you can use:
# complete
# average
# single
# ward

method_used = "complete"

hc = linkage(
    X_scaled,
    method=method_used,
    metric="euclidean"
)


# ============================================================
# 12. Dendrogram
# ============================================================
# If dataset is very large, dendrogram can be crowded.
# So we also make a sample dendrogram below.

plt.figure(figsize=(14, 6))
dendrogram(hc)
plt.title("Hierarchical Clustering Dendrogram - Full Data")
plt.xlabel("Observations")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()


# ============================================================
# 13. Sample Dendrogram for Large Datasets
# ============================================================

sample_size = min(80, X_scaled.shape[0])
X_sample = X_scaled[:sample_size]

hc_sample = linkage(
    X_sample,
    method=method_used,
    metric="euclidean"
)

plt.figure(figsize=(14, 6))
dendrogram(hc_sample)
plt.title("Hierarchical Clustering Dendrogram - Sample Data")
plt.xlabel("Observations")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()


# ============================================================
# 14. Cut Tree into Clusters
# ============================================================

k = 2   # CHANGE THIS if needed

cluster_labels = cut_tree(
    hc,
    n_clusters=k
).reshape(-1)

print("\n" + "=" * 70)
print("HIERARCHICAL CLUSTERING RESULT")
print("=" * 70)

print("\nCluster labels:")
print(cluster_labels)

print("\nCluster counts:")
print(pd.Series(cluster_labels).value_counts().sort_index())


# ============================================================
# 15. PCA Plot with Hierarchical Clusters
# ============================================================

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    alpha=0.8
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"PCA + Hierarchical Clustering, K = {k}")
plt.grid(True)
plt.show()


# ============================================================
# 16. Cluster Summary
# ============================================================

result_df = X.copy()
result_df["Cluster"] = cluster_labels

print("\n" + "=" * 70)
print("CLUSTER SUMMARY")
print("=" * 70)

print(result_df.groupby("Cluster").mean())


# ============================================================
# 17. Compare Clusters with True Labels if Available
# ============================================================

if true_labels is not None:
    print("\n" + "=" * 70)
    print("CROSSTAB: TRUE LABELS VS CLUSTERS")
    print("=" * 70)

    print(pd.crosstab(true_labels, cluster_labels))

    try:
        ari = adjusted_rand_score(true_labels, cluster_labels)
        print("\nAdjusted Rand Index:", round(ari, 4))
    except Exception:
        print("\nAdjusted Rand Index could not be calculated.")


# ============================================================
# 18. Final Comment
# ============================================================

print("""
Final Comment:

PCA was used to reduce dimensionality and visualize the dataset in two dimensions.

Hierarchical clustering was applied after standardization.

The linkage method used was complete linkage with Euclidean distance.

The dendrogram shows how observations merge into clusters.

The tree was cut into K clusters and the clusters were visualized using PC1 and PC2.

If true labels are available, clusters are compared with original labels
using crosstab and Adjusted Rand Index.
""")
