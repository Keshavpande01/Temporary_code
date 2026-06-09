# ============================================================
# USArrests Dataset - Hierarchical Clustering
# Question:
# (a) Complete linkage + Euclidean distance
# (b) Cut dendrogram into 3 clusters
# (c) Complete linkage + Correlation distance
# (d) Cut dendrogram into 3 clusters
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


# ============================================================
# 1. LOAD LOCAL CSV FILE
# ============================================================

# This will load USarrest.csv from the same directory as this Python file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "USArrests.csv")

USArrests = pd.read_csv(file_path)

print("First 5 rows:")
print(USArrests.head())

print("\nShape:")
print(USArrests.shape)

print("\nColumns:")
print(USArrests.columns)


# ============================================================
# 2. FIX STATE COLUMN
# ============================================================
# Usually first column contains state names.
# We set it as index so states appear in dendrogram labels.

first_col = USArrests.columns[0]

# If first column is state names, make it index
if USArrests[first_col].dtype == "object":
    USArrests = USArrests.set_index(first_col)

print("\nAfter setting state names as index:")
print(USArrests.head())


# ============================================================
# 3. KEEP ONLY NUMERIC COLUMNS
# ============================================================

X = USArrests.select_dtypes(include=["int64", "float64"])

print("\nNumeric data used for clustering:")
print(X.head())

print("\nNumeric columns:")
print(X.columns)


# ============================================================
# 4. STANDARDIZE THE DATA
# ============================================================
# Important because Murder, Assault, UrbanPop, Rape
# are measured on different scales.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# PART A:
# Hierarchical clustering with complete linkage
# and Euclidean distance
# ============================================================

euclidean_distance = pdist(X_scaled, metric="euclidean")

hc_euclidean = linkage(
    euclidean_distance,
    method="complete"
)

plt.figure(figsize=(14, 6))
dendrogram(
    hc_euclidean,
    labels=X.index,
    leaf_rotation=90
)
plt.title("Complete Linkage Hierarchical Clustering - Euclidean Distance")
plt.xlabel("States")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.show()


# ============================================================
# PART B:
# Cut dendrogram into 3 clusters
# ============================================================

euclidean_clusters = fcluster(
    hc_euclidean,
    t=3,
    criterion="maxclust"
)

result = X.copy()
result["Euclidean_Cluster"] = euclidean_clusters

print("\n" + "=" * 70)
print("Clusters using Complete Linkage + Euclidean Distance")
print("=" * 70)

for cluster_no in sorted(result["Euclidean_Cluster"].unique()):
    states = result[result["Euclidean_Cluster"] == cluster_no].index.tolist()

    print(f"\nCluster {cluster_no}:")
    for state in states:
        print(state)


# ============================================================
# PART C:
# Hierarchical clustering with complete linkage
# and correlation-based distance
# ============================================================
# Correlation distance = 1 - correlation

correlation_distance = pdist(X_scaled, metric="correlation")

hc_correlation = linkage(
    correlation_distance,
    method="complete"
)

plt.figure(figsize=(14, 6))
dendrogram(
    hc_correlation,
    labels=X.index,
    leaf_rotation=90
)
plt.title("Complete Linkage Hierarchical Clustering - Correlation Distance")
plt.xlabel("States")
plt.ylabel("Correlation Distance")
plt.tight_layout()
plt.show()


# ============================================================
# PART D:
# Cut dendrogram into 3 clusters
# ============================================================

correlation_clusters = fcluster(
    hc_correlation,
    t=3,
    criterion="maxclust"
)

result["Correlation_Cluster"] = correlation_clusters

print("\n" + "=" * 70)
print("Clusters using Complete Linkage + Correlation Distance")
print("=" * 70)

for cluster_no in sorted(result["Correlation_Cluster"].unique()):
    states = result[result["Correlation_Cluster"] == cluster_no].index.tolist()

    print(f"\nCluster {cluster_no}:")
    for state in states:
        print(state)


# ============================================================
# FINAL CLUSTER ASSIGNMENT TABLE
# ============================================================

print("\n" + "=" * 70)
print("Final Cluster Assignment Table")
print("=" * 70)

print(result[["Euclidean_Cluster", "Correlation_Cluster"]])