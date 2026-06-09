# ============================================================
# Ch12Ex13.csv - PCA + K-Means Clustering
# Gene Expression Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# ============================================================
# 1. Load dataset
# ============================================================

df = pd.read_csv("Ch12Ex13.csv")

print("First 5 rows:")
print(df.head())

print("\nOriginal shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)


# ============================================================
# 2. Remove index column if present
# ============================================================

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("\nShape after removing index column:")
print(df.shape)


# ============================================================
# 3. Fix orientation if needed
# ============================================================
# Expected shape should be:
# 40 samples x 1000 genes
#
# Sometimes file may be 1000 genes x 40 samples.
# If rows are not 40 but columns are 40, transpose the data.

if df.shape[0] != 40 and df.shape[1] == 40:
    df = df.T

print("\nFinal data shape used:")
print(df.shape)


# ============================================================
# 4. Create true labels
# ============================================================
# First 20 samples = Healthy
# Last 20 samples  = Diseased

true_labels = np.array(["Healthy"] * 20 + ["Diseased"] * 20)

true_binary = np.array([0] * 20 + [1] * 20)

print("\nTrue label counts:")
print(pd.Series(true_labels).value_counts())


# ============================================================
# 5. Standardize data
# ============================================================
# PCA and K-Means need scaling.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# ============================================================
# 6. PCA
# ============================================================

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("\nPCA shape:")
print(X_pca.shape)

print("\nExplained variance ratio first 5 PCs:")
print(pca.explained_variance_ratio_[:5])

print("\nCumulative variance first 5 PCs:")
print(np.cumsum(pca.explained_variance_ratio_[:5]))


# ============================================================
# 7. PCA plot: PC1 vs PC2 with true labels
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[true_binary == 0, 0],
    X_pca[true_binary == 0, 1],
    label="Healthy",
    s=80
)

plt.scatter(
    X_pca[true_binary == 1, 0],
    X_pca[true_binary == 1, 1],
    label="Diseased",
    s=80
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Plot: Healthy vs Diseased")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 8. Percent variance explained plot
# ============================================================

pve = pca.explained_variance_ratio_ * 100
cumulative_pve = np.cumsum(pve)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), pve[:10], marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Percent Variance Explained")
plt.title("Percent Variance Explained by First 10 PCs")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), cumulative_pve[:10], marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Percent Variance Explained")
plt.title("Cumulative Variance Explained by First 10 PCs")
plt.grid(True)
plt.show()


# ============================================================
# 9. K-Means Clustering with K = 2
# ============================================================

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

print("\nCluster labels:")
print(cluster_labels)

print("\nCluster counts:")
print(pd.Series(cluster_labels).value_counts().sort_index())


# ============================================================
# 10. Compare K-Means clusters with true labels
# ============================================================

comparison_table = pd.crosstab(
    true_labels,
    cluster_labels,
    rownames=["True Group"],
    colnames=["KMeans Cluster"]
)

print("\nComparison table:")
print(comparison_table)

ari = adjusted_rand_score(true_binary, cluster_labels)

print("\nAdjusted Rand Index:")
print(round(ari, 4))


# ============================================================
# 11. Plot K-Means clusters on PCA plot
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    s=80
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering on PCA Plot")
plt.grid(True)
plt.show()


# ============================================================
# 12. Final conclusion
# ============================================================

print("""
Final Comment:

PCA was applied to the Ch12Ex13 gene expression dataset.
The first 20 samples were considered Healthy and the next 20 samples Diseased.

The first two principal components were plotted to visualize whether the
two groups separate.

K-Means clustering was then applied with K = 2.
The obtained clusters were compared with the true Healthy/Diseased labels
using a crosstab and Adjusted Rand Index.

K-Means cluster labels are arbitrary, so Cluster 0 and Cluster 1 do not directly
mean Healthy or Diseased. Interpretation should be based on the crosstab.
""")