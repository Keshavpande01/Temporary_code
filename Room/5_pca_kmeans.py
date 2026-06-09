# ============================================================
# 5_pca_kmeans.py
# PCA + K-Means Clustering for Any CSV Dataset
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


# ============================================================
# 1. Load Dataset
# ============================================================

file_name = "your_dataset.csv"   # CHANGE THIS
target_col = None                # CHANGE THIS, example: "Purchase", "Direction", "medv"


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
# We keep it only for comparison after clustering.

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
# Scaling is compulsory for PCA and K-Means.

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
plt.title("PCA Plot Before K-Means")
plt.grid(True)
plt.show()


# ============================================================
# 11. Elbow Method
# ============================================================

wcss = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, 11),
    wcss,
    marker="o"
)
plt.xlabel("Number of Clusters K")
plt.ylabel("WCSS / Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

print("\n" + "=" * 70)
print("ELBOW METHOD WCSS VALUES")
print("=" * 70)

for k, value in zip(range(1, 11), wcss):
    print("K =", k, "WCSS =", round(value, 4))


# ============================================================
# 12. Silhouette Score
# ============================================================
# K = 1 is not valid for silhouette score.

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, cluster_labels)

    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)
plt.xlabel("Number of Clusters K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Best K")
plt.grid(True)
plt.show()

print("\n" + "=" * 70)
print("SILHOUETTE SCORES")
print("=" * 70)

for k, score in zip(range(2, 11), silhouette_scores):
    print("K =", k, "Silhouette Score =", round(score, 4))


# ============================================================
# 13. Select Best K
# ============================================================

best_k = np.argmax(silhouette_scores) + 2

print("\n" + "=" * 70)
print("BEST K")
print("=" * 70)
print("Best K based on highest silhouette score:", best_k)


# ============================================================
# 14. Final K-Means
# ============================================================

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

final_clusters = final_kmeans.fit_predict(X_scaled)

print("\n" + "=" * 70)
print("FINAL K-MEANS RESULT")
print("=" * 70)

print("\nCluster counts:")
print(pd.Series(final_clusters).value_counts().sort_index())


# ============================================================
# 15. PCA Plot with K-Means Clusters
# ============================================================

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=final_clusters,
    alpha=0.8
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"PCA + K-Means Clustering, K = {best_k}")
plt.grid(True)
plt.show()


# ============================================================
# 16. Cluster Summary
# ============================================================

result_df = X.copy()
result_df["Cluster"] = final_clusters

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

    print(pd.crosstab(true_labels, final_clusters))

    # ARI works if true_labels can be compared as categories
    try:
        ari = adjusted_rand_score(true_labels, final_clusters)
        print("\nAdjusted Rand Index:", round(ari, 4))
    except Exception:
        print("\nAdjusted Rand Index could not be calculated.")


# ============================================================
# 18. Final Comment
# ============================================================

print("""
Final Comment:

PCA was used to reduce dimensionality and visualize the dataset in two dimensions.

K-Means clustering was applied after standardization.

The Elbow Method was used to observe WCSS/inertia for different values of K.

The Silhouette Score was used to automatically select the best number of clusters.

The final clusters were visualized using PC1 and PC2.

If true labels are available, clusters are compared with the original labels
using crosstab and Adjusted Rand Index.
""")
