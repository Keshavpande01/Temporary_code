# ============================================================
# Q2 - NCI60 Dataset
# PCA + Variance Explained + Hierarchical Clustering
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ISLP import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram


# ============================================================
# 1. Load NCI60 data
# ============================================================

NCI60 = load_data("NCI60")

print("Type of NCI60 object:")
print(type(NCI60))

print("\nKeys in NCI60:")
print(NCI60.keys())


# ============================================================
# 2. Extract data and labels safely
# ============================================================

X = pd.DataFrame(NCI60["data"])

# labels are coming as 2D in your system, so convert safely
labels = pd.DataFrame(NCI60["labels"]).iloc[:, 0]

print("\nData matrix shape:")
print(X.shape)

print("\nLabels shape:")
print(labels.shape)

print("\nFirst 5 labels:")
print(labels.head())

print("\nLabel counts:")
print(labels.value_counts())


# ============================================================
# 3. Standardize the data
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# 4. Perform PCA
# ============================================================

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("\nPCA score matrix shape:")
print(X_pca.shape)


# ============================================================
# 5. PC1 vs PC2 plot
# ============================================================

plt.figure(figsize=(9, 6))

unique_labels = labels.unique()

for lab in unique_labels:
    idx = labels == lab
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        label=lab,
        alpha=0.8
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("NCI60 PCA Plot: PC1 vs PC2")
plt.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 6. PC1 vs PC3 plot
# ============================================================

plt.figure(figsize=(9, 6))

for lab in unique_labels:
    idx = labels == lab
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 2],
        label=lab,
        alpha=0.8
    )

plt.xlabel("PC1")
plt.ylabel("PC3")
plt.title("NCI60 PCA Plot: PC1 vs PC3")
plt.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 7. Percent variance explained
# ============================================================

pve = pca.explained_variance_ratio_ * 100
cumulative_pve = np.cumsum(pve)

print("\nPercent variance explained by first 10 PCs:")
for i in range(10):
    print(f"PC{i+1}: {pve[i]:.2f}%")

print("\nCumulative percent variance explained by first 10 PCs:")
for i in range(10):
    print(f"PC1 to PC{i+1}: {cumulative_pve[i]:.2f}%")


# ============================================================
# 8. Plot percent variance explained
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(pve) + 1),
    pve,
    marker="o"
)

plt.xlabel("Principal Component")
plt.ylabel("Percent Variance Explained")
plt.title("Percent Variance Explained by Principal Components")
plt.grid(True)
plt.show()


# ============================================================
# 9. Plot cumulative percent variance explained
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(cumulative_pve) + 1),
    cumulative_pve,
    marker="o"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Percent Variance Explained")
plt.title("Cumulative Percent Variance Explained")
plt.grid(True)
plt.show()


# ============================================================
# 10. Hierarchical clustering on first few PCs
# ============================================================

num_pcs = 5
X_pca_few = X_pca[:, :num_pcs]

hc_complete = linkage(
    X_pca_few,
    method="complete",
    metric="euclidean"
)

plt.figure(figsize=(14, 7))

dendrogram(
    hc_complete,
    labels=labels.astype(str).values,
    leaf_rotation=90
)

plt.title("Hierarchical Clustering on First 5 Principal Components")
plt.xlabel("Cancer Cell Lines")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()


# ============================================================
# 11. Final comment
# ============================================================

print("""
Final Comment:

PCA was performed on the NCI60 data after standardization.

The PC1 vs PC2 and PC1 vs PC3 plots show whether cancer cell lines from
the same cancer type are close together.

The percent variance explained plot shows how much variation each principal
component captures.

The cumulative percent variance explained plot shows how many principal
components are required to explain most of the variation.

Hierarchical clustering was performed on the first 5 principal component
score vectors using complete linkage.
""")