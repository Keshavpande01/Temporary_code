# ============================================================
# USArrests - Hierarchical Clustering
# Complete Linkage with Euclidean and Correlation Distance
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# ------------------------------------------------------------
# 1. Load and prepare data
# ------------------------------------------------------------

df = pd.read_csv("USArrests.csv")

# First column contains state names
df = df.rename(columns={"Unnamed: 0": "State"})
df = df.set_index("State")

X = df.select_dtypes(include=["int64", "float64"])

# ------------------------------------------------------------
# 2. Scale data
# ------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# A + B: Complete Linkage + Euclidean Distance
# ============================================================

eu_dist = pdist(X_scaled, metric="euclidean")

hc_eu = linkage(eu_dist, method="complete")

plt.figure(figsize=(12, 6))
dendrogram(hc_eu, labels=X.index, leaf_rotation=90)
plt.title("Complete Linkage - Euclidean Distance")
plt.xlabel("States")
plt.ylabel("Euclidean Distance")
plt.show()

eu_clusters = fcluster(hc_eu, t=3, criterion="maxclust")

# ============================================================
# C + D: Complete Linkage + Correlation Distance
# ============================================================

corr_dist = pdist(X_scaled, metric="correlation")

hc_corr = linkage(corr_dist, method="complete")

plt.figure(figsize=(12, 6))
dendrogram(hc_corr, labels=X.index, leaf_rotation=90)
plt.title("Complete Linkage - Correlation Distance")
plt.xlabel("States")
plt.ylabel("Correlation Distance")
plt.show()

corr_clusters = fcluster(hc_corr, t=3, criterion="maxclust")

# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

result = X.copy()
result["Euclidean_Cluster"] = eu_clusters
result["Correlation_Cluster"] = corr_clusters

print(result[["Euclidean_Cluster", "Correlation_Cluster"]])
