# ============================================================
# Q3 - Manual K-Means Clustering
# K = 2
# n = 6 observations
# d = 2 features
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Create data
# ============================================================

data = pd.DataFrame({
    "Observation": [1, 2, 3, 4, 5, 6],
    "X1": [1, 1, 0, 5, 6, 4],
    "X2": [4, 3, 4, 1, 2, 0]
})

print("Original Data:")
print(data)


# ============================================================
# 2. Plot observations
# ============================================================

plt.figure(figsize=(6, 5))

plt.scatter(data["X1"], data["X2"], s=100)

for i in range(len(data)):
    plt.text(
        data.loc[i, "X1"] + 0.1,
        data.loc[i, "X2"] + 0.1,
        str(data.loc[i, "Observation"])
    )

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Original Observations")
plt.grid(True)
plt.show()


# ============================================================
# 3. Randomly assign initial cluster labels
# ============================================================

np.random.seed(1)

K = 2

data["Cluster"] = np.random.choice([1, 2], size=len(data))

print("\nInitial Random Cluster Assignment:")
print(data[["Observation", "X1", "X2", "Cluster"]])


# ============================================================
# 4. Function to compute centroids
# ============================================================

def compute_centroids(df):
    centroids = {}

    for cluster in sorted(df["Cluster"].unique()):
        cluster_points = df[df["Cluster"] == cluster][["X1", "X2"]]
        centroid = cluster_points.mean().values
        centroids[cluster] = centroid

    return centroids


# ============================================================
# 5. Function to assign observations to nearest centroid
# ============================================================

def assign_clusters(df, centroids):
    new_clusters = []

    for i in range(len(df)):
        point = df.loc[i, ["X1", "X2"]].values.astype(float)

        distances = {}

        for cluster, centroid in centroids.items():
            distance = np.sqrt(np.sum((point - centroid) ** 2))
            distances[cluster] = distance

        nearest_cluster = min(distances, key=distances.get)
        new_clusters.append(nearest_cluster)

    return new_clusters


# ============================================================
# 6. Manual K-Means Iteration
# ============================================================

max_iterations = 10

for iteration in range(1, max_iterations + 1):

    print("\n" + "=" * 70)
    print("Iteration", iteration)
    print("=" * 70)

    old_clusters = data["Cluster"].copy()

    # Step 1: compute centroid for each cluster
    centroids = compute_centroids(data)

    print("\nCentroids:")

    for cluster, centroid in centroids.items():
        print(f"Cluster {cluster}: X1 = {centroid[0]:.2f}, X2 = {centroid[1]:.2f}")

    # Step 2: assign each observation to nearest centroid
    data["Cluster"] = assign_clusters(data, centroids)

    print("\nNew Cluster Assignments:")
    print(data[["Observation", "X1", "X2", "Cluster"]])

    # Step 3: stop if cluster labels do not change
    if np.array_equal(old_clusters.values, data["Cluster"].values):
        print("\nClusters stopped changing. Algorithm converged.")
        break


# ============================================================
# 7. Final result
# ============================================================

print("\n" + "=" * 70)
print("Final K-Means Clustering Result")
print("=" * 70)

print(data[["Observation", "X1", "X2", "Cluster"]])

final_centroids = compute_centroids(data)

print("\nFinal Centroids:")

for cluster, centroid in final_centroids.items():
    print(f"Cluster {cluster}: X1 = {centroid[0]:.2f}, X2 = {centroid[1]:.2f}")


# ============================================================
# 8. Final plot with cluster colors
# ============================================================

plt.figure(figsize=(6, 5))

for cluster in sorted(data["Cluster"].unique()):
    cluster_data = data[data["Cluster"] == cluster]

    plt.scatter(
        cluster_data["X1"],
        cluster_data["X2"],
        s=120,
        label=f"Cluster {cluster}"
    )

    for i in cluster_data.index:
        plt.text(
            data.loc[i, "X1"] + 0.1,
            data.loc[i, "X2"] + 0.1,
            str(data.loc[i, "Observation"])
        )

# Plot centroids
for cluster, centroid in final_centroids.items():
    plt.scatter(
        centroid[0],
        centroid[1],
        s=250,
        marker="X",
        edgecolor="black",
        label=f"Centroid {cluster}"
    )

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Final K-Means Clustering Result")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 9. Final comment
# ============================================================

print("""
Final Comment:

K-Means clustering was performed manually with K = 2.
First, each observation was randomly assigned to one of two clusters.
Then the centroid of each cluster was computed.

Each observation was reassigned to the nearest centroid using Euclidean distance.
This process was repeated until the cluster assignments stopped changing.

The final plot shows the two clusters and their centroids.
""")