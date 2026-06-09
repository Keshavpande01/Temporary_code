import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ISLP import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------ LOAD DATA ------------------
college = load_data("College")

# Convert to DataFrame
df = pd.DataFrame(college)

# ------------------ BASIC EDA ------------------
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)

# Convert Private (Yes/No → 1/0)
df['Private'] = df['Private'].map({'Yes': 1, 'No': 0})

print("\nMissing Values:\n", df.isnull().sum())

print("\nSummary Statistics:\n", df.describe())

# ------------------ FEATURE SELECTION ------------------
X = df.drop(columns=['Private'])   # keep only numeric features

# ------------------ STANDARDIZATION ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ PCA ------------------
pca = PCA()
X_pca_full = pca.fit_transform(X_scaled)

# Scree Plot
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Scree Plot (College Dataset)")
plt.grid()
plt.show()

# Reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance (2 PCs):", pca.explained_variance_ratio_)

# ------------------ PCA VISUALIZATION ------------------
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df['Private'],
    cmap='coolwarm',
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization (Private vs Public Colleges)")
plt.colorbar(label="Private (1=Yes, 0=No)")
plt.grid()
plt.show()

# ------------------ ELBOW METHOD ------------------
wcss = []
K_range = range(1, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, wcss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method (College Dataset)")
plt.grid()
plt.show()

# ------------------ KMEANS ------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# ------------------ CLUSTER VISUALIZATION ------------------
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap='viridis',
    alpha=0.7
)

# Centroids
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='red',
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering (College Dataset)")
plt.legend()
plt.grid()
plt.show()

# ------------------ SILHOUETTE SCORE ------------------
score = silhouette_score(X_pca, clusters)
print("\nSilhouette Score:", score)

# ------------------ INTERPRETATION ------------------
df['Cluster'] = clusters
print("\nCluster vs Private:")
print(pd.crosstab(df['Cluster'], df['Private']))