# ============================================
# Boston Dataset - Clustering + Tree Models + Boosting
# ============================================

from ISLP import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# -------------------------------
# 1. Load Dataset
# -------------------------------
Boston = load_data("Boston")

# ============================================
# 🔹 PART 1: K-MEANS CLUSTERING
# ============================================

X_cluster = Boston.drop("medv", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

Boston['Cluster'] = clusters

print("\n===== CENTROIDS =====")
print(np.round(kmeans.cluster_centers_, 2))

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
for i in range(3):
    plt.scatter(X_pca[clusters == i][:, 0],
                X_pca[clusters == i][:, 1],
                label=f'Cluster {i}')

plt.title("K-means Clustering (Boston)")
plt.legend()
plt.show()

# ============================================
# 🔹 PART 2: TREE MODELS (REGRESSION)
# ============================================

y = Boston["medv"]
X = Boston.drop(["medv", "Cluster"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print("RMSE:", rmse)
    print("R2:", r2)

# Decision Tree
tree = DecisionTreeRegressor(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
evaluate("Decision Tree", y_test, tree.predict(X_test))

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
evaluate("Random Forest", y_test, rf.predict(X_test))

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)
evaluate("Gradient Boosting", y_test, gb.predict(X_test))

# ============================================
# 🔹 PART 3: ADABOOST (FIXED VERSION)
# ============================================

# Convert regression → classification (IMPORTANT FIX)
y_class = (Boston["medv"] > Boston["medv"].median()).astype(int)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.3, random_state=42
)

base_tree = DecisionTreeClassifier(max_depth=1)

ada = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)

ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)

print("\n===== ADABOOST (CLASSIFICATION) =====")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))