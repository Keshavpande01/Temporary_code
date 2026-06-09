# ============================================================
# Portfolio Dataset from ISLP
# Bootstrap + Regression + PCA + Clustering
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ISLP import load_data

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram


# ============================================================
# 1. LOAD PORTFOLIO DATASET
# ============================================================

Portfolio = load_data("Portfolio")

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)
print(Portfolio.head())

print("\n" + "=" * 70)
print("SHAPE")
print("=" * 70)
print(Portfolio.shape)

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)
print(Portfolio.columns)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
print(Portfolio.isnull().sum())

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(Portfolio.describe())


# ============================================================
# 2. BASIC EDA
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)
print(Portfolio.corr())

plt.figure(figsize=(7, 5))
plt.scatter(Portfolio["X"], Portfolio["Y"])
plt.xlabel("Asset X Return")
plt.ylabel("Asset Y Return")
plt.title("Portfolio Dataset: X vs Y")
plt.grid(True)
plt.show()


# ============================================================
# 3. FUNCTION TO CALCULATE ALPHA
# ============================================================

def alpha_fn(data):
    """
    alpha = optimal proportion invested in X
    """

    X = data["X"]
    Y = data["Y"]

    var_x = np.var(X, ddof=1)
    var_y = np.var(Y, ddof=1)
    cov_xy = np.cov(X, Y)[0, 1]

    alpha = (var_y - cov_xy) / (var_x + var_y - 2 * cov_xy)

    return alpha


# ============================================================
# 4. ALPHA ESTIMATE USING FULL DATA
# ============================================================

alpha_hat = alpha_fn(Portfolio)

print("\n" + "=" * 70)
print("PORTFOLIO ALPHA ESTIMATE")
print("=" * 70)

print("Estimated alpha:", round(alpha_hat, 4))
print("Amount invested in X:", round(alpha_hat, 4))
print("Amount invested in Y:", round(1 - alpha_hat, 4))


# ============================================================
# 5. BOOTSTRAP FUNCTION
# ============================================================

def bootstrap_alpha(data, n_bootstrap=1000, random_state=42):
    np.random.seed(random_state)

    bootstrap_estimates = []

    n = len(data)

    for i in range(n_bootstrap):

        # sample rows with replacement
        sample_indices = np.random.choice(
            np.arange(n),
            size=n,
            replace=True
        )

        bootstrap_sample = data.iloc[sample_indices]

        alpha_boot = alpha_fn(bootstrap_sample)

        bootstrap_estimates.append(alpha_boot)

    return np.array(bootstrap_estimates)


# ============================================================
# 6. BOOTSTRAP STANDARD ERROR OF ALPHA
# ============================================================

bootstrap_estimates = bootstrap_alpha(
    Portfolio,
    n_bootstrap=1000,
    random_state=42
)

bootstrap_se = np.std(bootstrap_estimates, ddof=1)

print("\n" + "=" * 70)
print("BOOTSTRAP RESULT")
print("=" * 70)

print("Number of bootstrap samples:", len(bootstrap_estimates))
print("Mean bootstrap alpha:", round(np.mean(bootstrap_estimates), 4))
print("Bootstrap standard error:", round(bootstrap_se, 4))


# ============================================================
# 7. HISTOGRAM OF BOOTSTRAP ALPHA ESTIMATES
# ============================================================

plt.figure(figsize=(7, 5))
plt.hist(bootstrap_estimates, bins=30, edgecolor="black")
plt.axvline(alpha_hat, color="red", linestyle="--", label="Original alpha")
plt.xlabel("Bootstrap Alpha Estimates")
plt.ylabel("Frequency")
plt.title("Bootstrap Distribution of Alpha")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 8. 95% BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

lower_ci = np.percentile(bootstrap_estimates, 2.5)
upper_ci = np.percentile(bootstrap_estimates, 97.5)

print("\n" + "=" * 70)
print("95% BOOTSTRAP CONFIDENCE INTERVAL")
print("=" * 70)

print("Lower limit:", round(lower_ci, 4))
print("Upper limit:", round(upper_ci, 4))


# ============================================================
# 9. LINEAR REGRESSION
# Predict Y using X
# ============================================================

X_reg = Portfolio[["X"]]
y_reg = Portfolio["Y"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.30,
    random_state=42
)

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 70)
print("LINEAR REGRESSION: Predict Y using X")
print("=" * 70)

print("Intercept:", round(lr.intercept_, 4))
print("Slope:", round(lr.coef_[0], 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

plt.figure(figsize=(7, 5))
plt.scatter(X_test, y_test, label="Test Data")
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression on Portfolio Dataset")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 10. K-FOLD CROSS VALIDATION FOR LINEAR REGRESSION
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    LinearRegression(),
    X_reg,
    y_reg,
    cv=kf,
    scoring="neg_root_mean_squared_error"
)

rmse_scores = -cv_scores

print("\n" + "=" * 70)
print("5-FOLD CROSS VALIDATION")
print("=" * 70)

print("RMSE Scores:", rmse_scores)
print("Mean RMSE:", round(rmse_scores.mean(), 4))
print("Std RMSE:", round(rmse_scores.std(), 4))


# ============================================================
# 11. PCA
# ============================================================

X_scaled = StandardScaler().fit_transform(Portfolio)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

pve = pca.explained_variance_ratio_
cum_pve = np.cumsum(pve)

print("\n" + "=" * 70)
print("PCA RESULT")
print("=" * 70)

print("Explained variance ratio:", pve)
print("Cumulative explained variance:", cum_pve)

plt.figure(figsize=(7, 5))
plt.bar(range(1, len(pve) + 1), pve)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance - Portfolio")
plt.grid(True)
plt.show()


# ============================================================
# 12. K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

Portfolio["KMeans_Cluster"] = cluster_labels

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)

print(Portfolio["KMeans_Cluster"].value_counts())

print("\nMean values by cluster:")
print(Portfolio.groupby("KMeans_Cluster")[["X", "Y"]].mean())

plt.figure(figsize=(7, 5))
plt.scatter(Portfolio["X"], Portfolio["Y"], c=cluster_labels)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-Means Clustering on Portfolio Dataset")
plt.grid(True)
plt.show()


# ============================================================
# 13. HIERARCHICAL CLUSTERING
# ============================================================

hc = linkage(
    X_scaled,
    method="complete",
    metric="euclidean"
)

plt.figure(figsize=(12, 6))
dendrogram(hc)
plt.title("Hierarchical Clustering Dendrogram - Portfolio")
plt.xlabel("Observation Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()


# ============================================================
# 14. CLASSIFICATION VERSION
# Convert Y into high/low return
# ============================================================

Portfolio["Y_high"] = (
    Portfolio["Y"] > Portfolio["Y"].median()
).astype(int)

X_class = Portfolio[["X"]]
y_class = Portfolio["Y_high"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_class,
    y_class,
    test_size=0.30,
    random_state=42,
    stratify=y_class
)

scaler_class = StandardScaler()

Xc_train_scaled = scaler_class.fit_transform(Xc_train)
Xc_test_scaled = scaler_class.transform(Xc_test)

log_model = LogisticRegression()

log_model.fit(Xc_train_scaled, yc_train)

yc_pred = log_model.predict(Xc_test_scaled)

print("\n" + "=" * 70)
print("CLASSIFICATION: Predict High/Low Y Return")
print("=" * 70)

print("Accuracy:", round(accuracy_score(yc_test, yc_pred), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(yc_test, yc_pred))

print("\nClassification Report:")
print(classification_report(yc_test, yc_pred))


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("""
Implemented on Portfolio dataset:

1. Loaded Portfolio dataset from ISLP
2. Basic EDA
3. Scatter plot between X and Y
4. Correlation matrix
5. Portfolio allocation alpha calculation
6. Bootstrap estimation of alpha
7. Bootstrap standard error
8. Bootstrap confidence interval
9. Linear Regression
10. K-Fold Cross Validation
11. PCA
12. K-Means clustering
13. Hierarchical clustering
14. Classification after converting Y into high/low return

Main use of Portfolio dataset:
Bootstrap estimation of optimal portfolio allocation alpha.
""")