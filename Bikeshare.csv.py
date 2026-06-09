# ============================================================
# Bikeshare Dataset from ISLP
# Regression + PCA + Clustering + Classification
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ISLP import load_data

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor
)
from sklearn.svm import SVR, SVC

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
# 1. LOAD BIKESHARE DATASET
# ============================================================

Bikeshare = load_data("Bikeshare")

print("First 5 rows:")
print(Bikeshare.head())

print("\nShape:")
print(Bikeshare.shape)

print("\nColumns:")
print(Bikeshare.columns)

print("\nMissing values:")
print(Bikeshare.isnull().sum())

print("\nData types:")
print(Bikeshare.dtypes)

print("\nSummary statistics:")
print(Bikeshare.describe())


# ============================================================
# 2. TARGET AND FEATURES
# ============================================================

target_column = "bikers"

X = Bikeshare.drop(columns=[target_column])
y = Bikeshare[target_column]

print("\nTarget variable:", target_column)
print("Feature shape:", X.shape)


# ============================================================
# 3. ENCODE CATEGORICAL VARIABLES
# ============================================================

X = pd.get_dummies(X, drop_first=True)

print("\nFeature shape after encoding:")
print(X.shape)

print("\nFeature columns:")
print(X.columns)


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. REGRESSION EVALUATION FUNCTION
# ============================================================

def evaluate_regression(model_name, model, X_train_data, X_test_data):
    model.fit(X_train_data, y_train)

    y_train_pred = model.predict(X_train_data)
    y_test_pred = model.predict(X_test_data)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("Train RMSE:", round(train_rmse, 4))
    print("Test RMSE :", round(test_rmse, 4))

    print("Train R2  :", round(train_r2, 4))
    print("Test R2   :", round(test_r2, 4))

    return {
        "Model": model_name,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train R2": train_r2,
        "Test R2": test_r2
    }


results = []


# ============================================================
# 6. SIMPLE LINEAR REGRESSION
# bikers ~ temp
# ============================================================

if "temp" in Bikeshare.columns:

    X_temp = Bikeshare[["temp"]]
    y_bikers = Bikeshare["bikers"]

    X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
        X_temp,
        y_bikers,
        test_size=0.30,
        random_state=42
    )

    simple_lr = LinearRegression()
    simple_lr.fit(X_temp_train, y_temp_train)

    y_temp_pred = simple_lr.predict(X_temp_test)

    print("\n" + "=" * 70)
    print("Simple Linear Regression: bikers ~ temp")
    print("=" * 70)

    print("Intercept:", simple_lr.intercept_)
    print("Slope:", simple_lr.coef_[0])
    print("RMSE:", round(np.sqrt(mean_squared_error(y_temp_test, y_temp_pred)), 4))
    print("R2:", round(r2_score(y_temp_test, y_temp_pred), 4))

    plt.figure(figsize=(7, 5))
    plt.scatter(X_temp_test, y_temp_test, label="Test Data")
    plt.plot(X_temp_test, y_temp_pred, color="red", label="Regression Line")
    plt.xlabel("Temperature")
    plt.ylabel("Number of Bikers")
    plt.title("Simple Linear Regression: Bikers vs Temperature")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# 7. MULTIPLE LINEAR REGRESSION
# ============================================================

results.append(
    evaluate_regression(
        "Multiple Linear Regression",
        LinearRegression(),
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 8. RIDGE REGRESSION
# ============================================================

results.append(
    evaluate_regression(
        "Ridge Regression",
        Ridge(alpha=1.0),
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 9. LASSO REGRESSION
# ============================================================

results.append(
    evaluate_regression(
        "Lasso Regression",
        Lasso(alpha=0.01, max_iter=10000),
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 10. POLYNOMIAL REGRESSION
# bikers ~ temp + temp^2
# ============================================================

if "temp" in Bikeshare.columns:

    poly_model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("linear", LinearRegression())
    ])

    poly_model.fit(X_temp_train, y_temp_train)

    y_poly_pred = poly_model.predict(X_temp_test)

    print("\n" + "=" * 70)
    print("Polynomial Regression: bikers ~ temp + temp^2")
    print("=" * 70)

    print("RMSE:", round(np.sqrt(mean_squared_error(y_temp_test, y_poly_pred)), 4))
    print("R2:", round(r2_score(y_temp_test, y_poly_pred), 4))


# ============================================================
# 11. DECISION TREE REGRESSOR
# ============================================================

results.append(
    evaluate_regression(
        "Decision Tree Regressor",
        DecisionTreeRegressor(max_depth=5, random_state=42),
        X_train,
        X_test
    )
)


# ============================================================
# 12. RANDOM FOREST REGRESSOR
# ============================================================

results.append(
    evaluate_regression(
        "Random Forest Regressor",
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        X_train,
        X_test
    )
)


# ============================================================
# 13. SVR WITH RBF KERNEL
# ============================================================

results.append(
    evaluate_regression(
        "SVR with RBF Kernel",
        SVR(kernel="rbf", C=10, gamma="scale"),
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 14. GRADIENT BOOSTING REGRESSOR
# ============================================================

results.append(
    evaluate_regression(
        "Gradient Boosting Regressor",
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
        X_train,
        X_test
    )
)


# ============================================================
# 15. XGBOOST REGRESSOR
# ============================================================

try:
    from xgboost import XGBRegressor

    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    results.append(
        evaluate_regression(
            "XGBoost Regressor",
            xgb_model,
            X_train,
            X_test
        )
    )

except Exception as e:
    print("\nXGBoost not available.")
    print("Install using: pip install xgboost")
    print("Error:", e)


# ============================================================
# 16. REGRESSION MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("REGRESSION MODEL COMPARISON")
print("=" * 70)

print(results_df.sort_values(by="Test RMSE"))


# ============================================================
# 17. 10-FOLD CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("10-FOLD CROSS VALIDATION")
print("=" * 70)

kf = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

cv_models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

for name, model in cv_models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="neg_root_mean_squared_error"
    )

    rmse_scores = -scores

    print("\nModel:", name)
    print("RMSE Scores:", rmse_scores)
    print("Mean RMSE:", round(rmse_scores.mean(), 4))
    print("Std RMSE :", round(rmse_scores.std(), 4))


# ============================================================
# 18. PCA
# ============================================================

X_scaled_full = StandardScaler().fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled_full)

pve = pca.explained_variance_ratio_
cumulative_pve = np.cumsum(pve)

print("\n" + "=" * 70)
print("PCA RESULTS")
print("=" * 70)

print("Explained variance ratio:")
print(pve)

print("\nCumulative explained variance:")
print(cumulative_pve)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pve) + 1), cumulative_pve, marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Bikeshare PCA Cumulative Explained Variance")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Bikeshare PCA Plot Colored by Number of Bikers")
plt.colorbar(label="Bikers")
plt.grid(True)
plt.show()


# ============================================================
# 19. K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled_full)

Bikeshare["KMeans_Cluster"] = kmeans_labels

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)

print(Bikeshare["KMeans_Cluster"].value_counts())

print("\nMean bikers by cluster:")
print(Bikeshare.groupby("KMeans_Cluster")["bikers"].mean())

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering on Bikeshare Dataset")
plt.grid(True)
plt.show()


# ============================================================
# 20. HIERARCHICAL CLUSTERING
# ============================================================

sample_size = min(50, X_scaled_full.shape[0])
X_sample = X_scaled_full[:sample_size]

hc = linkage(
    X_sample,
    method="complete",
    metric="euclidean"
)

plt.figure(figsize=(12, 6))
dendrogram(hc)
plt.title("Hierarchical Clustering Dendrogram - First 50 Observations")
plt.xlabel("Observation Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()


# ============================================================
# 21. CLASSIFICATION VERSION
# Convert bikers into high/low
# ============================================================

Bikeshare["bikers_high"] = (
    Bikeshare["bikers"] > Bikeshare["bikers"].median()
).astype(int)

X_class = Bikeshare.drop(
    columns=["bikers", "bikers_high", "KMeans_Cluster"]
)

y_class = Bikeshare["bikers_high"]

X_class = pd.get_dummies(X_class, drop_first=True)

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


def evaluate_classifier(model_name, model, X_train_data, X_test_data):

    model.fit(X_train_data, yc_train)

    y_pred = model.predict(X_test_data)

    acc = accuracy_score(yc_test, y_pred)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("Accuracy:", round(acc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(yc_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(yc_test, y_pred))

    return {
        "Model": model_name,
        "Accuracy": acc
    }


classification_results = []

classification_results.append(
    evaluate_classifier(
        "Logistic Regression Classifier",
        LogisticRegression(max_iter=5000),
        Xc_train_scaled,
        Xc_test_scaled
    )
)

classification_results.append(
    evaluate_classifier(
        "SVM RBF Classifier",
        SVC(kernel="rbf", C=1, gamma="scale"),
        Xc_train_scaled,
        Xc_test_scaled
    )
)

classification_results.append(
    evaluate_classifier(
        "Decision Tree Classifier",
        DecisionTreeClassifier(max_depth=5, random_state=42),
        Xc_train,
        Xc_test
    )
)

classification_results.append(
    evaluate_classifier(
        "Random Forest Classifier",
        RandomForestClassifier(n_estimators=200, random_state=42),
        Xc_train,
        Xc_test
    )
)

classification_results_df = pd.DataFrame(classification_results)

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 70)

print(classification_results_df.sort_values(by="Accuracy", ascending=False))


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("""
Implemented on Bikeshare dataset:

Regression:
1. Simple Linear Regression
2. Multiple Linear Regression
3. Polynomial Regression
4. Ridge Regression
5. Lasso Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. SVR with RBF Kernel
9. Gradient Boosting Regressor
10. XGBoost Regressor, if installed
11. 10-fold Cross Validation

Unsupervised Learning:
12. PCA
13. K-Means Clustering
14. Hierarchical Clustering

Classification:
15. Converted bikers into high/low rental class
16. Logistic Regression Classifier
17. SVM Classifier
18. Decision Tree Classifier
19. Random Forest Classifier

Bikeshare is mainly a regression dataset because bikers is a continuous count variable.
""")