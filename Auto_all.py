# ============================================================
# Auto.csv Complete Machine Learning Practical
# Regression + PCA + Clustering + Classification
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram


# ============================================================
# 1. LOAD Auto.csv
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "Auto.csv")

Auto = pd.read_csv(file_path)

print("\nFirst 5 rows:")
print(Auto.head())

print("\nShape:")
print(Auto.shape)

print("\nColumns:")
print(Auto.columns)

print("\nMissing values before cleaning:")
print(Auto.isnull().sum())


# ============================================================
# 2. DATA CLEANING
# ============================================================
# horsepower may contain '?' in Auto.csv

Auto["horsepower"] = pd.to_numeric(Auto["horsepower"], errors="coerce")

Auto = Auto.dropna()

# Drop name column because it is text and not useful directly
if "name" in Auto.columns:
    Auto = Auto.drop(columns=["name"])

print("\nShape after cleaning:")
print(Auto.shape)

print("\nMissing values after cleaning:")
print(Auto.isnull().sum())

print("\nData types:")
print(Auto.dtypes)


# ============================================================
# 3. BASIC EDA
# ============================================================

print("\nSummary statistics:")
print(Auto.describe())

print("\nCorrelation with mpg:")
print(Auto.corr(numeric_only=True)["mpg"].sort_values(ascending=False))

plt.figure(figsize=(7, 5))
plt.scatter(Auto["horsepower"], Auto["mpg"])
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("MPG vs Horsepower")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(Auto["weight"], Auto["mpg"])
plt.xlabel("Weight")
plt.ylabel("MPG")
plt.title("MPG vs Weight")
plt.grid(True)
plt.show()


# ============================================================
# 4. REGRESSION TARGET AND FEATURES
# ============================================================

X = Auto.drop(columns=["mpg"])
y = Auto["mpg"]

# Treat origin as categorical if present
if "origin" in X.columns:
    X["origin"] = X["origin"].astype("category")

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

print("\nFeature shape after encoding:")
print(X_encoded.shape)


# ============================================================
# 5. TRAIN-TEST SPLIT FOR REGRESSION
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.30,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. FUNCTION FOR REGRESSION EVALUATION
# ============================================================

def evaluate_regression_model(model_name, model, X_train_data, X_test_data):
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


regression_results = []


# ============================================================
# 7. SIMPLE LINEAR REGRESSION: mpg ~ horsepower
# ============================================================

X_hp = Auto[["horsepower"]]
y_hp = Auto["mpg"]

X_hp_train, X_hp_test, y_hp_train, y_hp_test = train_test_split(
    X_hp,
    y_hp,
    test_size=0.30,
    random_state=42
)

simple_lr = LinearRegression()
simple_lr.fit(X_hp_train, y_hp_train)

y_hp_pred = simple_lr.predict(X_hp_test)

print("\n" + "=" * 70)
print("Simple Linear Regression: mpg ~ horsepower")
print("=" * 70)
print("Intercept:", simple_lr.intercept_)
print("Slope:", simple_lr.coef_[0])
print("RMSE:", np.sqrt(mean_squared_error(y_hp_test, y_hp_pred)))
print("R2:", r2_score(y_hp_test, y_hp_pred))

plt.figure(figsize=(7, 5))
plt.scatter(X_hp_test, y_hp_test, label="Test Data")
plt.plot(X_hp_test, y_hp_pred, color="red", label="Regression Line")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Simple Linear Regression: MPG vs Horsepower")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 8. MULTIPLE REGRESSION MODELS
# ============================================================

regression_results.append(
    evaluate_regression_model(
        "Multiple Linear Regression",
        LinearRegression(),
        X_train_scaled,
        X_test_scaled
    )
)

regression_results.append(
    evaluate_regression_model(
        "Ridge Regression",
        Ridge(alpha=1.0),
        X_train_scaled,
        X_test_scaled
    )
)

regression_results.append(
    evaluate_regression_model(
        "Lasso Regression",
        Lasso(alpha=0.01, max_iter=10000),
        X_train_scaled,
        X_test_scaled
    )
)

regression_results.append(
    evaluate_regression_model(
        "Decision Tree Regressor",
        DecisionTreeRegressor(max_depth=4, random_state=42),
        X_train,
        X_test
    )
)

regression_results.append(
    evaluate_regression_model(
        "Random Forest Regressor",
        RandomForestRegressor(n_estimators=200, random_state=42),
        X_train,
        X_test
    )
)

regression_results.append(
    evaluate_regression_model(
        "SVR with RBF Kernel",
        SVR(kernel="rbf", C=10, gamma="scale"),
        X_train_scaled,
        X_test_scaled
    )
)

regression_results.append(
    evaluate_regression_model(
        "Gradient Boosting Regressor",
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
        X_train,
        X_test
    )
)


# ============================================================
# 9. XGBOOST REGRESSOR
# ============================================================

try:
    from xgboost import XGBRegressor

    xgb_reg = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    regression_results.append(
        evaluate_regression_model(
            "XGBoost Regressor",
            xgb_reg,
            X_train,
            X_test
        )
    )

except Exception as e:
    print("\nXGBoost not available.")
    print("Install using: pip install xgboost")
    print("Error:", e)


# ============================================================
# 10. POLYNOMIAL REGRESSION: mpg ~ horsepower + horsepower^2
# ============================================================

poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

poly_model.fit(X_hp_train, y_hp_train)

y_poly_pred = poly_model.predict(X_hp_test)

print("\n" + "=" * 70)
print("Polynomial Regression: mpg ~ horsepower + horsepower^2")
print("=" * 70)
print("RMSE:", round(np.sqrt(mean_squared_error(y_hp_test, y_poly_pred)), 4))
print("R2:", round(r2_score(y_hp_test, y_poly_pred), 4))


# ============================================================
# 11. REGRESSION MODEL COMPARISON
# ============================================================

regression_results_df = pd.DataFrame(regression_results)

print("\n" + "=" * 70)
print("REGRESSION MODEL COMPARISON")
print("=" * 70)

print(regression_results_df.sort_values(by="Test RMSE"))


# ============================================================
# 12. K-FOLD CROSS VALIDATION FOR REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("10-FOLD CROSS VALIDATION FOR REGRESSION")
print("=" * 70)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

cv_models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

for name, model in cv_models.items():
    scores = cross_val_score(
        model,
        X_encoded,
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
# 13. PCA ON AUTO FEATURES
# ============================================================

X_scaled_full = StandardScaler().fit_transform(X_encoded)

pca = PCA()
X_pca = pca.fit_transform(X_scaled_full)

pve = pca.explained_variance_ratio_
cum_pve = np.cumsum(pve)

print("\n" + "=" * 70)
print("PCA RESULTS")
print("=" * 70)

print("Explained variance ratio:")
print(pve)

print("\nCumulative explained variance:")
print(cum_pve)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pve) + 1), cum_pve, marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Auto["mpg"])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Plot of Auto Dataset Colored by MPG")
plt.colorbar(label="MPG")
plt.grid(True)
plt.show()


# ============================================================
# 14. K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled_full)

Auto["KMeans_Cluster"] = kmeans_labels

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)

print(Auto["KMeans_Cluster"].value_counts())

print("\nMean MPG by KMeans cluster:")
print(Auto.groupby("KMeans_Cluster")["mpg"].mean())

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering on Auto Dataset")
plt.grid(True)
plt.show()


# ============================================================
# 15. HIERARCHICAL CLUSTERING
# ============================================================

sample_size = min(50, X_scaled_full.shape[0])
X_sample = X_scaled_full[:sample_size]

linked = linkage(X_sample, method="complete", metric="euclidean")

plt.figure(figsize=(12, 6))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram - First 50 Cars")
plt.xlabel("Car Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()


# ============================================================
# 16. CLASSIFICATION VERSION
# Convert mpg into high/low mileage
# ============================================================

Auto["mpg_high"] = (Auto["mpg"] > Auto["mpg"].median()).astype(int)

X_class = Auto.drop(columns=["mpg", "mpg_high", "KMeans_Cluster"])
y_class = Auto["mpg_high"]

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
        DecisionTreeClassifier(max_depth=4, random_state=42),
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
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("""
Implemented on Auto.csv:

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
10. XGBoost Regressor if installed
11. 10-fold Cross Validation

Unsupervised Learning:
12. PCA
13. K-Means Clustering
14. Hierarchical Clustering

Classification:
15. Converted mpg into high/low mileage
16. Logistic Regression Classifier
17. SVM Classifier
18. Decision Tree Classifier
19. Random Forest Classifier

Auto.csv is mainly a regression dataset because mpg is continuous.
However, by converting mpg into high/low, we can also use it for classification.
""")