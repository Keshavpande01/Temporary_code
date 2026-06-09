# ============================================================
# Credit.csv Complete ML Practical
# Regression + PCA + Clustering + Classification
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# 1. LOAD Credit.csv
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "Credit.csv")

Credit = pd.read_csv(file_path)

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)
print(Credit.head())

print("\n" + "=" * 70)
print("SHAPE")
print("=" * 70)
print(Credit.shape)

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)
print(Credit.columns)

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)
print(Credit.dtypes)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
print(Credit.isnull().sum())


# ============================================================
# 2. BASIC CLEANING
# ============================================================

# Remove unnecessary index column if present
if "Unnamed: 0" in Credit.columns:
    Credit = Credit.drop(columns=["Unnamed: 0"])

# Some versions have ID column
if "ID" in Credit.columns:
    Credit = Credit.drop(columns=["ID"])

# Drop missing values if any
Credit = Credit.dropna()

print("\nShape after cleaning:")
print(Credit.shape)


# ============================================================
# 3. BASIC EDA
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(Credit.describe())

print("\n" + "=" * 70)
print("CATEGORICAL VALUE COUNTS")
print("=" * 70)

categorical_cols = Credit.select_dtypes(include=["object", "category"]).columns.tolist()

for col in categorical_cols:
    print("\nColumn:", col)
    print(Credit[col].value_counts())


# ------------------------------------------------------------
# Balance distribution
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))
plt.hist(Credit["Balance"], bins=30, edgecolor="black")
plt.xlabel("Balance")
plt.ylabel("Frequency")
plt.title("Distribution of Credit Card Balance")
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# Balance vs Income
# ------------------------------------------------------------

if "Income" in Credit.columns:
    plt.figure(figsize=(7, 5))
    plt.scatter(Credit["Income"], Credit["Balance"], alpha=0.6)
    plt.xlabel("Income")
    plt.ylabel("Balance")
    plt.title("Balance vs Income")
    plt.grid(True)
    plt.show()


# ------------------------------------------------------------
# Balance vs Limit
# ------------------------------------------------------------

if "Limit" in Credit.columns:
    plt.figure(figsize=(7, 5))
    plt.scatter(Credit["Limit"], Credit["Balance"], alpha=0.6)
    plt.xlabel("Limit")
    plt.ylabel("Balance")
    plt.title("Balance vs Credit Limit")
    plt.grid(True)
    plt.show()


# ============================================================
# 4. REGRESSION TASK
# Target = Balance
# ============================================================

target_column = "Balance"

X = Credit.drop(columns=[target_column])
y = Credit[target_column]

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

print("\n" + "=" * 70)
print("FEATURE MATRIX")
print("=" * 70)
print("Feature shape after encoding:", X_encoded.shape)
print("Feature columns:")
print(X_encoded.columns)


# ============================================================
# 5. TRAIN-TEST SPLIT
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
# 6. REGRESSION EVALUATION FUNCTION
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


regression_results = []


# ============================================================
# 7. SIMPLE LINEAR REGRESSION
# Balance ~ Income
# ============================================================

if "Income" in Credit.columns:
    X_income = Credit[["Income"]]
    y_balance = Credit["Balance"]

    X_income_train, X_income_test, y_income_train, y_income_test = train_test_split(
        X_income,
        y_balance,
        test_size=0.30,
        random_state=42
    )

    simple_lr = LinearRegression()
    simple_lr.fit(X_income_train, y_income_train)

    y_income_pred = simple_lr.predict(X_income_test)

    print("\n" + "=" * 70)
    print("Simple Linear Regression: Balance ~ Income")
    print("=" * 70)

    print("Intercept:", round(simple_lr.intercept_, 4))
    print("Slope:", round(simple_lr.coef_[0], 4))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_income_test, y_income_pred)), 4))
    print("R2:", round(r2_score(y_income_test, y_income_pred), 4))

    plt.figure(figsize=(7, 5))
    plt.scatter(X_income_test, y_income_test, alpha=0.6, label="Test Data")
    plt.plot(X_income_test, y_income_pred, color="red", label="Regression Line")
    plt.xlabel("Income")
    plt.ylabel("Balance")
    plt.title("Simple Linear Regression: Balance vs Income")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# 8. SIMPLE LINEAR REGRESSION
# Balance ~ Limit
# ============================================================

if "Limit" in Credit.columns:
    X_limit = Credit[["Limit"]]
    y_balance = Credit["Balance"]

    X_limit_train, X_limit_test, y_limit_train, y_limit_test = train_test_split(
        X_limit,
        y_balance,
        test_size=0.30,
        random_state=42
    )

    limit_lr = LinearRegression()
    limit_lr.fit(X_limit_train, y_limit_train)

    y_limit_pred = limit_lr.predict(X_limit_test)

    print("\n" + "=" * 70)
    print("Simple Linear Regression: Balance ~ Limit")
    print("=" * 70)

    print("Intercept:", round(limit_lr.intercept_, 4))
    print("Slope:", round(limit_lr.coef_[0], 4))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_limit_test, y_limit_pred)), 4))
    print("R2:", round(r2_score(y_limit_test, y_limit_pred), 4))

    plt.figure(figsize=(7, 5))
    plt.scatter(X_limit_test, y_limit_test, alpha=0.6, label="Test Data")
    plt.plot(X_limit_test, y_limit_pred, color="red", label="Regression Line")
    plt.xlabel("Limit")
    plt.ylabel("Balance")
    plt.title("Simple Linear Regression: Balance vs Limit")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# 9. POLYNOMIAL REGRESSION
# Balance ~ Income + Income^2
# ============================================================

if "Income" in Credit.columns:
    poly_model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("linear", LinearRegression())
    ])

    poly_model.fit(X_income_train, y_income_train)

    y_poly_pred = poly_model.predict(X_income_test)

    print("\n" + "=" * 70)
    print("Polynomial Regression: Balance ~ Income + Income^2")
    print("=" * 70)

    print("RMSE:", round(np.sqrt(mean_squared_error(y_income_test, y_poly_pred)), 4))
    print("R2:", round(r2_score(y_income_test, y_poly_pred), 4))


# ============================================================
# 10. MULTIPLE LINEAR REGRESSION
# ============================================================

regression_results.append(
    evaluate_regression(
        "Multiple Linear Regression",
        LinearRegression(),
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 11. RIDGE REGRESSION
# ============================================================

regression_results.append(
    evaluate_regression(
        "Ridge Regression",
        Ridge(alpha=1.0),
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 12. LASSO REGRESSION
# ============================================================

regression_results.append(
    evaluate_regression(
        "Lasso Regression",
        Lasso(alpha=0.01, max_iter=10000),
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 13. DECISION TREE REGRESSOR
# ============================================================

regression_results.append(
    evaluate_regression(
        "Decision Tree Regressor",
        DecisionTreeRegressor(max_depth=5, random_state=42),
        X_train,
        X_test
    )
)


# ============================================================
# 14. RANDOM FOREST REGRESSOR
# ============================================================

regression_results.append(
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
# 15. SVR WITH RBF KERNEL
# ============================================================

regression_results.append(
    evaluate_regression(
        "SVR with RBF Kernel",
        SVR(kernel="rbf", C=10, gamma="scale"),
        X_train_scaled,
        X_test_scaled
    )
)


# ============================================================
# 16. GRADIENT BOOSTING REGRESSOR
# ============================================================

regression_results.append(
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
# 17. XGBOOST REGRESSOR
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

    regression_results.append(
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
# 18. REGRESSION MODEL COMPARISON
# ============================================================

regression_results_df = pd.DataFrame(regression_results)

print("\n" + "=" * 70)
print("REGRESSION MODEL COMPARISON")
print("=" * 70)

print(regression_results_df.sort_values(by="Test RMSE"))


# ============================================================
# 19. 10-FOLD CROSS VALIDATION
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
# 20. PCA
# ============================================================

X_scaled_full = StandardScaler().fit_transform(X_encoded)

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
plt.title("Credit PCA Cumulative Explained Variance")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Credit PCA Plot Colored by Balance")
plt.colorbar(label="Balance")
plt.grid(True)
plt.show()


# ============================================================
# 21. K-MEANS CLUSTERING
# ============================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled_full)

Credit["KMeans_Cluster"] = kmeans_labels

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)

print(Credit["KMeans_Cluster"].value_counts())

print("\nMean Balance by cluster:")
print(Credit.groupby("KMeans_Cluster")["Balance"].mean())

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering on Credit Dataset")
plt.grid(True)
plt.show()


# ============================================================
# 22. HIERARCHICAL CLUSTERING
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
plt.title("Hierarchical Clustering Dendrogram - First 50 Customers")
plt.xlabel("Customer Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()


# ============================================================
# 23. CLASSIFICATION VERSION
# Convert Balance into high/low balance
# ============================================================

Credit["Balance_high"] = (
    Credit["Balance"] > Credit["Balance"].median()
).astype(int)

X_class = Credit.drop(
    columns=["Balance", "Balance_high", "KMeans_Cluster"]
)

y_class = Credit["Balance_high"]

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
# 24. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("""
Implemented on Credit.csv:

Regression:
1. Simple Linear Regression: Balance ~ Income
2. Simple Linear Regression: Balance ~ Limit
3. Polynomial Regression
4. Multiple Linear Regression
5. Ridge Regression
6. Lasso Regression
7. Decision Tree Regressor
8. Random Forest Regressor
9. SVR with RBF Kernel
10. Gradient Boosting Regressor
11. XGBoost Regressor, if installed
12. 10-fold Cross Validation

Unsupervised Learning:
13. PCA
14. K-Means Clustering
15. Hierarchical Clustering

Classification:
16. Converted Balance into high/low balance
17. Logistic Regression Classifier
18. SVM Classifier
19. Decision Tree Classifier
20. Random Forest Classifier

Credit.csv is mainly a regression dataset because Balance is continuous.
It is useful for studying how income, credit limit, rating and student status
affect credit card balance.
""")