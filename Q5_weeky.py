# ============================================================
# Weekly Dataset - Improved LOOCV Logistic Regression
# Compare different feature sets
# ============================================================

import numpy as np
import pandas as pd

from ISLP import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# 1. Load Weekly dataset
# ============================================================

Weekly = load_data("Weekly")

print("First 5 rows:")
print(Weekly.head())

print("\nShape:")
print(Weekly.shape)

print("\nColumns:")
print(Weekly.columns)

print("\nDirection counts:")
print(Weekly["Direction"].value_counts())


# ============================================================
# 2. Convert target
# ============================================================
# Down = 0
# Up   = 1

y = Weekly["Direction"].map({
    "Down": 0,
    "Up": 1
})


# ============================================================
# 3. Function for manual LOOCV
# ============================================================

def loocv_logistic_regression(X, y):
    n = len(X)
    correct_predictions = []

    for i in range(n):

        # ith observation as test data
        X_test = X.iloc[[i]]
        y_test = y.iloc[i]

        # remaining observations as training data
        X_train = X.drop(index=X.index[i])
        y_train = y.drop(index=y.index[i])

        # Pipeline: scaling + logistic regression
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("log_reg", LogisticRegression(max_iter=5000))
        ])

        # Train model
        model.fit(X_train, y_train)

        # Predict test observation
        y_pred = model.predict(X_test)[0]

        # Store 1 if correct, 0 if wrong
        if y_pred == y_test:
            correct_predictions.append(1)
        else:
            correct_predictions.append(0)

    accuracy = np.mean(correct_predictions)
    error = 1 - accuracy

    return accuracy, error


# ============================================================
# 4. Different feature sets
# ============================================================

feature_sets = {
    "Lag1 + Lag2": ["Lag1", "Lag2"],

    "Lag1 to Lag5": ["Lag1", "Lag2", "Lag3", "Lag4", "Lag5"],

    "Lag1 to Lag5 + Volume": ["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume"],

    "All numeric predictors except Today": [
        "Year", "Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume"
    ]
}


# ============================================================
# 5. Run LOOCV for each feature set
# ============================================================

results = []

print("\n" + "=" * 70)
print("LOOCV RESULTS FOR DIFFERENT FEATURE SETS")
print("=" * 70)

for name, features in feature_sets.items():

    X = Weekly[features]

    accuracy, error = loocv_logistic_regression(X, y)

    results.append({
        "Feature Set": name,
        "Accuracy": accuracy,
        "Error": error
    })

    print("\nFeature Set:", name)
    print("Features:", features)
    print("LOOCV Accuracy:", round(accuracy, 4))
    print("LOOCV Error   :", round(error, 4))


# ============================================================
# 6. Final comparison table
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)

print(results_df.sort_values(by="Accuracy", ascending=False))