# ============================================================
# Heart Dataset - Logistic Regression Classifier
# Accuracy, Precision, Sensitivity, Specificity, F1-score,
# ROC Curve and AUC
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    classification_report
)


# ============================================================
# 1. LOAD LOCAL heart.csv FILE
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "Heart.csv")

heart = pd.read_csv(file_path)

print("First 5 rows:")
print(heart.head())

print("\nShape before cleaning:")
print(heart.shape)

print("\nColumns:")
print(heart.columns)

print("\nMissing values before cleaning:")
print(heart.isnull().sum())


# ============================================================
# 2. CLEAN DATA
# ============================================================

# Remove unnecessary index column if present
if "Unnamed: 0" in heart.columns:
    heart = heart.drop(columns=["Unnamed: 0"])

# Drop missing values
# Ca has 4 missing values and Thal has 2 missing values
heart = heart.dropna()

print("\nShape after dropping missing values:")
print(heart.shape)

print("\nMissing values after cleaning:")
print(heart.isnull().sum())


# ============================================================
# 3. DEFINE TARGET AND FEATURES
# ============================================================

target_column = "AHD"

# Convert target:
# No  = 0
# Yes = 1
heart[target_column] = heart[target_column].map({
    "No": 0,
    "Yes": 1
})

X = heart.drop(columns=[target_column])
y = heart[target_column]

print("\nTarget value counts:")
print(y.value_counts())


# ============================================================
# 4. ENCODE CATEGORICAL COLUMNS
# ============================================================
# ChestPain and Thal are categorical columns

X = pd.get_dummies(X, drop_first=True)

print("\nFeature columns after encoding:")
print(X.columns)

print("\nFinal feature shape:")
print(X.shape)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ============================================================
# 6. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 7. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(max_iter=5000)

model.fit(X_train_scaled, y_train)

# Probability of class 1 = Heart disease Yes
y_prob = model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# 8. FUNCTION TO CALCULATE METRICS
# ============================================================

def calculate_metrics(y_true, y_probability, threshold):
    y_pred = (y_probability >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "Threshold": threshold,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1-score": f1,
        "Confusion_Matrix": cm
    }


# ============================================================
# 9. MULTIPLE THRESHOLDS
# ============================================================

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

all_results = []

print("\n" + "=" * 70)
print("METRICS AT DIFFERENT THRESHOLDS")
print("=" * 70)

for threshold in thresholds:
    result = calculate_metrics(y_test, y_prob, threshold)
    all_results.append(result)

    print("\nThreshold:", threshold)

    print("Confusion Matrix:")
    print(result["Confusion_Matrix"])

    print("Accuracy    :", round(result["Accuracy"], 4))
    print("Precision   :", round(result["Precision"], 4))
    print("Sensitivity :", round(result["Sensitivity"], 4))
    print("Specificity :", round(result["Specificity"], 4))
    print("F1-score    :", round(result["F1-score"], 4))


# ============================================================
# 10. SUMMARY TABLE
# ============================================================

results_df = pd.DataFrame(all_results)

summary_table = results_df[
    [
        "Threshold",
        "TP",
        "TN",
        "FP",
        "FN",
        "Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "F1-score"
    ]
]

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

print(summary_table)


# ============================================================
# 11. CLASSIFICATION REPORT AT DEFAULT THRESHOLD 0.5
# ============================================================

y_pred_default = (y_prob >= 0.5).astype(int)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT AT THRESHOLD 0.5")
print("=" * 70)

print(classification_report(y_test, y_pred_default, target_names=["No Disease", "Disease"]))


# ============================================================
# 12. ROC CURVE AND AUC
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

print("\nAUC:", round(roc_auc, 4))

plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, label=f"Logistic Regression AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Sensitivity")
plt.title("ROC Curve - Heart Disease Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 13. BEST THRESHOLD BY F1-SCORE
# ============================================================

best_row = summary_table.loc[summary_table["F1-score"].idxmax()]

print("\n" + "=" * 70)
print("BEST THRESHOLD BASED ON F1-SCORE")
print("=" * 70)

print(best_row)


# ============================================================
# 14. FINAL COMMENT
# ============================================================

print("""
Final Comment:

The target column is AHD.
AHD = No means no heart disease and was encoded as 0.
AHD = Yes means heart disease and was encoded as 1.

Logistic Regression was trained to predict the probability of heart disease.
Different thresholds were used to convert probabilities into class labels.

At lower thresholds, sensitivity generally increases because more patients
are predicted as disease positive.

At higher thresholds, specificity generally increases because fewer patients
are predicted as disease positive.

ROC curve shows the trade-off between sensitivity and false positive rate.
AUC gives the overall ability of the model to separate disease and non-disease cases.
""")