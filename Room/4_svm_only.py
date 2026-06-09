# ============================================================
# 4_svm_only.py
# Short SVM Classification Script for Any CSV Dataset
#
# Models:
# Linear SVM
# RBF SVM
# Polynomial SVM
#
# Change only:
# file_name
# target_col
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 1. Load Dataset
# ============================================================

file_name = "your_dataset.csv"   # CHANGE THIS
target_col = "target"            # CHANGE THIS

df = pd.read_csv(file_name)


# ============================================================
# 2. Basic Cleaning
# ============================================================

for col in ["Unnamed: 0", "ID", "id", "index"]:
    if col in df.columns:
        df = df.drop(columns=[col])


# ============================================================
# 3. Basic EDA
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget counts:")
print(df[target_col].value_counts())


# ============================================================
# 4. Separate X and y
# ============================================================

X = df.drop(columns=[target_col])
y = df[target_col]


# ============================================================
# 5. Handle Missing Values
# ============================================================

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

if y.isnull().sum() > 0:
    y = y.fillna(y.mode()[0])


# ============================================================
# 6. Encode Features and Target
# ============================================================

X = pd.get_dummies(X, drop_first=True)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nTarget encoding:")
for original_class, encoded_value in zip(
    label_encoder.classes_,
    range(len(label_encoder.classes_))
):
    print(original_class, "->", encoded_value)

print("\nFeature shape after encoding:")
print(X.shape)


# ============================================================
# 7. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# 8. SVM Models and Parameter Grids
# ============================================================

svm_models = {

    "Linear SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear"))
        ]),
        {
            "svm__C": [0.01, 0.1, 1, 10, 100]
        }
    ),

    "RBF SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf"))
        ]),
        {
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", "auto", 0.01, 0.1, 1]
        }
    ),

    "Polynomial SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="poly"))
        ]),
        {
            "svm__C": [0.1, 1, 10],
            "svm__degree": [2, 3, 4],
            "svm__gamma": ["scale", "auto"]
        }
    )
}


# ============================================================
# 9. GridSearchCV and Evaluation
# ============================================================

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

results = []

for model_name, (model, param_grid) in svm_models.items():

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("Best Parameters:")
    print(grid.best_params_)

    print("\nBest CV Accuracy:", round(grid.best_score_, 4))
    print("Test Accuracy   :", round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    results.append({
        "Model": model_name,
        "Best CV Accuracy": grid.best_score_,
        "Test Accuracy": accuracy,
        "Best Parameters": grid.best_params_
    })


# ============================================================
# 10. Final Comparison
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Test Accuracy",
    ascending=False
)

print("\n" + "=" * 90)
print("FINAL SVM COMPARISON")
print("=" * 90)

print(results_df)


# ============================================================
# 11. Best SVM Model
# ============================================================

best_row = results_df.iloc[0]

print("\n" + "=" * 90)
print("BEST SVM MODEL")
print("=" * 90)

print("Best Model:", best_row["Model"])
print("Best Test Accuracy:", round(best_row["Test Accuracy"], 4))
print("Best Parameters:", best_row["Best Parameters"])


# ============================================================
# 12. Final Comment
# ============================================================

print("""
Final Comment:

SVM classification was performed using three kernels:

1. Linear kernel
2. RBF kernel
3. Polynomial kernel

Important SVM parameters:
C      = regularization / margin softness
gamma  = used in RBF and polynomial kernels
degree = used in polynomial kernel

SVM does not use alpha. Alpha is used in Ridge/Lasso/SGD-type models.

The best SVM model was selected based on the highest test accuracy.
""")
