# ============================================================
# Normal EDA Code
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Load dataset
# ============================================================

df = pd.read_csv("USArrests.csv")   # change file name here


# ============================================================
# 2. First look at data
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())


# ============================================================
# 3. Shape of dataset
# ============================================================

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 4. Column names
# ============================================================

print("\nColumn names:")
print(df.columns)


# ============================================================
# 5. Data types
# ============================================================

print("\nData types:")
print(df.dtypes)


# ============================================================
# 6. Dataset information
# ============================================================

print("\nDataset info:")
print(df.info())


# ============================================================
# 7. Missing values
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 8. Duplicate rows
# ============================================================

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 9. Summary statistics
# ============================================================

print("\nSummary statistics:")
print(df.describe())


# ============================================================
# 10. Categorical summary
# ============================================================

categorical_cols = df.select_dtypes(include=["object"]).columns

if len(categorical_cols) > 0:
    print("\nCategorical summary:")
    print(df[categorical_cols].describe())
else:
    print("\nNo categorical columns found.")


# ============================================================
# 11. Value counts for categorical columns
# ============================================================

if len(categorical_cols) > 0:
    for col in categorical_cols:
        print("\nValue counts for:", col)
        print(df[col].value_counts())


# ============================================================
# 12. Numeric columns
# ============================================================

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

print("\nNumeric columns:")
print(numeric_cols)


# ============================================================
# 13. Histograms for numeric columns
# ============================================================

if len(numeric_cols) > 0:
    df[numeric_cols].hist(figsize=(15, 10), bins=20)
    plt.suptitle("Histograms of Numeric Columns")
    plt.tight_layout()
    plt.show()


# ============================================================
# 14. Boxplots for numeric columns
# ============================================================

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[col].dropna())
    plt.title("Boxplot of " + col)
    plt.ylabel(col)
    plt.grid(True)
    plt.show()


# ============================================================
# 15. Correlation matrix
# ============================================================

if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()

    print("\nCorrelation matrix:")
    print(corr)


# ============================================================
# 16. Correlation heatmap
# ============================================================

if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# ============================================================
# 17. Final simple summary
# ============================================================

print("\nEDA completed.")