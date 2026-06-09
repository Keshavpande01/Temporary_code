# ============================================
# Default Dataset - Linear SVM vs RBF SVM
# ============================================

# -------------------------------
# Import libraries
# -------------------------------
from ISLP import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, roc_curve, auc

# -------------------------------
# 1. Load Data
# -------------------------------
Default = load_data("Default")

# Features and target
X = Default[["balance", "income"]]
y = (Default["default"] == "Yes").astype(int)

# -------------------------------
# 2. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# 3. Standardization
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 4. Linear SVM
# -------------------------------
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train_scaled, y_train)

y_pred_linear = linear_svm.predict(X_test_scaled)

print("\n===== LINEAR SVM =====")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_linear))
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Classification Report:\n", classification_report(y_test, y_pred_linear))

# -------------------------------
# 5. RBF Kernel SVM
# -------------------------------
rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train_scaled, y_train)

y_pred_rbf = rbf_svm.predict(X_test_scaled)

print("\n===== RBF SVM =====")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rbf))
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("Classification Report:\n", classification_report(y_test, y_pred_rbf))

# -------------------------------
# 6. ROC Curve (RBF SVM)
# -------------------------------
y_score = rbf_svm.decision_function(X_test_scaled)

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # random classifier line

plt.title("ROC Curve (RBF SVM)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# ============================================
# END
# ============================================