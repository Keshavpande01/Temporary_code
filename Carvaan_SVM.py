# ============================================
# Caravan Dataset - SVM (Linear vs RBF)
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
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# -------------------------------
# 1. Load Dataset
# -------------------------------
Caravan = load_data("Caravan")

# Features and target
X = Caravan.drop("Purchase", axis=1)
y = (Caravan["Purchase"] == "Yes").astype(int)

# -------------------------------
# 2. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1000, random_state=42
)

# -------------------------------
# 3. Standardization (VERY IMPORTANT)
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 4. Linear SVM
# -------------------------------
linear_svm = SVC(kernel='linear', class_weight='balanced')
linear_svm.fit(X_train_scaled, y_train)

y_pred_linear = linear_svm.predict(X_test_scaled)

print("\n===== LINEAR SVM =====")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_linear))
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Classification Report:\n", classification_report(y_test, y_pred_linear))

# -------------------------------
# 5. RBF Kernel SVM
# -------------------------------
rbf_svm = SVC(kernel='rbf', class_weight='balanced')
rbf_svm.fit(X_train_scaled, y_train)

y_pred_rbf = rbf_svm.predict(X_test_scaled)

print("\n===== RBF SVM =====")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rbf))
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("Classification Report:\n", classification_report(y_test, y_pred_rbf))

# ============================================
# Caravan Dataset - Tree Models + Boosting
# ============================================

# -------------------------------
# Import libraries
# -------------------------------
from ISLP import load_data
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Tree models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# -------------------------------
# 1. Load Dataset
# -------------------------------
Caravan = load_data("Caravan")

# Target
y = (Caravan["Purchase"] == "Yes").astype(int)

# Features
X = Caravan.drop("Purchase", axis=1)

# -------------------------------
# 2. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1000, random_state=42
)

# -------------------------------
# 3. Decision Tree
# -------------------------------
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

# -------------------------------
# 4. Random Forest
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# -------------------------------
# 5. Gradient Boosting
# -------------------------------
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)

# -------------------------------
# 6. Evaluation Function
# -------------------------------
def evaluate(name, y_true, y_pred):
    print(f"\n{name}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# -------------------------------
# 7. Results
# -------------------------------
evaluate("Decision Tree", y_test, y_pred_tree)
evaluate("Random Forest", y_test, y_pred_rf)
evaluate("Gradient Boosting", y_test, y_pred_gb)
evaluate("Gradient Boosting", y_test, y_pred_gb)