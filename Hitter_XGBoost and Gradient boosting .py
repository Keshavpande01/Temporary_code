# ============================================
# Hitters Dataset - XGBoost Regression
# ============================================

# -------------------------------
# Import libraries
# -------------------------------
from ISLP import load_data
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# -------------------------------
# 1. Load Dataset
# -------------------------------
Hitters = load_data("Hitters")

# Drop missing values (VERY IMPORTANT)
Hitters = Hitters.dropna()

# Target = Salary
y = Hitters["Salary"]

# Features
X = Hitters.drop("Salary", axis=1)

# -------------------------------
# 2. Convert categorical → numeric
# -------------------------------
X = pd.get_dummies(X, drop_first=True)

# -------------------------------
# 3. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# 4. XGBoost Model
# -------------------------------
xgb = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

xgb.fit(X_train, y_train)

# -------------------------------
# 5. Predictions
# -------------------------------
y_pred = xgb.predict(X_test)

# -------------------------------
# 6. Evaluation
# -------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== XGBoost Regression (Hitters) =====")
print("RMSE:", rmse)
print("R2 Score:", r2)


# ============================================
# Hitters Dataset - Gradient Boosting
# ============================================

# -------------------------------
# Import libraries
# -------------------------------
from ISLP import load_data
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load Data
# -------------------------------
Hitters = load_data("Hitters")

# Drop missing values
Hitters = Hitters.dropna()

# Target
y = Hitters["Salary"]

# Features
X = Hitters.drop("Salary", axis=1)

# Convert categorical → numeric
X = pd.get_dummies(X, drop_first=True)

# -------------------------------
# 2. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# 3. Gradient Boosting Model
# -------------------------------
gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)

# -------------------------------
# 4. Prediction
# -------------------------------
y_pred = gb.predict(X_test)

# -------------------------------
# 5. Evaluation
# -------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== GRADIENT BOOSTING =====")
print("RMSE:", rmse)
print("R2 Score:", r2)