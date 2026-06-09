# bagging classifier using sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score , classification_report

iris = load_iris()
X,y = iris.data, iris.target
feature_names = iris.feature_names
df = pd.DataFrame(X, columns = feature_names)

# EDA
print("Dataset Overview:")
print(df.head())
print(df.info())
print("Summary Statistics:")
print(df.describe())
print("Unique Values:")
print(df.nunique())
print("column names : ")
print(df.columns)
print("Is there any missing values: ")
print(df.isnull().sum())
print("Duplicate values: ")
print(df.duplicated().sum())

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Train the model
Scaler = StandardScaler()
X_train_scaled  = Scaler.fit_transform(X_train)
X_test_scaled = Scaler.transform(X_test)

# Training the model
bag_clf = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=100, max_features=1.0, max_samples=1.0, bootstrap_features=False, bootstrap=True, random_state=42, n_jobs=-1)
bag_clf.fit(X_train_scaled, y_train)
y_pred = bag_clf.predict(X_test_scaled)
print("Predicted y-values: ")
print(y_pred)

# Evaluate the model -
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy * 100, "%")
class_report = classification_report(y_test, y_pred)
print("Classification Report: ")
print(class_report)



#======================================================================
# Implement Bagging from scratch
#======================================================================

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score


# -----------------------------
# Load Data
# -----------------------------
def load_data():
    data = load_diabetes(as_frame=True)
    return data.data, data.target


# -----------------------------
# Tree Builder (Improved)
# -----------------------------
def build_tree(X, y, max_depth=5, min_samples=10, depth=0, max_features=None):
    n_samples = len(y)

    if (
        n_samples <= min_samples
        or depth >= max_depth
        or len(np.unique(y)) == 1
    ):
        return np.mean(y)

    features = X.columns

    # Feature subsampling
    if max_features:
        features = np.random.choice(features, max_features, replace=False)

    best_feature, best_threshold = None, None
    min_error = float("inf")

    for feature in features:
        values = np.sort(X[feature].values)
        thresholds = (values[:-1] + values[1:]) / 2  # midpoint splits

        for t in thresholds:
            left = X[feature] < t
            right = X[feature] >= t

            if left.sum() == 0 or right.sum() == 0:
                continue

            y_left = y[left]
            y_right = y[right]

            error = (
                np.sum((y_left - y_left.mean()) ** 2)
                + np.sum((y_right - y_right.mean()) ** 2)
            )

            if error < min_error:
                min_error = error
                best_feature = feature
                best_threshold = t

    if best_feature is None:
        return np.mean(y)

    left = X[best_feature] < best_threshold
    right = X[best_feature] >= best_threshold

    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": build_tree(X[left], y[left], max_depth, min_samples, depth+1, max_features),
        "right": build_tree(X[right], y[right], max_depth, min_samples, depth+1, max_features),
    }


# -----------------------------
# Prediction
# -----------------------------
def predict_tree(tree, row):
    while isinstance(tree, dict):
        if row[tree["feature"]] < tree["threshold"]:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return tree


def predict_forest(trees, X):
    preds = np.array([
        X.apply(lambda row: predict_tree(tree, row), axis=1)
        for tree in trees
    ])
    return preds.mean(axis=0)


# -----------------------------
# Bagging
# -----------------------------
def bagging(X_train, y_train, n_trees=10, max_depth=10, max_features=None):
    trees = []
    n = len(X_train)

    for _ in range(n_trees):
        # Bootstrap sampling
        indices = np.random.choice(n, n, replace=True)
        X_sample = X_train.iloc[indices]
        y_sample = y_train.iloc[indices]

        tree = build_tree(
            X_sample,
            y_sample,
            max_depth=max_depth,
            max_features=max_features
        )
        trees.append(tree)

    return trees


# -----------------------------
# Cross Validation
# -----------------------------
def kfold_evaluate(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        trees = bagging(X_train, y_train, n_trees=10, max_depth=5, max_features=5)
        y_pred = predict_forest(trees, X_val)

        scores.append(r2_score(y_val, y_pred))

    return np.mean(scores)


# -----------------------------
# Main
# -----------------------------
def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trees = bagging(X_train, y_train, n_trees=10, max_depth=5, max_features=5)

    y_pred = predict_forest(trees, X_test)

    print("\n--- Test Performance ---")
    print("R2 Score:", r2_score(y_test, y_pred))

    print("\n--- Cross Validation ---")
    cv_score = kfold_evaluate(X_train, y_train)
    print("CV R2:", cv_score)


if __name__ == "__main__":
    main()


#======================================================================
# Implement Bagging using Random Forest
#======================================================================


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()    # loading the iris dataset
X, y = iris.data, iris.target    # X: features, y: target
feature_names = iris.feature_names    # column names
df = pd.DataFrame(X, columns=feature_names)    # converting into dataframe


# Split the dataset into 70% train set & 30% test set -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling -
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the model -
random_forest_clf = RandomForestClassifier(criterion='entropy', max_depth=3, n_estimators=100, max_features='sqrt', bootstrap=True, random_state=42, n_jobs=-1)
random_forest_clf.fit(X_train_scaled, y_train)
y_pred = random_forest_clf.predict(X_test_scaled)
print("Predicted y-values: ")
print(y_pred)

# Evaluate the model -
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy*100,"%")
class_report = classification_report(y_test, y_pred)
print("Classification Report: ")
print(class_report)

#==================================================================
### Implement Adaboost Classifier from scratch using the iris dataset.
#==================================================================


import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data():
    data = pd.read_csv("Iris.csv")
    X = data.drop(columns=["Species", "Id"])   # Drop non-feature columns
    y = data['Species']   # Target variable (label)
    return X, y

def data_processing(X_train, X_test, y_train, y_test):
    encoder = LabelEncoder()
    y_encoded_train = encoder.fit_transform(y_train)
    y_encoded_test = encoder.transform(y_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_encoded_train, y_encoded_test, encoder

class adaboost_classifier:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators   # No. of boosting rounds
        self.alphas = []   # List to store alpha values for each weak learner.
        self.models = []   # List to store weak learners,
        self.classes = None  # List of unique classes

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)  # Get unique class labels
        K = len(self.classes)  # Total no. of classes (there are 3 in iris dataset)
        w = np.ones(n_samples) / n_samples   # Initialize uniform sample weights

        for t in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)  # Weak learner (decision stump)
            model.fit(X, y, sample_weight=w)   # Train with current weights
            y_pred = model.predict(X)   # Predict on training data

            incorrect = (y_pred != y).astype(int)   # True if wrong - converts to 1 (misclassified), False if correct - converts to 0 (correctly classified).
            err_t = np.dot(w, incorrect) / np.sum(w)  # Computes the weighted error of the current weak learner.
            err_t = max(err_t, 1e-10)   # To avoid division by 0 or taking log of 0 in the next step, we cap the min. error to a small positive value (1e-10).
            if err_t >= 1 - 1e-10:   # If error is too high, stop boosting.
                break
            # Compute alpha (weight of weak learner)
            alpha_t = np.log((1 - err_t) / err_t) + np.log(K - 1)   # err_t: How bad the learner is , (1 - err_t) / err_t: Higher if the learner is good.

            # Update weights for next iteration -
            for i in range(len(w)):
                if incorrect[i] == 1:
                    w[i] *= np.exp(alpha_t)  # If a sample was misclassified, its weight increases — so it's more likely to be chosen next round.
                else:
                    w[i] *= np.exp(-alpha_t)  # If a sample was correctly classified, its weight decreases.
            # Normalize weights to maintain a probability distribution
            w /= np.sum(w)
            self.models.append(model)   # Store the trained weak learner (stump)
            self.alphas.append(alpha_t)  # Store alpha_t (its importance for final prediction)

    def predict(self, X):
        pred = np.zeros((X.shape[0], len(self.classes)))   # Creates a 2D array to accumulate scores for each class. Each cell will store the sum of alpha values for that class.
        for alpha, model in zip(self.alphas, self.models):  # Iterates over each trained model and its alpha.
            y_pred = model.predict(X)   # Predicts the class for all the samples.
            pred[np.arange(X.shape[0]), y_pred] += alpha  # Adds that model’s weight (alpha) to the predicted class score for each sample.
        return self.classes[np.argmax(pred, axis=1)]   # gets the index (class) with the highest vote score ; self.classes[] maps that index back to the original class label

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4343)
    X_train_scaled, X_test_scaled, y_train, y_test, y_encoder = data_processing(X_train, X_test, y_train, y_test)

    print("\nScikit-learn AdaBoost Classifier:")
    weak_learner = DecisionTreeClassifier(max_depth=1)
    adaboost = AdaBoostClassifier(estimator=weak_learner, n_estimators=50, random_state=42)
    adaboost.fit(X_train_scaled, y_train)
    y_pred_sklearn = adaboost.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_sklearn, target_names=y_encoder.classes_))

    print("\nCustom AdaBoost Classifier:")
    custom_adaboost = adaboost_classifier(n_estimators=50)
    custom_adaboost.fit(X_train_scaled, y_train)
    y_pred_custom = custom_adaboost.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_custom, target_names=y_encoder.classes_))

if __name__ == "__main__":
    main()


#======================================================================
# Implement Adaboost classifier using scikit-learn. Use the Iris dataset.
#======================================================================


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=50,   # n_estimators: no. of weak learners
    learning_rate=0.1, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("AdaBoost Classifier Accuracy (Iris):",accuracy*100,"%")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))    # This will show a matrix of true positives, false positives, etc., which helps in understanding the misclassifications.
