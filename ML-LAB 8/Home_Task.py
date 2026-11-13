import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold,
    TimeSeriesSplit, GroupKFold, cross_val_predict, cross_val_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

from imblearn.over_sampling import SMOTE


# In a notebook "display()" exists. In a .py file it does not,
# so we define our own simple version that prints a table.
def display(table):
    print(table.to_string())
    print()


# Folder where this .py file is saved (so the CSV is found easily)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



# 1. IMPORT LIBRARIES AND LOAD DATA

print("=" * 70)
print("1. LOAD DATA")
print("=" * 70)

# Load the Titanic dataset (titanic.csv must be next to this file)
raw_data = pd.read_csv(os.path.join(BASE_DIR, "titanic.csv"))

# These are the actual column names in this file.
feature_columns = ["Pclass", "Gender", "Age", "SibSp", "Parch", "Fare"]
data = raw_data[["Survived"] + feature_columns].copy()

# Keep Ticket separately for Group K-Fold.
# Ticket is a grouping variable, not a model feature.
group_labels = raw_data.loc[data.index, "Ticket"].fillna(
    pd.Series(data.index.astype(str), index=data.index)
).astype(str)

# Encode the actual Gender column.
encoder = LabelEncoder()
data["Gender"] = encoder.fit_transform(data["Gender"].astype(str))

# Fill missing Age values so all 891 rows can be used.
age_imputer = SimpleImputer(strategy="median")
data[["Age"]] = age_imputer.fit_transform(data[["Age"]])

# Separate input features (X) and target (y).
X = data.drop("Survived", axis=1)
y = data["Survived"]

print("Dataset shape:", data.shape)
print("Original CSV shape:", raw_data.shape)
print("Feature columns:", list(X.columns))
display(data.head())

print("\nSurvival counts:")
print(y.value_counts())

print("\nMissing values after preprocessing:")
print(data.isna().sum())

print("\nNumber of ticket groups:", group_labels.nunique())


# A. TRAIN, VALIDATION AND TEST SPLIT

print("\n" + "=" * 70)
print("A. TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

# First, separate 20% test data
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Split the remaining 80% into training (60%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

depth_values = [2, 4, 6, 8, None]
results = []

# Train one Random Forest for each depth
for depth in depth_values:
    model = RandomForestClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    val_accuracy = accuracy_score(y_val, model.predict(X_val))

    results.append([depth, train_accuracy, val_accuracy])

# Show the results in a table
split_results = pd.DataFrame(
    results, columns=["Max Depth", "Training Accuracy", "Validation Accuracy"]
)
display(split_results)

# Select the depth with the highest validation accuracy
best_depth = split_results.loc[
    split_results["Validation Accuracy"].idxmax(), "Max Depth"
]
if pd.isna(best_depth):
    best_depth = None
else:
    best_depth = int(best_depth)
print("Best depth:", best_depth)

# Train the selected model and evaluate it on the test data
best_model = RandomForestClassifier(max_depth=best_depth, random_state=42)
best_model.fit(X_train, y_train)

print("Final training accuracy:", accuracy_score(y_train, best_model.predict(X_train)))
print("Final validation accuracy:", accuracy_score(y_val, best_model.predict(X_val)))
print("Final test accuracy:", accuracy_score(y_test, best_model.predict(X_test)))



# B. CROSS-VALIDATION AND METRICS

print("\n" + "=" * 70)
print("B. CROSS-VALIDATION AND METRICS")
print("=" * 70)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(random_state=42)

# Get predictions for rows when they are in a test fold
predictions = cross_val_predict(model, X, y, cv=folds)

# Get predicted probabilities for the ROC curve and AUC
probabilities = cross_val_predict(
    model, X, y, cv=folds, method="predict_proba"
)[:, 1]

# Confusion matrix values
tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()

accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions, zero_division=0)
recall = recall_score(y, predictions, zero_division=0)
f1 = f1_score(y, predictions, zero_division=0)
specificity = tn / (tn + fp)
auc = roc_auc_score(y, probabilities)

print("Confusion Matrix:")
print(confusion_matrix(y, predictions))

print("\nEvaluation Results")
print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1-score:", round(f1, 3))
print("Specificity:", round(specificity, 3))
print("ROC-AUC:", round(auc, 3))



# C. COMPARING SIX SPLITTING METHODS

print("\n" + "=" * 70)
print("C. COMPARING SIX SPLITTING METHODS")
print("=" * 70)

comparison_model = RandomForestClassifier(
    n_estimators=100, random_state=42
)

comparison_results = []

# 1. Hold-Out: one 80/20 split
X_train_ho, X_test_ho, y_train_ho, y_test_ho = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
holdout_model = RandomForestClassifier(n_estimators=100, random_state=42)
holdout_model.fit(X_train_ho, y_train_ho)
holdout_accuracy = accuracy_score(y_test_ho, holdout_model.predict(X_test_ho))
comparison_results.append({
    "Method": "Hold-Out",
    "Mean Accuracy": holdout_accuracy,
    "Std Accuracy": 0.0,
    "Number of Test Splits": 1
})

# 2. K-Fold: five folds, without class stratification
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(
    comparison_model, X, y, cv=kfold, scoring="accuracy"
)
comparison_results.append({
    "Method": "K-Fold",
    "Mean Accuracy": kfold_scores.mean(),
    "Std Accuracy": kfold_scores.std(),
    "Number of Test Splits": len(kfold_scores)
})

# 3. Stratified K-Fold: five folds with similar class ratios
stratified_folds = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)
stratified_scores = cross_val_score(
    comparison_model, X, y, cv=stratified_folds, scoring="accuracy"
)
comparison_results.append({
    "Method": "Stratified K-Fold",
    "Mean Accuracy": stratified_scores.mean(),
    "Std Accuracy": stratified_scores.std(),
    "Number of Test Splits": len(stratified_scores)
})

# 4. Repeated K-Fold for classification: repeat stratified K-Fold 3 times
repeated_folds = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=3, random_state=42
)
repeated_scores = cross_val_score(
    comparison_model, X, y, cv=repeated_folds, scoring="accuracy"
)
comparison_results.append({
    "Method": "Repeated K-Fold",
    "Mean Accuracy": repeated_scores.mean(),
    "Std Accuracy": repeated_scores.std(),
    "Number of Test Splits": len(repeated_scores)
})

# 5. Time Series Split: earlier rows train, later rows test.
# Titanic has no timestamp, so this is only a row-order demonstration.
time_folds = TimeSeriesSplit(n_splits=5)
time_scores = cross_val_score(
    comparison_model, X.reset_index(drop=True), y.reset_index(drop=True),
    cv=time_folds, scoring="accuracy"
)
comparison_results.append({
    "Method": "Time Series Split (row order demo)",
    "Mean Accuracy": time_scores.mean(),
    "Std Accuracy": time_scores.std(),
    "Number of Test Splits": len(time_scores)
})

# 6. Group K-Fold: all rows with the same ticket stay in one group
if group_labels.nunique() >= 5:
    group_folds = GroupKFold(n_splits=5)
    group_scores = cross_val_score(
        comparison_model, X, y, cv=group_folds,
        groups=group_labels, scoring="accuracy"
    )
    comparison_results.append({
        "Method": "Group K-Fold (Ticket groups)",
        "Mean Accuracy": group_scores.mean(),
        "Std Accuracy": group_scores.std(),
        "Number of Test Splits": len(group_scores)
    })
else:
    print("Group K-Fold skipped: fewer than 5 unique groups are available.")

cv_comparison_table = pd.DataFrame(comparison_results)
display(cv_comparison_table.round(3))

# Visual comparison of mean accuracy
plt.figure(figsize=(10, 5))
plt.bar(
    cv_comparison_table["Method"],
    cv_comparison_table["Mean Accuracy"]
)
plt.ylabel("Mean Accuracy")
plt.title("Comparison of Data-Splitting and Cross-Validation Methods")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()   # close the plot window to continue



# D. ROC CURVE

print("\n" + "=" * 70)
print("D. ROC CURVE")
print("=" * 70)

fpr, tpr, thresholds = roc_curve(y, probabilities)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="Random Forest")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()   # close the plot window to continue

print("ROC-AUC:", round(roc_auc_score(y, probabilities), 3))


# E. CLASSIFICATION THRESHOLD COMPARISON

print("\n" + "=" * 70)
print("E. CLASSIFICATION THRESHOLD COMPARISON")
print("=" * 70)

threshold_results = []

for threshold in np.arange(0.1, 1.0, 0.1):
    threshold_predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, threshold_predictions).ravel()

    threshold_results.append({
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy_score(y, threshold_predictions),
        "Precision": precision_score(y, threshold_predictions, zero_division=0),
        "Recall": recall_score(y, threshold_predictions, zero_division=0),
        "F1-score": f1_score(y, threshold_predictions, zero_division=0),
        "Specificity": tn / (tn + fp)
    })

threshold_table = pd.DataFrame(threshold_results)
display(threshold_table)

# Find the threshold with the highest F1-score
best_row = threshold_table.loc[threshold_table["F1-score"].idxmax()]
print("Best threshold:", best_row["Threshold"])
print("Best F1-score:", round(best_row["F1-score"], 3))



# F. OVERFITTING AND UNDERFITTING

print("\n" + "=" * 70)
print("F. OVERFITTING AND UNDERFITTING")
print("=" * 70)

depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, None]
fit_results = []

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)

    # Accuracy on the full dataset after fitting (training accuracy)
    tree.fit(X, y)
    train_accuracy = accuracy_score(y, tree.predict(X))

    # Average accuracy on held-out folds
    cv_accuracy = cross_val_score(tree, X, y, cv=folds).mean()

    fit_results.append({
        "Max Depth": str(depth),
        "Training Accuracy": train_accuracy,
        "Cross-validation Accuracy": cv_accuracy
    })

fit_table = pd.DataFrame(fit_results)
display(fit_table)

plt.figure(figsize=(8, 5))
plt.plot(fit_table["Max Depth"], fit_table["Training Accuracy"], marker="o", label="Training")
plt.plot(fit_table["Max Depth"], fit_table["Cross-validation Accuracy"], marker="o", label="Cross-validation")
plt.xlabel("Decision Tree Maximum Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting and Underfitting")
plt.legend()
plt.grid(True)
plt.show()   # close the plot window to continue



# G. CLASS IMBALANCE

print("\n" + "=" * 70)
print("G. CLASS IMBALANCE")
print("=" * 70)

# Keep all people who died and randomly select 47 survivors
died = data[data["Survived"] == 0]
survivors = data[data["Survived"] == 1].sample(47, random_state=42)

imbalanced_data = pd.concat([died, survivors]).sample(frac=1, random_state=42)

X_imb = imbalanced_data.drop("Survived", axis=1)
y_imb = imbalanced_data["Survived"]

print("Class counts:")
print(y_imb.value_counts())
print("Survivor percentage:", round(y_imb.mean() * 100, 2), "%")



# G. SMOTE DEMONSTRATION

print("\n" + "=" * 70)
print("G. SMOTE DEMONSTRATION")
print("=" * 70)

# Select one training fold
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_index, test_index = next(folds.split(X_imb, y_imb))

X_train_smote = X_imb.iloc[train_index]
y_train_smote = y_imb.iloc[train_index]

print("Before SMOTE:")
print(y_train_smote.value_counts())

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_smote, y_train_smote)

print("\nAfter SMOTE:")
print(pd.Series(y_train_balanced).value_counts())



# H. TRAIN AND SAVE THE FINAL MODEL

print("\n" + "=" * 70)
print("H. TRAIN AND SAVE THE FINAL MODEL")
print("=" * 70)

# Train the final model using all available labelled data
final_model = RandomForestClassifier(
    max_depth=best_depth,
    n_estimators=100,
    random_state=42
)
final_model.fit(X, y)

# Save the model and preprocessing information together
model_bundle = {
    "model": final_model,
    "gender_encoder": encoder,
    "age_imputer": age_imputer,
    "feature_columns": list(X.columns),
    "target_name": "Survived"
}

model_path = os.path.join(BASE_DIR, "titanic_random_forest_model.joblib")
joblib.dump(model_bundle, model_path)
print("Final model saved as titanic_random_forest_model.joblib")

# Example: load the saved model and make a prediction
loaded_bundle = joblib.load(model_path)
loaded_model = loaded_bundle["model"]
print("Loaded model prediction for the first passenger:", loaded_model.predict(X.iloc[[0]]))