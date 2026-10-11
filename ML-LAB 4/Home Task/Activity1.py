import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE

CSV_NAME = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"

folder = os.path.dirname(os.path.abspath(__file__))

# Step 1: Load the dataset and split into X and y
print("Downloading dataset (about 144 MB, this can take a minute)...")
df = pd.read_csv(CSV_NAME)
print("Download complete.")

print("Shape:", df.shape)
print("\nClass counts:")
print(df["Class"].value_counts())
fraud_pct = df["Class"].mean() * 100
print(f"Fraud transactions: {fraud_pct:.3f}% of all samples")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train class counts:", y_train.value_counts().to_dict())
print("Test class counts :", y_test.value_counts().to_dict())

# Step 3: Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Apply SMOTE to handle class imbalance (fit on TRAINING data only)
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print("Resampled train class counts:", pd.Series(y_train_res).value_counts().to_dict())

# Step 4a: Train a baseline KNN classifier with k=5
print("\nTraining baseline KNN (k=5) on the resampled data...")
t0 = time.time()
knn_default = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_default.fit(X_train_res, y_train_res)
y_pred_default = knn_default.predict(X_test_scaled)
print(f"Done in {time.time() - t0:.1f}s")

acc_default = accuracy_score(y_test, y_pred_default)
f1_default = f1_score(y_test, y_pred_default)
print(f"k=5  ->  Accuracy: {acc_default:.4f}   F1-score: {f1_default:.4f}")
print(classification_report(y_test, y_pred_default, target_names=["Legit", "Fraud"]))

# Step 4b: Tune k using GridSearchCV

print("\nTuning k with GridSearchCV (on a subsample for speed)...")
SAMPLE_SIZE = 20000
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(X_train_res), size=min(SAMPLE_SIZE, len(X_train_res)), replace=False)
X_sample = X_train_res[sample_idx]
y_sample = np.asarray(y_train_res)[sample_idx]

param_grid = {"n_neighbors": [3, 5, 7, 9]}
grid = GridSearchCV(
    KNeighborsClassifier(n_jobs=-1), param_grid,
    scoring="f1", cv=3, n_jobs=-1
)
t0 = time.time()
grid.fit(X_sample, y_sample)
print(f"Grid search done in {time.time() - t0:.1f}s")
print("CV results (mean F1 per k):")
for k, score in zip(param_grid["n_neighbors"], grid.cv_results_["mean_test_score"]):
    print(f"  k={k}: {score:.4f}")
best_k = grid.best_params_["n_neighbors"]
print("Best k:", best_k)

print(f"\nTraining final KNN (k={best_k}) on the full resampled data...")
t0 = time.time()
knn_best = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn_best.fit(X_train_res, y_train_res)
y_pred_best = knn_best.predict(X_test_scaled)
print(f"Done in {time.time() - t0:.1f}s")

acc_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
print(f"k={best_k}  ->  Accuracy: {acc_best:.4f}   F1-score: {f1_best:.4f}")
print(classification_report(y_test, y_pred_best, target_names=["Legit", "Fraud"]))

# Step 5: Compare model performance (Accuracy and F1-Score)
comparison = pd.DataFrame({
    "Model": [f"KNN (k=5, default)", f"KNN (k={best_k}, tuned)"],
    "Accuracy": [acc_default, acc_best],
    "F1-Score": [f1_default, f1_best],
})
print("\n=== Model comparison ===")
print(comparison.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

x_pos = np.arange(len(comparison))
axes[0].bar(x_pos - 0.2, comparison["Accuracy"], width=0.4, label="Accuracy", color="teal")
axes[0].bar(x_pos + 0.2, comparison["F1-Score"], width=0.4, label="F1-Score", color="orange")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(comparison["Model"], rotation=10)
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Accuracy vs F1-Score")
axes[0].legend()

cm_default = confusion_matrix(y_test, y_pred_default)
ConfusionMatrixDisplay(cm_default, display_labels=["Legit", "Fraud"]).plot(
    ax=axes[1], cmap="Blues", colorbar=False)
axes[1].set_title("Confusion Matrix (k=5)")

cm_best = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(cm_best, display_labels=["Legit", "Fraud"]).plot(
    ax=axes[2], cmap="Blues", colorbar=False)
axes[2].set_title(f"Confusion Matrix (k={best_k}, tuned)")

plt.tight_layout()
plt.savefig(os.path.join(folder, "fraud_knn_results.png"), dpi=150)
plt.show()