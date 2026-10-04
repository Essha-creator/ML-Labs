import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Dataset file used for Activity 1 (Telco Customer Churn).
# Keep this CSV in the same folder as this script.
CSV_NAME = "Telco Customer Churn.csv"

# Step 1: Load the dataset and preprocess categorical variables (encoding)
folder = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(folder, CSV_NAME))

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("Missing values in the whole file:", df.isnull().sum().sum())

FEATURES = ["tenure", "MonthlyCharges", "Contract", "InternetService"]
TARGET = "Churn"

print("\nFeatures used:", FEATURES)
print("Target:", TARGET)
print("\nChurn counts (No / Yes):")
print(df[TARGET].value_counts())

# Keep only the columns we need
data = df[FEATURES + [TARGET]].copy()

# Make sure the numeric columns are really numeric
data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce")
data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"], errors="coerce")
data = data.dropna()

# Encode the target: No -> 0, Yes -> 1
data[TARGET] = data[TARGET].map({"No": 0, "Yes": 1})

# Encode the categorical features (Contract, InternetService) with one-hot encoding
# drop_first=True removes one column per feature so the columns are not duplicates
data = pd.get_dummies(data, columns=["Contract", "InternetService"], drop_first=True)
data = data.astype(float)

print("\nColumns after encoding:", list(data.columns))
print("\nSummary statistics:")
print(data.describe())
print("\nCorrelation with Churn:")
print(data.corr()[TARGET])

X = data.drop(columns=[TARGET])
y = data[TARGET]

# Step 2: Train / test split and scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain rows:", len(X_train), " Test rows:", len(X_test))

# Fit the scaler on the training data only, then apply it to both sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]   # probability of churn (class 1)

# Step 4: Evaluate using Accuracy, Precision, Recall, F1-score
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_prob)

scores = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"],
    "Value": [accuracy, precision, recall, f1, auc],
})
pd.options.display.float_format = "{:,.4f}".format
print("\n=== Logistic Regression performance (test set) ===")
print(scores.to_string(index=False))

print("\nClassification report:")
print(metrics.classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))

cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion matrix (rows = actual, columns = predicted):")
print(cm)

print("\nModel coefficients (positive = more likely to churn):")
coef_table = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]})
print(coef_table.sort_values("Coefficient", ascending=False).to_string(index=False))

print("\nNote: churn is an imbalanced target (fewer customers churn than stay),")
print("so look at Precision, Recall and F1 for the 'Churn' class, not only Accuracy.")

# Step 5: Plot Confusion Matrix and ROC Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Confusion matrix
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No churn", "Churn"])
disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Confusion Matrix (test set)")

# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color="red", linewidth=2, label=f"Logistic Regression (AUC = {auc:.3f})")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess (AUC = 0.500)")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(folder, "telco_churn_confusion_matrix_roc.png"), dpi=150)
plt.show()

# Bar chart of the four evaluation metrics
plt.figure(figsize=(8, 5))
plt.bar(["Accuracy", "Precision", "Recall", "F1-score"],
        [accuracy, precision, recall, f1],
        color=["teal", "orange", "purple", "green"])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Telco Churn - Logistic Regression metrics (test set)")
for i, v in enumerate([accuracy, precision, recall, f1]):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(folder, "telco_churn_metrics.png"), dpi=150)
plt.show()