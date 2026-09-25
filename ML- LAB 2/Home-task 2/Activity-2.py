import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score)

CSV_NAME = "Telco Customer Churn.csv"

# Step 1: Load dataset
folder = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(folder, CSV_NAME))

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)

features = ["tenure", "MonthlyCharges", "Contract", "InternetService"]
data = df[features + ["Churn"]].copy()

print("\nMissing values in the columns we use:")
print(data.isnull().sum())
print("\nChurn counts:")
print(data["Churn"].value_counts())
print("\nContract values:", data["Contract"].unique())
print("InternetService values:", data["InternetService"].unique())

# Step 2: Preprocess categorical variables (Encoding)
le = LabelEncoder()
data["Churn"] = le.fit_transform(data["Churn"])
print("\nChurn encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

data = pd.get_dummies(data, columns=["Contract", "InternetService"],
                      drop_first=True, dtype=int)

print("\nData after encoding:")
print(data.head())

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = ["tenure", "MonthlyCharges"]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Step 3: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nCoefficients:")
print(pd.Series(model.coef_[0], index=X_train.columns))
print("Intercept:", model.intercept_[0])

# Step 4: Evaluate using Accuracy, Precision, Recall, F1-score
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1-score :", round(f1_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# Step 5: Plot Confusion Matrix & ROC Curve
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", round(auc, 4))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(
    ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Confusion Matrix")

axes[1].plot(fpr, tpr, color="darkorange", linewidth=2,
             label=f"Logistic Regression (AUC = {auc:.2f})")
axes[1].plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random guess")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(folder, "churn_confusion_matrix_roc.png"), dpi=150)
plt.show()