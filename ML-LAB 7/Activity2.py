import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier      # pip install xgboost

CSV_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

folder = os.path.dirname(os.path.abspath(__file__))

# Step 1: Load the CSV and explore it
df = pd.read_csv(os.path.join(folder, CSV_FILE))

print("Shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))
print("\nMissing values (isnull):")
print(df.isnull().sum())
print("\nDistribution of Churn:")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True).round(3))

# Step 2: TotalCharges is text with 11 blank values -> convert to numbers, handle blanks
print("\nBlank TotalCharges values:", int((df["TotalCharges"].astype(str).str.strip() == "").sum()))

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")   # blanks become NaN
print("NaN after conversion:", int(df["TotalCharges"].isnull().sum()))
print(df.loc[df["TotalCharges"].isnull(), ["tenure", "MonthlyCharges", "TotalCharges"]])

# Justification: every blank row has tenure = 0, so these are brand-new customers
# who have not been billed yet. Their true total charge so far is 0, so we fill with 0
# (a median fill would invent a charge that these customers never paid).
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Step 3: Drop customerID, encode Churn (Yes=1, No=0), one-hot encode, split 80/20
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

y = df["Churn"]
X = df.drop("Churn", axis=1)
X = pd.get_dummies(X, drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nFeatures after encoding: {X.shape[1]}")
print(f"Train: {X_train.shape[0]}   Test: {X_test.shape[0]}")


def evaluate(model):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return [
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, zero_division=0),
        recall_score(y_test, pred, zero_division=0),
        f1_score(y_test, pred, zero_division=0),
        roc_auc_score(y_test, proba),
    ]


# Step 4: Train GradientBoosting, AdaBoost and XGBoost with default parameters
gb = GradientBoostingClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)
xgb_default = XGBClassifier(random_state=42, eval_metric="logloss")

gb.fit(X_train, y_train)
ada.fit(X_train, y_train)
xgb_default.fit(X_train, y_train)

default_table = pd.DataFrame(
    [["Gradient Boosting"] + evaluate(gb),
     ["AdaBoost"] + evaluate(ada),
     ["XGBoost (default)"] + evaluate(xgb_default)],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
)
print("\nDefault models on the test set:")
print(default_table.round(4).to_string(index=False))

# Step 5: Tune XGBClassifier with GridSearchCV (five-fold, scoring='f1')
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
}

grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss"),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

print("\nBest parameters:", grid.best_params_)
print(f"Best cross-validated F1 score: {grid.best_score_:.4f}")

xgb_tuned = grid.best_estimator_

# Step 6: Compare default and tuned XGBoost on the test set
compare = pd.DataFrame(
    [["XGBoost (default)"] + evaluate(xgb_default),
     ["XGBoost (tuned)"] + evaluate(xgb_tuned)],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
)
print("\nDefault vs tuned XGBoost (test set):")
print(compare.round(4).to_string(index=False))
compare.to_csv(os.path.join(folder, "activity2_results.csv"), index=False)

# Step 7: Plot the five most important features of the tuned model
importances = pd.Series(xgb_tuned.feature_importances_, index=X.columns)
top5 = importances.sort_values(ascending=False).head(5)
print("\nTop 5 features:")
print(top5.round(4))

plt.figure(figsize=(9, 5))
top5.sort_values().plot(kind="barh", color="steelblue")
plt.xlabel("Feature importance")
plt.title("Top 5 features - tuned XGBoost")
plt.tight_layout()
plt.savefig(os.path.join(folder, "activity2_top5_features.png"), dpi=150)
plt.show()

# Step 8: Why is accuracy alone misleading here?
baseline_acc = 1 - y_test.mean()
print("\nWhy accuracy alone is misleading:")
print(f"  Only {y_test.mean():.1%} of the test customers churn.")
print(f"  A useless model that always predicts 'No churn' already gets "
      f"{baseline_acc:.1%} accuracy, but it finds 0% of the customers who leave.")
print("  Recall, precision, F1 and ROC-AUC show how well the minority class (churners) is found.")