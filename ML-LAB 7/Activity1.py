import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)

CSV_FILE = "water_potability.csv"

folder = os.path.dirname(os.path.abspath(__file__))

# Step 1: Load the CSV and explore it
df = pd.read_csv(os.path.join(folder, CSV_FILE))

print("Shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))
print("\nMissing values:")
print(df.isnull().sum())
print("\nClass distribution of Potability:")
print(df["Potability"].value_counts())
print(df["Potability"].value_counts(normalize=True).round(3))

# Step 2: Split the data 80/20 with stratify=y and random_state=42
X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]}   Test: {X_test.shape[0]}")

# Step 3: Fill missing values (ph, Sulfate, Trihalomethanes) with the TRAINING set median
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)       # use the training medians, not the test medians
print("\nMissing values left in train:", int(X_train.isnull().sum().sum()))
print("Missing values left in test: ", int(X_test.isnull().sum().sum()))

# Step 4: Train the single models (scale the data for Logistic Regression and SVC)
log_reg = Pipeline([("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000, random_state=42))])
svc = Pipeline([("scaler", StandardScaler()),
                ("model", SVC(probability=True, random_state=42))])
rf = RandomForestClassifier(n_estimators=100, random_state=42)

log_reg.fit(X_train, y_train)
svc.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Step 5: Combine them in a VotingClassifier (hard voting and soft voting)
estimators = [("lr", log_reg), ("svc", svc), ("rf", rf)]

voting_hard = VotingClassifier(estimators=estimators, voting="hard")
voting_soft = VotingClassifier(estimators=estimators, voting="soft")
voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)

# Step 6: Train the BaggingClassifier models and one single decision tree
bag_10 = BaggingClassifier(DecisionTreeClassifier(random_state=42),
                           n_estimators=10, max_samples=0.5,
                           bootstrap=True, random_state=42)
bag_100 = BaggingClassifier(DecisionTreeClassifier(random_state=42),
                            n_estimators=100, max_samples=0.5,
                            bootstrap=True, random_state=42)
pasting_100 = BaggingClassifier(DecisionTreeClassifier(random_state=42),
                                n_estimators=100, max_samples=0.5,
                                bootstrap=False, random_state=42)
single_tree = DecisionTreeClassifier(random_state=42)

bag_10.fit(X_train, y_train)
bag_100.fit(X_train, y_train)
pasting_100.fit(X_train, y_train)
single_tree.fit(X_train, y_train)

# Step 7: Compare all models in a table (accuracy, precision, recall, F1-score)
models = {
    "Logistic Regression": log_reg,
    "SVC": svc,
    "Random Forest": rf,
    "Decision Tree (single)": single_tree,
    "Voting (hard)": voting_hard,
    "Voting (soft)": voting_soft,
    "Bagging (10, bootstrap)": bag_10,
    "Bagging (100, bootstrap)": bag_100,
    "Pasting (100, no bootstrap)": pasting_100,
}

rows = []
for name, model in models.items():
    pred = model.predict(X_test)
    rows.append([
        name,
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, zero_division=0),
        recall_score(y_test, pred, zero_division=0),
        f1_score(y_test, pred, zero_division=0),
    ])

results = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"])
print("\nComparison table:")
print(results.round(4).to_string(index=False))
results.to_csv(os.path.join(folder, "activity1_results.csv"), index=False)

# Step 8: Confusion matrix of the best model (highest accuracy)
best_name = results.loc[results["Accuracy"].idxmax(), "Model"]
best_model = models[best_name]
cm = confusion_matrix(y_test, best_model.predict(X_test))
print(f"\nBest model by accuracy: {best_name}")
print(cm)

plt.figure(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=["Not potable", "Potable"]).plot(
    cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(os.path.join(folder, "activity1_confusion_matrix.png"), dpi=150)
plt.show()

# Step 9: Did voting or bagging beat the best single model?
single_names = ["Logistic Regression", "SVC", "Random Forest", "Decision Tree (single)"]
ensemble_names = ["Voting (hard)", "Voting (soft)", "Bagging (10, bootstrap)",
                  "Bagging (100, bootstrap)", "Pasting (100, no bootstrap)"]

best_single = results[results["Model"].isin(single_names)].sort_values("Accuracy").iloc[-1]
best_ens = results[results["Model"].isin(ensemble_names)].sort_values("Accuracy").iloc[-1]

print(f"\nBest single model:   {best_single['Model']}  "
      f"(accuracy {best_single['Accuracy']:.4f}, F1 {best_single['F1-score']:.4f})")
print(f"Best ensemble model: {best_ens['Model']}  "
      f"(accuracy {best_ens['Accuracy']:.4f}, F1 {best_ens['F1-score']:.4f})")
if best_ens["Accuracy"] > best_single["Accuracy"]:
    print("-> A voting/bagging ensemble beat the best single model on accuracy.")
else:
    print("-> No voting/bagging ensemble beat the best single model on accuracy.")
# Now write your two-sentence explanation in your report using these numbers.