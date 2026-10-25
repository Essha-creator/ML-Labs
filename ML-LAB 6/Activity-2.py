
# Activity 2: Random Forest Classification with GridSearchCV

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# --------------------------------------------------
# TASK 1: Load dataset and display basic information
# --------------------------------------------------

df = pd.read_csv("heart_disease_uci.csv", na_values="?")

print("Dataset Shape:", df.shape)
print("\nFirst 10 Rows:")
print(df.head(10))

print("\nMissing Values:")
print(df.isnull().sum())

# Create the target variable
if "num" in df.columns:
    df["target"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype("Int64")
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    df = df.drop(columns=["num"])
elif "HeartDisease" in df.columns:
    df["target"] = df["HeartDisease"].astype(int)
    df = df.drop(columns=["HeartDisease"])
elif "target" in df.columns:
    df["target"] = df["target"].astype(int)
else:
    raise ValueError("Target column not found!")

# Remove ID and dataset-source columns if present
df = df.drop(columns=["id", "dataset"], errors="ignore")

print("\nTarget Class Distribution:")
print(df["target"].value_counts())

print("\nTarget Class Percentages:")
print(df["target"].value_counts(normalize=True) * 100)

# --------------------------------------------------
# TASK 2: Handle missing values and categorical data
# --------------------------------------------------

X = df.drop(columns=["target"])
y = df["target"]

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

# Fill missing numerical values with the median
# Fill missing categorical values with the most frequent value
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

print("\nNumerical Columns:", list(numerical_cols))
print("Categorical Columns:", list(categorical_cols))

# --------------------------------------------------
# TASK 3: Separate X and y
# --------------------------------------------------

print("\nInput Features (X):")
print(X.head())

print("\nTarget Variable (y):")
print(y.head())

# --------------------------------------------------
# TASK 4: Split data into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# --------------------------------------------------
# TASK 5: Train basic Random Forest
# --------------------------------------------------

basic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

basic_model.fit(X_train, y_train)

basic_predictions = basic_model.predict(X_test)

print("\nBasic Random Forest Accuracy:",
      accuracy_score(y_test, basic_predictions))

# --------------------------------------------------
# TASK 6: Define parameter grid
# --------------------------------------------------

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
    "classifier__max_features": ["sqrt", "log2"]
}

# --------------------------------------------------
# TASK 7: Apply GridSearchCV with 5-fold validation
# --------------------------------------------------

grid_search = GridSearchCV(
    estimator=basic_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# --------------------------------------------------
# TASK 8: Display best parameters and CV accuracy
# --------------------------------------------------

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)

# --------------------------------------------------
# TASK 9: Evaluate the best model on test data
# --------------------------------------------------

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1-Score:", f1_score(y_test, y_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# --------------------------------------------------
# TASK 10: Plot feature importance
# --------------------------------------------------

# Get the trained Random Forest classifier
forest = best_model.named_steps["classifier"]

# Get feature names after preprocessing
feature_names = best_model.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": forest.feature_importances_
})

# Display the 3 most influential features
importance = importance.sort_values(
    by="Importance", ascending=False
)

print("\nTop 3 Most Important Features:")
print(importance.head(3))

# Plot feature importance
top_features = importance.head(15).sort_values(
    by="Importance", ascending=True
)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
