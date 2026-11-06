import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              StackingRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_FILE = "insurance.csv"

folder = os.path.dirname(os.path.abspath(__file__))

# Step 1: Load insurance.csv and explore it
df = pd.read_csv(os.path.join(folder, CSV_FILE))

print("Shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))
print("\nMissing values:")
print(df.isnull().sum())
print("\nSummary statistics of charges:")
print(df["charges"].describe())

# Step 2: Encode sex and smoker (0/1), one-hot encode region, split 80/20
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
df = pd.get_dummies(df, columns=["region"], dtype=int)

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nFeatures: {list(X.columns)}")
print(f"Train: {X_train.shape[0]}   Test: {X_test.shape[0]}")

# Step 3: Train Ridge (with scaling), RandomForestRegressor, GradientBoostingRegressor
ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gbr = GradientBoostingRegressor(random_state=42)

ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)

# Step 4: Build a StackingRegressor (LinearRegression as final_estimator, cv=5)
stack = StackingRegressor(
    estimators=[("ridge", ridge), ("rf", rf), ("gbr", gbr)],
    final_estimator=LinearRegression(),
    cv=5,
)
stack.fit(X_train, y_train)

# Step 5: Compare all four models on the test set (MAE, RMSE, R2)
models = {
    "Ridge": ridge,
    "Random Forest": rf,
    "Gradient Boosting": gbr,
    "Stacking": stack,
}

rows = []
for name, model in models.items():
    pred = model.predict(X_test)
    rows.append([
        name,
        mean_absolute_error(y_test, pred),
        np.sqrt(mean_squared_error(y_test, pred)),
        r2_score(y_test, pred),
    ])

results = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"])
print("\nComparison table (test set):")
print(results.round(4).to_string(index=False))
results.to_csv(os.path.join(folder, "activity3_results.csv"), index=False)

best_name = results.loc[results["RMSE"].idxmin(), "Model"]
print(f"\nLowest RMSE: {best_name}")
if best_name == "Stacking":
    print("-> Stacking beat every single model on RMSE.")
else:
    print("-> Stacking did NOT beat every single model on RMSE in this run.")

# Step 6: Coefficients of the final estimator
base_names = [name for name, _ in stack.estimators]
coefs = pd.Series(stack.final_estimator_.coef_, index=base_names)

print("\nFinal estimator (LinearRegression) coefficients:")
print(coefs.round(4))
print(f"Intercept: {stack.final_estimator_.intercept_:.4f}")
print(f"\nThe stack trusts '{coefs.idxmax()}' the most (largest coefficient).")

# Step 7: Plot actual versus predicted charges for the stack
stack_pred = stack.predict(X_test)

plt.figure(figsize=(7, 7))
plt.scatter(y_test, stack_pred, alpha=0.6, color="steelblue", label="Test samples")
lims = [min(y_test.min(), stack_pred.min()), max(y_test.max(), stack_pred.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual charges")
plt.ylabel("Predicted charges")
plt.title("Stacking Regressor: actual vs predicted charges")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder, "activity3_actual_vs_predicted.png"), dpi=150)
plt.show()