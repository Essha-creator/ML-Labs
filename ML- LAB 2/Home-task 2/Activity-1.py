import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Step 1: Load and explore the dataset
CSV_NAME = "Medical Cost Personal Datasets.csv"
folder = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(folder, CSV_NAME))

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nInfo:")
df.info()
print("\nSummary statistics:")
print(df.describe())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nSmoker counts:")
print(df["smoker"].value_counts())
print("\nRegion counts:")
print(df["region"].value_counts())
print("\nAverage charges by smoker:")
print(df.groupby("smoker")["charges"].mean())

# Step 2: Preprocessing

# 2a. Features and target
features = ["age", "bmi", "children", "smoker", "region"]
X = df[features].copy()
y = df["charges"]

# 2b. Encoding the categorical features
le = LabelEncoder()
X["smoker"] = le.fit_transform(X["smoker"])

X = pd.get_dummies(X, columns=["region"], drop_first=True, dtype=int)

print("\nData after encoding:")
print(X.head())

# 2c. Train/test split (80% train, 20% test) - done BEFORE scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2d. Scale the numerical features (fit on train only, then apply to test)
num_cols = ["age", "bmi", "children"]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Step 3: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nIntercept:", model.intercept_)
print("Coefficients:")
print(pd.Series(model.coef_, index=X_train.columns))

# Step 4: Evaluate using RMSE and R2 score
y_pred = model.predict(X_test)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

print("\nRMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Step 5: Plot Predicted vs Actual costs
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, color="green", alpha=0.6, label="Predictions")

low, high = y_test.min(), y_test.max()
plt.plot([low, high], [low, high], color="red", linestyle="--",
         linewidth=2, label="Perfect prediction")

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Predicted vs Actual Insurance Costs")
plt.legend()
plt.savefig(os.path.join(folder, "predicted_vs_actual.png"), dpi=150)
plt.show()