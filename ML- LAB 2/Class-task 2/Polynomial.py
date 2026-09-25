import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
data = pd.read_csv("Position_Salaries.csv")

# Independent variable (Level) and dependent variable (Salary)
X = data[["Level"]].values
y = data["Salary"].values

# Polynomial transformation (degree=6 for better curve fitting)
poly = PolynomialFeatures(degree=8)
X_poly = poly.fit_transform(X)

# Train polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Predictions
y_pred = poly_model.predict(X_poly)

# Plot results
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Polynomial Fit (deg=8)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Regression")
plt.legend()
plt.show()

# Find out its R2 score and RMSE
r2 = metrics.r2_score(y, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
print("R2 Score:", r2)
print("RMSE:", rmse)