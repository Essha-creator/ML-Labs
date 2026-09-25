import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv("Admission_Predict.csv")
columns = df.columns
df.drop("Serial No.", axis=1, inplace=True)
y = df['Chance of Admit ']
df.drop("Chance of Admit ", axis=1, inplace=True)
print(df.head())

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

# --- Simple linear regression (GRE Score only) ---
simple_lr = LinearRegression()
simple_lr.fit(df[['GRE Score']], y)

plt.scatter(df['GRE Score'], y, color='green', label='Actual Data', alpha=0.5)
plt.plot(df['GRE Score'], simple_lr.predict(df[['GRE Score']]), color='red', linewidth=3,
         label='Regression Line')
plt.xlabel('GRE Score')
plt.ylabel('Admission chance')
plt.title('GRE Score vs Admission chance')
plt.legend()
plt.show()

# --- Multivariate linear regression (all features) ---
lr = LinearRegression()
lr.fit(x_train, y_train)

coefficients = lr.coef_
features = df.columns

plt.figure(figsize=(8, 5))
plt.barh(features, coefficients, color='teal')
plt.xlabel('Coefficient Value (Weight)')
plt.title('Feature Importance in Multivariate Linear Regression')
plt.axvline(x=0, color='black', linewidth=0.8)
plt.show()

# --- Evaluate the multivariate model ---
pred = lr.predict(x_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
r2 = metrics.r2_score(y_test, pred)
print("RMSE:", rmse)
print("R2 Score:", r2)