# ==================== ACTIVITY 3: Random Forest Regression + GridSearchCV ====================
# Needs: train.csv from Kaggle "House Prices: Advanced Regression Techniques", in the same folder.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------- Task 1: Load train.csv ---------------------
df = pd.read_csv('train.csv')

# -------------------- Task 2: Shape, first 10 rows, missing values, SalePrice summary ---------------------
print("Shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))

missing = df.isnull().sum()
print("\nColumns with missing values:")
print(missing[missing > 0].sort_values(ascending=False))

print("\nSalePrice summary statistics:")
print(df['SalePrice'].describe())

# -------------------- Task 3: Separate X and y, drop Id ---------------------
y = df['SalePrice']
X = df.drop(columns=['SalePrice', 'Id'])    # Id is only an identifier, not a house attribute

# -------------------- Task 4: Missing values + categorical encoding ---------------------
num_cols = X.select_dtypes(include='number').columns   # numeric columns
cat_cols = X.select_dtypes(exclude='number').columns   # text (categorical) columns

# In this dataset NaN in a categorical column usually means "feature absent" (e.g. no alley, no pool)
X[cat_cols] = X[cat_cols].fillna('None')
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

X = pd.get_dummies(X, columns=cat_cols, dtype=int)   # categories -> 0/1 columns
print("\nFeatures after encoding:", X.shape[1], "| Missing values left:", X.isnull().sum().sum())

# -------------------- Task 5: 80/20 split ---------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Task 6: Basic Random Forest Regressor ---------------------
def evaluate(model, name):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    return {'Model': name, 'MAE': mae, 'MSE': mse, 'RMSE': np.sqrt(mse), 'R2': r2_score(y_test, pred)}

rf_default = RandomForestRegressor(random_state=42)
rf_default.fit(X_train, y_train)
default_results = evaluate(rf_default, 'Default RF')
print("\nDefault model performance:")
print(default_results)

# -------------------- Task 7: Parameter grid ---------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': [0.7, 1.0],
}

# -------------------- Task 8: GridSearchCV (5-fold, neg MSE) ---------------------
grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
)
grid.fit(X_train, y_train)

# -------------------- Task 9: Best params, CV score -> MSE -> RMSE ---------------------
best_mse = -grid.best_score_          # sklearn returns NEGATIVE MSE (higher = better), so flip the sign
best_rmse = np.sqrt(best_mse)
print("\nBest hyperparameters:", grid.best_params_)
print(f"Best CV score (neg MSE): {grid.best_score_:.2f}")
print(f"Best CV MSE : {best_mse:.2f}")
print(f"Best CV RMSE: {best_rmse:.2f}")

# -------------------- Task 10: Evaluate best model on test set ---------------------
best_model = grid.best_estimator_
tuned_results = evaluate(best_model, 'Tuned RF (GridSearchCV)')
print("\nTuned model test performance:")
print(tuned_results)

# -------------------- Task 11: Compare default vs optimized ---------------------
comparison = pd.DataFrame([default_results, tuned_results]).set_index('Model')
print("\nComparison table:")
print(comparison.round(4))

# -------------------- Task 12: Actual vs predicted + top 5 feature importances ---------------------
y_pred = best_model.predict(X_test)

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue')
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, 'r--', label='Perfect prediction')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid(True)
plt.show()

importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
top5 = importances.head(5).sort_values()
plt.figure(figsize=(8, 4))
plt.barh(top5.index, top5.values, color='seagreen')
plt.xlabel('Importance')
plt.title('Top 5 Feature Importances')
plt.tight_layout()
plt.show()

print("\nFive most influential features:")
print(importances.head(5))