import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# Dataset file used for Activity 2 (Car Price Prediction).
CSV_NAME = "Car Price Prediction.csv"

# Step 1: Load the dataset and choose features
folder = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(folder, CSV_NAME))

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("Missing values in the whole file:", df.isnull().sum().sum())

FEATURES = ["enginesize", "horsepower", "citympg"]
TARGET = "price"

print("\nFeatures used:", FEATURES)
print("Target:", TARGET)
print("\nSummary statistics:")
print(df[FEATURES + [TARGET]].describe())
print("\nCorrelation with price:")
print(df[FEATURES + [TARGET]].corr()[TARGET])

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTrain rows:", len(X_train), " Test rows:", len(X_test))

results = []

def evaluate(name, model):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    results.append({
        "Model": name,
        "Train R2": metrics.r2_score(y_train, pred_train),
        "Test R2": metrics.r2_score(y_test, pred_test),
        "Test RMSE": np.sqrt(metrics.mean_squared_error(y_test, pred_test)),
        "Test MAE": metrics.mean_absolute_error(y_test, pred_test),
    })

# Step 2: Linear Regression (record performance)
linear_model = make_pipeline(StandardScaler(), LinearRegression())
linear_model.fit(X_train, y_train)
evaluate("Linear", linear_model)

# Step 3: Polynomial Regression (degree = 2, 3, 4) and compare
for degree in [2, 3, 4]:
    poly_model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    )
    poly_model.fit(X_train, y_train)
    n_terms = poly_model.named_steps["polynomialfeatures"].n_output_features_
    print(f"Degree {degree}: {n_terms} polynomial terms created from {len(FEATURES)} features")
    evaluate(f"Polynomial (degree={degree})", poly_model)

results_df = pd.DataFrame(results)
pd.options.display.float_format = "{:,.4f}".format
pd.options.display.width = 120
print("\n=== Comparison of all models (3 features) ===")
print(results_df.to_string(index=False))

best = results_df.loc[results_df["Test R2"].idxmax(), "Model"]
print("\nBest model on the test set (highest Test R2):", best)
print("Note: with only 205 cars, high-degree polynomials fit the training data")
print("very closely but can perform badly on new data (overfitting).")

# Step 4: Plot fitted curves for different polynomial degrees
colors = {1: "red", 2: "green", 3: "blue", 4: "purple"}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, feature in zip(axes, ["horsepower", "enginesize"]):
    x_tr = X_train[[feature]]
    x_te = X_test[[feature]]
    x_line = pd.DataFrame(
        np.linspace(X[feature].min(), X[feature].max(), 300), columns=[feature]
    )

    ax.scatter(X_train[feature], y_train, color="lightgray", s=20, label="Train data")
    ax.scatter(X_test[feature], y_test, color="black", s=20, label="Test data")

    for degree in [1, 2, 3, 4]:
        curve = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression(),
        )
        curve.fit(x_tr, y_train)
        r2 = metrics.r2_score(y_test, curve.predict(x_te))
        name = "Linear (degree=1)" if degree == 1 else f"Polynomial (degree={degree})"
        ax.plot(x_line[feature], curve.predict(x_line), color=colors[degree],
                linewidth=2, label=f"{name}, test R2 = {r2:.3f}")

    ax.set_xlabel(feature)
    ax.set_ylabel("price")
    ax.set_title(f"Fitted curves: price vs {feature}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(folder, "car_price_fitted_curves.png"), dpi=150)
plt.show()

x_pos = np.arange(len(results_df))
plt.figure(figsize=(9, 5))
plt.bar(x_pos - 0.2, results_df["Train R2"], width=0.4, label="Train R2", color="teal")
plt.bar(x_pos + 0.2, results_df["Test R2"], width=0.4, label="Test R2", color="orange")
plt.xticks(x_pos, results_df["Model"], rotation=15)
plt.ylabel("R2 score")
plt.title("Train vs Test R2 (3 features)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder, "car_price_model_comparison.png"), dpi=150)
plt.show()