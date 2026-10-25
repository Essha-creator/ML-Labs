# ==================== ACTIVITY 1: PCA (Fashion MNIST) ====================
# Needs: fashion-mnist_train.csv (Kaggle "Fashion MNIST" dataset), placed in the same folder.
# Columns: 'label' + 784 pixel columns (pixel1 ... pixel784)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FILE = 'fashion-mnist_train.csv'   # change if your file has a different name

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# -------------------- Task 1: Load, shape, first 10 rows, class distribution ---------------------
df = pd.read_csv(FILE)
print("Shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))
print("\nClass distribution:")
print(df['label'].value_counts().sort_index())

# -------------------- Task 2: Separate X / y, normalize, StandardScaler ---------------------
X = df.drop('label', axis=1)
y = df['label']

X_norm = X / 255.0                      # pixel values 0-255  ->  0-1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm) # each pixel column -> mean 0, std 1

print("\nBefore StandardScaler -> mean of first 3 pixels:", X_norm.iloc[:, :3].mean().round(3).values,
      "| std:", X_norm.iloc[:, :3].std().round(3).values)
print("After  StandardScaler -> mean of first 3 pixels:", X_scaled[:, :3].mean(axis=0).round(3),
      "| std:", X_scaled[:, :3].std(axis=0).round(3))
print("""
Effect of scaling:
PCA finds directions of maximum VARIANCE. If one feature has a much larger
variance (bigger scale), it dominates the principal components. StandardScaler
gives every pixel mean 0 and std 1, so every pixel gets an equal vote.
(Dividing by 255 only shrinks the range; StandardScaler is what equalises variance.)
""")

# -------------------- Task 3 & 4: Three PCA approaches ---------------------
def run_pca(n_components, name):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    cum_var = pca.explained_variance_ratio_.sum()
    print(f"--- {name} ---")
    print("Original number of features :", X_scaled.shape[1])
    print("Retained components         :", pca.n_components_)
    print(f"Cumulative explained variance: {cum_var:.4f} ({cum_var*100:.2f}%)\n")
    return pca, X_pca, cum_var

pca1, X_pca1, var1 = run_pca(50,   "Method 1: n_components=50")
pca2, X_pca2, var2 = run_pca(0.95, "Method 2: n_components=0.95")
pca3, X_pca3, var3 = run_pca(None, "Method 3: n_components=None")

# -------------------- Task 5: Cumulative explained variance plot ---------------------
cum_var_all = np.cumsum(pca3.explained_variance_ratio_)   # uses ALL components
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_var_all) + 1), cum_var_all, color='blue')
plt.axhline(0.95, color='red', linestyle='--', label='95% variance')
plt.axvline(pca2.n_components_, color='green', linestyle='--',
            label=f'{pca2.n_components_} components for 95%')
plt.axvline(50, color='orange', linestyle=':', label='50 components')
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative explained variance')
plt.title('Cumulative Explained Variance (Fashion MNIST)')
plt.legend()
plt.grid(True)
plt.show()

# -------------------- Task 6: 2D visualization with first two PCs ---------------------
plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_pca3[:, 0], X_pca3[:, 1], c=y, cmap='tab10', s=3, alpha=0.6)
handles, _ = scatter.legend_elements()
plt.legend(handles, class_names, title='Class', loc='best', fontsize=8)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Fashion MNIST - first two principal components')
plt.show()

# -------------------- Task 7: Comparison table ---------------------
comparison = pd.DataFrame({
    'Method': ['1: n_components=50', '2: n_components=0.95', '3: n_components=None'],
    'Original features': [784, 784, 784],
    'Retained components': [pca1.n_components_, pca2.n_components_, pca3.n_components_],
    'Cumulative variance': [round(var1, 4), round(var2, 4), round(var3, 4)],
})
comparison['Dimensionality reduction (%)'] = (
    (1 - comparison['Retained components'] / comparison['Original features']) * 100).round(2)
print(comparison.to_string(index=False))

best_reduction = comparison.loc[comparison['Retained components'].idxmin(), 'Method']
best_info = comparison.loc[comparison['Cumulative variance'].idxmax(), 'Method']
print(f"\nGreatest dimensionality reduction : Method {best_reduction}")
print(f"Preserves the most information    : Method {best_info} (keeps 100% of variance, but no reduction)")