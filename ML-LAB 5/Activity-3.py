import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

CSV_NAME = "Mall_Customers.csv"

# Step 1: Load the dataset and display it
folder = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(folder, CSV_NAME))

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nSummary statistics:")
print(df.describe())
print("\nMissing values:", df.isnull().sum().sum())

INCOME = "Annual Income (k$)"
SPEND = "Spending Score (1-100)"

X = df[[INCOME, SPEND]].values

# Step 2: Apply K-Means & DBSCAN on the RAW (unscaled) features, plot clusters
kmeans_raw = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_kmeans_raw = kmeans_raw.fit_predict(X)

dbscan_raw = DBSCAN(eps=5, min_samples=5)
labels_dbscan_raw = dbscan_raw.fit_predict(X)
n_clusters_dbscan_raw = len(set(labels_dbscan_raw)) - (1 if -1 in labels_dbscan_raw else 0)
n_noise_raw = list(labels_dbscan_raw).count(-1)
print(f"\nDBSCAN (unscaled, eps=5): {n_clusters_dbscan_raw} clusters, {n_noise_raw} noise points")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_kmeans_raw, cmap="viridis", s=40)
axes[0].scatter(kmeans_raw.cluster_centers_[:, 0], kmeans_raw.cluster_centers_[:, 1],
               c="red", marker="X", s=150, label="Centroids")
axes[0].set_xlabel(INCOME)
axes[0].set_ylabel(SPEND)
axes[0].set_title("K-Means (unscaled, k=5)")
axes[0].legend()

axes[1].scatter(X[:, 0], X[:, 1], c=labels_dbscan_raw, cmap="viridis", s=40)
axes[1].set_xlabel(INCOME)
axes[1].set_ylabel(SPEND)
axes[1].set_title(f"DBSCAN (unscaled, eps=5) - {n_clusters_dbscan_raw} clusters")

plt.tight_layout()
plt.savefig(os.path.join(folder, "clusters_unscaled.png"), dpi=150)
plt.show()

# Step 3: Scale features using StandardScaler and re-run K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_scaled = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_kmeans_scaled = kmeans_scaled.fit_predict(X_scaled)

dbscan_scaled = DBSCAN(eps=0.3, min_samples=5)
labels_dbscan_scaled = dbscan_scaled.fit_predict(X_scaled)
n_clusters_dbscan_scaled = len(set(labels_dbscan_scaled)) - (1 if -1 in labels_dbscan_scaled else 0)
n_noise_scaled = list(labels_dbscan_scaled).count(-1)
print(f"DBSCAN (scaled, eps=0.3): {n_clusters_dbscan_scaled} clusters, {n_noise_scaled} noise points")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_kmeans_scaled, cmap="viridis", s=40)
axes[0].set_xlabel(INCOME)
axes[0].set_ylabel(SPEND)
axes[0].set_title("K-Means (scaled features, k=5)")

axes[1].scatter(X[:, 0], X[:, 1], c=labels_dbscan_scaled, cmap="viridis", s=40)
axes[1].set_xlabel(INCOME)
axes[1].set_ylabel(SPEND)
axes[1].set_title(f"DBSCAN (scaled, eps=0.3) - {n_clusters_dbscan_scaled} clusters")

plt.tight_layout()
plt.savefig(os.path.join(folder, "clusters_scaled.png"), dpi=150)
plt.show()

# Step 4: Compare both results - did scaling change the clusters?
agreement = np.mean(labels_kmeans_raw == labels_kmeans_scaled) * 100
print(f"\nK-Means labels unchanged for {agreement:.1f}% of points after scaling")
print("Note: cluster NUMBERS (0,1,2...) can differ even when the grouping is the same,")
print("so this percentage is only a rough guide - look at the plots to really compare.")
print("Since both features (Income, Spending Score) already use similar 0-100ish scales,")
print("StandardScaler has a much smaller effect here than it would on features with")
print("very different ranges (e.g. income in dollars vs age in years).")

# Step 5: Use the Elbow Method to find the best k
inertias = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(k_range), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("Elbow Method for Optimal k")
plt.xticks(list(k_range))
plt.tight_layout()
plt.savefig(os.path.join(folder, "elbow_method.png"), dpi=150)
plt.show()

# Step 6: Briefly interpret each cluster (using the scaled K-Means result, k=5)
df["Cluster"] = labels_kmeans_scaled
print("\n=== Cluster profile (mean values, scaled K-Means, k=5) ===")
profile = df.groupby("Cluster")[[INCOME, SPEND, "Age"]].mean().round(1)
profile["Count"] = df["Cluster"].value_counts().sort_index()
print(profile)

print("""
Typical interpretation for 5 clusters on Income vs Spending Score:
  - Low income, low spending    -> careful/budget shoppers
  - Low income, high spending   -> impulsive shoppers despite limited income
  - Mid income, mid spending    -> average, unremarkable customers
  - High income, low spending   -> cautious/saving high earners (prime upsell target)
  - High income, high spending  -> ideal customers (target for loyalty programs)
Match each cluster number above to one of these descriptions using its own
Income/Spending averages - the exact cluster numbers can vary between runs.
""")

# Also: add a fake feature "LuckyNumber" (random 0-100) and explain its effect
rng = np.random.RandomState(42)
df["LuckyNumber"] = rng.randint(0, 101, size=len(df))

X3 = df[[INCOME, SPEND, "LuckyNumber"]].values
X3_scaled = StandardScaler().fit_transform(X3)

kmeans_fake = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_fake = kmeans_fake.fit_predict(X3_scaled)

agreement_fake = np.mean(labels_kmeans_scaled == labels_fake) * 100
print(f"K-Means labels unchanged for {agreement_fake:.1f}% of points after adding LuckyNumber")
print("""
LuckyNumber is random noise with no real relationship to spending behaviour.
Adding it as a 3rd clustering feature still lets K-Means treat it as equally
important as Income and Spending Score, since all 3 are scaled to the same
range. This typically scrambles the clean, interpretable Income/Spending
clusters into noisier, harder-to-explain groups - a reminder that K-Means
cannot tell a meaningful feature from an irrelevant one; that judgement has
to come from the person choosing the features.
""")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_kmeans_scaled, cmap="viridis", s=40)
axes[0].set_xlabel(INCOME)
axes[0].set_ylabel(SPEND)
axes[0].set_title("K-Means WITHOUT LuckyNumber (2 features)")

axes[1].scatter(X[:, 0], X[:, 1], c=labels_fake, cmap="viridis", s=40)
axes[1].set_xlabel(INCOME)
axes[1].set_ylabel(SPEND)
axes[1].set_title("K-Means WITH LuckyNumber (3 features)")

plt.tight_layout()
plt.savefig(os.path.join(folder, "luckynumber_effect.png"), dpi=150)
plt.show()