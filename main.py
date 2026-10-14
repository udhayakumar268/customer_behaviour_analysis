# ============================================================
# CUSTOMER BEHAVIOUR ANALYSIS USING UNSUPERVISED LEARNING
# ============================================================\

# 1. IMPORT LIBRARIES

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 2. PROJECT PATHS AND SETTINGS

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Mall_Customers.csv"
PLOTS_DIR = BASE_DIR / "plots"
OUTPUTS_DIR = BASE_DIR / "outputs"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv("Mall_Customers.csv")

# Optional initial data inspection

# print(data.shape)
# print(data.head())
# print(data.info())
# print(data.isnull().sum())
# print(data.dtypes)
# print(data.describe())
# print(data["Genre"].value_counts())


# 3. DATA PREPARATION AND FEATURE SCALING

X = data[["Annual Income (k$)","Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X.shape)
print(X_scaled.shape)
print(X_scaled[:5])

# Optional initial data visualization inspection

# sns.scatterplot(data=data, x="Annual Income (k$)", y="Spending Score (1-100)")
# plt.title("annual income vs spending score")
# plt.show()

# 4. COMPARE SILHOUETTE SCORES FOR K = 2 TO 10

silhouette_scores = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    silhouette_scores.append(score)

    print(f"K = {k}, Silhouette Score = {score:.3f}")

# 5. IDENTIFY THE BEST K BY SILHOUETTE SCORE

best_k = range(2, 11)[
    np.argmax(silhouette_scores)
]

print("\nBest K by silhouette score:", best_k)

# 6. VISUALISE SILHOUETTE SCORES

plt.figure(figsize=(10, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Comparison")
plt.xticks(range(2, 11))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "silhouette_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 7. K-MEANS: ELBOW METHOD

inertia_value = []
for k in range(1,11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia_value.append(model.inertia_)
        
plt.figure(figsize=(10,5))
plt.plot(range(1,11),inertia_value, marker = "o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "elbow_method.png",
    dpi=300,
    bbox_inches="tight")
plt.show()

# 8. TRAIN K-MEANS MODEL WITH 5 CLUSTERS

Kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
data["Cluster"] = Kmeans.fit_predict(X_scaled)

print(data[["Cluster","Annual Income (k$)","Spending Score (1-100)"]].head(10))
print("\nCustomers in each cluster:")
print(data["Cluster"].value_counts().sort_index())

# 9. EVALUATE THE FINAL K-MEANS MODEL

score = silhouette_score(X_scaled, data["Cluster"])

print("\nFinal K-Means silhouette score:", round(score, 3))

# 10. VISUALISE CUSTOMER SEGMENTS

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=data,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="rocket",
    s=100)
plt.title("Customer Segments Using K-means")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "customer_segments.png",
    dpi=300,
    bbox_inches="tight")

plt.show()

# 11. CUSTOMER CLUSTER SUMMARY

Cluster_summary = data.groupby("Cluster").agg(customer_count = ("CustomerID","count"),avg_age = ("Age", "mean"), avg_income = ("Annual Income (k$)", "mean"), avg_spending = ("Spending Score (1-100)", "mean")).round(2)

print(Cluster_summary)

Cluster_summary.to_csv(
    OUTPUTS_DIR / "cluster_summary.csv")

# 12. ANOMALY DETECTION: GAUSSIAN PROBABILITY

X_anomaly = data[["Annual Income (k$)","Spending Score (1-100)"]]

means = X_anomaly.mean()
stds = X_anomaly.std(ddof=0)


print("Feature means:", means)
print(means)

print("\nFeature standard deviations:")
print(stds)

# Calculate probability density for each feature

feature_densities = norm.pdf (X_anomaly, loc = means , scale = stds)

# Multiply feature densities to obtain anomaly scores

data["anomaly_score"] = feature_densities.prod(axis = 1)

print(
    data[[
        "Annual Income (k$)",
        "Spending Score (1-100)",
        "anomaly_score"
    ]].head(10)
)

# 13. FLAG POTENTIAL ANOMALIES

threshold = data["anomaly_score"].quantile(0.05)
data["is_anomaly"] = data["anomaly_score"] < threshold
print("Anomaly threshold:", threshold)
print("\nNumber of potential anomalies:")
print(data["is_anomaly"].sum())

print("\nPotential anomalies:")
print(data.loc[data["is_anomaly"],["Annual Income (k$)", "Spending Score (1-100)", "anomaly_score"]])

# 14. VISUALISE GAUSSIAN ANOMALY SCORES

plt.figure(figsize=(10, 5))

sns.histplot(
    data=data,
    x="anomaly_score",
    bins=30,
    kde=True
)

plt.axvline(
    threshold,
    linestyle="--",
    color="red",
    label="5th percentile threshold"
)

plt.xlabel("Anomaly Score")
plt.ylabel("Number of Customers")
plt.title("Distribution of Gaussian Anomaly Scores")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "gaussian_anomaly_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 15. VISUALISE POTENTIAL ANOMALIES

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=data[data["is_anomaly"] == False],
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="steelblue",
    s=80,
    label="Typical customers"
)

sns.scatterplot(
    data=data[data["is_anomaly"] == True],
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="red",
    marker="X",
    s=180,
    label="Potential anomalies"
)

plt.title("Customer Anomaly Detection")
plt.legend()
plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "customer_anomalies.png",
    dpi=300,
    bbox_inches="tight")

plt.show()

# 16. K-MEANS AFTER EXCLUDING FLAGGED CUSTOMERS

normal_data = data[data["is_anomaly"] == False].copy()
X_normal = normal_data[ ["Annual Income (k$)", "Spending Score (1-100)"]]


scaler_normal = StandardScaler()
X_normal_scaled = scaler_normal.fit_transform(X_normal)

kmeans_normal = KMeans(n_clusters=5, random_state=42, n_init=10)
normal_data["cluster_without_flagged"] = (
    kmeans_normal.fit_predict(X_normal_scaled)
)

# 17. COMPARE CLUSTER SIZES AND CENTROIDS

print("\nCluster sizes without flagged customers:")
print(normal_data["cluster_without_flagged"].value_counts().sort_index())


print("\nCentroids without flagged customers:")
print(
    scaler_normal.inverse_transform(kmeans_normal.cluster_centers_)
)

# 18. EXPORT RESULTS TO CSV

data.to_csv(
    OUTPUTS_DIR / "customer_analysis_results.csv",
    index=False
)

normal_data.to_csv(
    OUTPUTS_DIR / "customers_without_flagged_anomalies.csv",
    index=False
)

print("\nProject completed successfully!")

print("\nPlots saved in:")
print(PLOTS_DIR)

print("\nCSV outputs saved in:")
print(OUTPUTS_DIR)
