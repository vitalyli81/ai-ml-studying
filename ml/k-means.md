# K-Means Clustering

## What Is It?

K-Means is an **unsupervised** algorithm — it finds groups (clusters) in your data **without labels**. You tell it how many groups (K), and it figures out which data points belong together.

Think of it like sorting a pile of mixed candy into groups by color — nobody told you what the colors are, you just group similar ones together.

## Real-World Examples

- **Customer segmentation** — group users into "budget shoppers," "premium buyers," "occasional visitors"
- **Image compression** — reduce millions of colors to K representative colors
- **Document grouping** — cluster articles by topic
- **Anomaly detection** — points far from any cluster center might be outliers

## How It Works (Step by Step)

### 1. Pick K (number of clusters)

You decide: "I want 3 groups."

### 2. Initialize K Centroids

Place K points randomly in the data space. These are the initial cluster centers.

```
Step 0: Random centroids placed
  o  o        x (centroid 1)
    o   o
  o       o
        x (centroid 2)
    o o
          x (centroid 3)
```

### 3. Assign Each Point to the Nearest Centroid

Every data point joins the cluster of its closest centroid.

```
Step 1: Points assigned to nearest centroid
  [1] [1]        x₁
    [1]  [2]
  [1]      [2]
        x₂
    [3] [3]
          x₃
```

### 4. Move Centroids to the Center of Their Cluster

Each centroid moves to the **average position** of all its assigned points.

```
Step 2: Centroids move to cluster centers
  [1] [1]
    x₁  [2]
  [1]      x₂
        [2]
    [3] x₃
          [3]
```

### 5. Repeat Steps 3-4 Until Centroids Stop Moving

Usually converges in 10-20 iterations.

## The Full Animation

```
Iteration 1:  Random centroids → Assign → Move
Iteration 2:  New centroids → Reassign → Move
Iteration 3:  New centroids → Reassign → Move (barely changed)
Iteration 4:  Converged! Centroids stopped moving.
```

## Supervised vs Unsupervised

| Supervised (KNN, SVM, etc.) | Unsupervised (K-Means) |
|-----------------------------|------------------------|
| Has labeled data (X → y) | No labels, just data (X) |
| "Predict the answer" | "Find the structure" |
| You know the categories | You discover the categories |
| Training + testing | No test set (no "right answer") |

## How to Choose K

The biggest question: **how many clusters?**

### The Elbow Method

Run K-Means for K = 1, 2, 3, ... 10 and plot the **inertia** (total distance from points to their centroid):

```
Inertia
  |
  |\
  |  \
  |    \___         ← "elbow" here → K = 3
  |        \____
  |             \_____
  |___________________ K
  1  2  3  4  5  6  7
```

Pick K where the curve **bends** (diminishing returns after that).

### Silhouette Score

Measures how similar a point is to its own cluster vs other clusters. Range: -1 to 1.

- **1** → perfectly clustered
- **0** → on the boundary between clusters
- **-1** → probably in the wrong cluster

## Limitations and Gotchas

| Problem | Why It Happens |
|---------|---------------|
| **Sensitive to initialization** | Random starting centroids can give different results. Fix: use `init='k-means++'` (default in scikit-learn) |
| **Must specify K** | You have to guess the number of clusters. Use elbow method. |
| **Assumes spherical clusters** | K-Means draws circular boundaries. Won't find elongated or irregular shapes. |
| **Sensitive to scale** | Scale your features first! Same reason as KNN. |
| **Sensitive to outliers** | One extreme point can pull a centroid far away. |

## When to Use It

| Good For | Bad For |
|----------|---------|
| Discovering natural groups | When clusters aren't blob-shaped |
| Customer/user segmentation | When you don't know if clusters exist |
| Data exploration (what's in here?) | Very different cluster sizes |
| Feature engineering (cluster ID as feature) | High-dimensional data (use with PCA) |

## Python Example

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample customer data: [annual_income, spending_score]
X = np.array([
    [15, 39], [16, 81], [17, 6], [18, 77], [19, 40],
    [39, 75], [40, 35], [54, 5], [55, 73], [60, 34],
    [70, 14], [72, 82], [75, 5], [77, 79], [78, 34],
])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal K with elbow method
inertias = []
for k in range(1, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"K={k}: inertia={km.inertia_:.1f}")

# Train with chosen K
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)

# See results
for i in range(3):
    cluster_points = X[labels == i]
    print(f"\nCluster {i}: {len(cluster_points)} customers")
    print(f"  Avg income: {cluster_points[:, 0].mean():.0f}")
    print(f"  Avg spending: {cluster_points[:, 1].mean():.0f}")
```

## Key Takeaway

K-Means is the **simplest way to discover groups** in your data. It's fast, intuitive, and widely used for customer segmentation and data exploration. Remember: scale your data, use `k-means++` initialization (default), and use the elbow method to pick K. It finds blob-shaped clusters — if your clusters are weird shapes, look into DBSCAN.
