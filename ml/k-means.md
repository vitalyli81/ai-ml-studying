# K-Means Clustering

## TL;DR

K-Means finds K natural groups in your data without any labels — you don't tell it the answers, it discovers the structure on its own. It works by placing K "center points" (centroids) and iteratively assigning every data point to its nearest center, then moving each center to the average of its group, until nothing moves. Use it to segment customers, discover topics, compress data, or explore what natural groups exist in your dataset.

> 💡 **Key Insight:** K-Means is the rare unsupervised algorithm — there's no "right answer" to learn from. You're asking "what groups are hidden in this data?" rather than "predict this label." The algorithm doesn't know if it found meaningful groups; you have to evaluate that yourself.

---

## The Mental Model

Think of **sorting a pile of mixed fruit with your eyes closed, only allowed to feel them**.

You reach in, grab a few pieces, group similar-feeling ones together. You move the "center" of each group (a representative fruit), grab more pieces, reassign anything that now feels closer to a different group. Keep repeating until the groups stabilize.

Mapping:
- A piece of fruit → one data point (row in your dataset)
- How similar two fruits feel → Euclidean distance between their feature vectors
- K groups you're creating → K clusters
- Your hand's position for each group → the centroid (average position)
- "Feels closest to group 3" → assigned to the nearest centroid
- Groups stop changing → algorithm converged

You chose K (number of groups) before starting — that's K-Means' biggest limitation and most important decision.

---

## Why It Exists

### The Problem

Sometimes you don't have labels. You have data — millions of customer transactions, thousands of documents, billions of pixels — and you want to understand what natural structure exists. Supervised learning can't help: there's nothing to predict.

```
Business question: "We have 500,000 customers. Are there different types
                   of customers? What are their behaviors?"

You have: purchase history, browsing data, demographics
You don't have: a list of customer types (the very thing you want to discover!

Supervised ML: useless here (no labels to train on)
K-Means:       discovers the structure in the data itself
```

### The Solution

Minimize within-cluster variance: each point should be closer to its own cluster center than to any other cluster center. K-Means achieves this with an iterative assign-then-update algorithm.

### What Changed

K-Means (developed in the 1950s) enabled data exploration at scale. Today it's used everywhere: Google segments YouTube viewers, Spotify groups listeners for recommendations, retailers identify customer segments for targeted marketing — all without labeled training data.

---

## Core Concepts

### 1. Centroids

**One-line definition:** A centroid is the mean position of all data points in a cluster — the "center of gravity" of that group.

**Analogy:** The centroid of a city is the average lat/long of all its buildings. If you placed every building on a scale, the centroid is the balance point.

```
Cluster members: [(1,2), (3,4), (2,3)]
Centroid: ((1+3+2)/3, (2+4+3)/3) = (2.0, 3.0)  ← average position

When new points are assigned or old ones leave, the centroid moves.
```

**Common misconception:** The centroid must be one of the data points. No — centroids are averages and rarely correspond to any actual data point. If your data is [1, 3], the centroid is 2, which isn't in the dataset.

---

### 2. The Assignment-Update Loop

**One-line definition:** Two alternating steps: assign every point to its nearest centroid, then move each centroid to the average of its assigned points. Repeat until stable.

**Analogy:** A social network where people sit next to whoever they feel most similar to, and group representatives move to the center of their group after everyone sits. Then people check if they're closer to a different representative and move. Repeat until nobody moves.

```
Iteration 1:
  Assign: each point → nearest centroid
  Update: each centroid → mean of assigned points

Iteration 2:
  Assign: points re-evaluate (some may switch clusters!)
  Update: centroids move to new means

Iteration 3... → convergence (no point changes cluster)

Convergence guaranteed: Loss (inertia) strictly decreases each iteration
                        Finite data points → finite configurations
                        → Must converge in finite steps
```

**Common misconception:** K-Means might run forever. It always converges — but it might converge to a poor LOCAL minimum (not the global optimum). Run multiple times with different initializations and pick the best result.

---

### 3. Inertia (Within-Cluster Sum of Squares)

**One-line definition:** The total sum of squared distances from each point to its assigned centroid — the K-Means loss function to minimize.

**Analogy:** If you're organizing seats at a concert and want to minimize how far each person walks to their assigned section — that total walking distance is the inertia.

```
Inertia = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²

Where:
  i = cluster index
  Cᵢ = set of points in cluster i
  μᵢ = centroid of cluster i

Low inertia = points are close to their centroids = tight clusters ← good
High inertia = points are spread far from centroids = loose clusters ← bad

Inertia always decreases with more K (K=n → inertia=0)
→ Can't use raw inertia to pick K; use the elbow method
```

**Common misconception:** Minimize inertia to find the best K. Inertia always gets better with more clusters — at K=n, every point IS its own cluster (inertia=0). Use the elbow method to find meaningful K.

---

### 4. Choosing K — The Elbow Method

**One-line definition:** Plot inertia vs K values and look for the "elbow" where adding more clusters stops providing much improvement.

**Analogy:** Buying a camera lens. Going from 50mm to 100mm makes a huge difference. Going from 490mm to 500mm barely matters. The elbow is where marginal improvement drops dramatically.

```
Inertia
  |
1000|×
 800|  ×
 600|     ×           ← big drops here (K=1,2,3 add a lot)
 400|        ×
 350|           ×     ← elbow at K=4 (diminishing returns after here)
 340|              ×
 335|                 ×
  ──┼──────────────────── K
    1  2  3  4  5  6  7

The "bend" at K=4 suggests 4 natural clusters.
After K=4, adding more clusters barely helps.
```

**Common misconception:** The elbow is always obvious. Often the elbow is ambiguous — the curve is smooth. Complement with domain knowledge and the Silhouette Score.

---

### 5. Silhouette Score

**One-line definition:** A score from -1 to 1 that measures how well each point fits its own cluster vs how well it would fit the next-closest cluster.

**Analogy:** After being assigned to a seat at a dinner table, ask: "Do I belong here? How much better would I fit at another table?" A score of 1 means you clearly belong here. Score of 0 means you're on the border. Score of -1 means you're at the wrong table.

```
For each point x:
  a = average distance to other points in x's cluster (cohesion)
  b = average distance to points in the nearest OTHER cluster (separation)
  
  silhouette(x) = (b - a) / max(a, b)
  
  Score = 1:   point perfectly belongs to its cluster
  Score = 0:   point is on the boundary between two clusters
  Score = -1:  point is probably in the wrong cluster

Overall silhouette = average across all points
Higher = better clustering (aim for > 0.5 for meaningful clusters)
```

**Common misconception:** High silhouette score means you found the "true" number of clusters. It means the data is well-separated at that K — but doesn't guarantee it's meaningful. Customer segments with K=10 might score great but be useless for marketing.

---

## How It Actually Works (Step-by-Step)

```
Dataset: customer annual income vs spending score (2D for visualization)

Points: (15,39), (25,60), (60,5), (70,85), (40,40), (50,80)
Goal: find K=2 clusters

Step 1: Initialize centroids (K-Means++ — smart initialization)
  Centroid A = (15, 39)  ← picked from data
  Centroid B = (70, 85)  ← picked to be far from A

Step 2: Assignment round 1
  (15,39) → dist to A: 0.0, dist to B: 73.5 → Cluster A
  (25,60) → dist to A: 24.0, dist to B: 54.0 → Cluster A
  (60,5)  → dist to A: 55.9, dist to B: 81.4 → Cluster A (!)
  (70,85) → dist to A: 73.5, dist to B: 0.0  → Cluster B
  (40,40) → dist to A: 25.0, dist to B: 52.2 → Cluster A
  (50,80) → dist to A: 51.5, dist to B: 20.6 → Cluster B

Step 3: Update centroids
  Cluster A: [(15,39), (25,60), (60,5), (40,40)] 
    → new centroid: ((15+25+60+40)/4, (39+60+5+40)/4) = (35, 36)
  Cluster B: [(70,85), (50,80)]
    → new centroid: (60, 82.5)

Step 4: Reassign with new centroids
  (60,5) → dist to A(35,36): 39.8, dist to B(60,82.5): 77.5 → stays A
  All others settle → no changes!

Step 5: Converged!
  Cluster A: low-income/low-spending customers
  Cluster B: high-income/high-spending customers
```

---

## Code in Practice

### 1. Hello World — Basic Clustering

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Customer data: [income, spending_score]
X = np.array([
    [15, 39], [16, 81], [17, 6], [18, 77], [19, 40],
    [39, 75], [40, 35], [54, 5], [55, 73], [60, 34],
    [70, 14], [72, 82], [75, 5], [77, 79], [78, 34],
])

# Scale features — K-Means uses Euclidean distance!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# n_init=10: run 10 times with different starting points, keep best
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)

for k in range(3):
    mask = labels == k
    print(f"Cluster {k}: {mask.sum()} customers, "
          f"avg income={X[mask,0].mean():.0f}, "
          f"avg spending={X[mask,1].mean():.0f}")
```

### 2. Practical — Elbow Method + Silhouette Score

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np

X_scaled = StandardScaler().fit_transform(X)

inertias = []
silhouettes = []
K_range = range(2, 10)

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"K={k}: inertia={model.inertia_:.1f}, silhouette={silhouettes[-1]:.3f}")

# Best K is where silhouette is highest (and elbow in inertia)
best_k = K_range[np.argmax(silhouettes)]
print(f"\nBest K by silhouette: {best_k}")
```

### 3. Real-World Pattern — Customer Segmentation Pipeline

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

# Realistic customer data
customers = pd.DataFrame({
    'annual_spend': [500, 2000, 10000, 800, 15000, 3000],
    'purchase_frequency': [2, 5, 20, 3, 30, 8],
    'avg_order_value': [250, 400, 500, 267, 500, 375],
    'days_since_last_purchase': [5, 2, 1, 10, 1, 3],
})

# Pipeline handles scaling automatically
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=3, random_state=42, n_init=10))
])

customers['segment'] = pipe.fit_predict(customers)

# Interpret segments by their mean feature values
segment_profiles = customers.groupby('segment').mean()
print(segment_profiles)

# Name the segments based on their profiles:
# Segment 0: low spend, infrequent → "Occasional Shoppers"
# Segment 1: medium spend, regular → "Regular Customers"
# Segment 2: high spend, frequent  → "VIP / Power Users"
```

---

## Gotchas & Pitfalls

```
❌ Not scaling features before K-Means
   Income (0-200K) completely dominates age (0-100) in distance calculations
✅ Always StandardScaler() before KMeans — income and age need equal weight

❌ Not using n_init > 1
   K-Means with bad starting centroids converges to a poor local minimum
✅ Always set n_init=10 (or use default in newer scikit-learn which uses 'auto')

❌ Setting K arbitrarily (e.g., K=5 "because it sounds right")
   You might miss the actual structure or over-segment
✅ Use elbow method + silhouette score; supplement with domain knowledge

❌ Using K-Means when clusters are non-spherical (elongated, ring-shaped)
   K-Means assumes circular clusters (minimizes Euclidean distance to centroid)
✅ Use DBSCAN for arbitrary-shaped clusters, or Gaussian Mixture Models

❌ Including ID columns or irrelevant features in clustering
   Unique ID numbers dominate distances; irrelevant features add noise
✅ Feature selection is critical — only include features that capture what you care about

❌ Not interpreting the clusters after fitting
   Clustering without interpretation is useless — "3 groups" tells you nothing
✅ Profile each cluster: what is its average for each feature? Give it a name.

❌ Using K-Means on high-dimensional data without reduction
   Distances become meaningless in high dimensions (curse of dimensionality)
✅ Apply PCA or UMAP first to reduce to 5-20 dimensions before clustering
```

---

## When to Use / When NOT to Use

### Use K-Means When:
- Discovering customer segments, user personas, or behavioral groups
- Exploring what natural structure exists in your data before building supervised models
- Creating a "cluster ID" feature to feed into a supervised model
- Image compression (reduce millions of colors to K representative ones)

### Don't Use K-Means When:
- You know your cluster shapes are elongated, ring-shaped, or irregular (use DBSCAN)
- Clusters have very different densities or sizes (use Gaussian Mixture Models)
- You're working with text — raw word vectors cluster poorly without special embeddings
- You have a defined target variable — use supervised learning instead

---

## Related Concepts

| Concept | Connection |
|---------|------------|
| PCA | Often applied before K-Means to reduce dimensions and improve cluster quality |
| DBSCAN | Alternative clustering: finds arbitrary shapes, automatically determines cluster count |
| KNN | Both use distances, but KNN is supervised (needs labels); K-Means is unsupervised |
| Feature Scaling | K-Means uses Euclidean distance — scaling is as critical as it is for SVM/KNN |
| Gaussian Mixture Models | "Soft" version of K-Means — points can belong to multiple clusters with probabilities |

---

## Cheat Sheet

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Full pipeline:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=3,    # K — the most important choice
    n_init=10,       # Run 10 times, keep best (prevents bad initialization)
    random_state=42  # Reproducibility
)
labels = model.fit_predict(X_scaled)

# Key attributes:
model.cluster_centers_   # Centroid coordinates (in scaled space)
model.inertia_           # Total within-cluster variance (lower = tighter)
model.labels_            # Cluster assignment for each training point

# How to pick K:
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    print(f"K={k}: inertia={km.inertia_:.0f}, silhouette={silhouette_score(X_scaled, km.labels_):.3f}")

Remember:
  1. Scale features — Euclidean distances are scale-sensitive
  2. Use n_init=10 to avoid bad local minima
  3. Always interpret clusters — give them names based on their profiles
```

---

## Self-Check Questions

<details>
<summary>Click to reveal answers</summary>

**Q1: What does "convergence" mean in K-Means, and is it guaranteed?**
Convergence means no data point changes its cluster assignment between iterations — the centroids stop moving. It IS guaranteed because each iteration strictly reduces inertia (or keeps it the same), and there's a finite number of possible cluster assignments. However, it converges to a local minimum, not necessarily the global minimum.

**Q2: Why do we run K-Means multiple times (n_init=10) and keep the best result?**
K-Means is sensitive to initial centroid placement. With unlucky starting positions, it can converge to a poor local minimum — clusters that are clearly wrong but mathematically stable. Running 10 times with different random starts and keeping the lowest inertia result significantly improves the chance of finding the global (or near-global) optimum.

**Q3: Explain why K-Means assumes spherical clusters.**
K-Means assigns each point to the nearest centroid using Euclidean distance and updates centroids as means. This implicitly assumes clusters are roughly circular (spherical in higher dimensions) with similar sizes. If a cluster is elongated like a banana, its centroid is in the middle — and points at the banana tips might be closer to a different cluster's centroid. DBSCAN handles arbitrary shapes better.

**Q4: A silhouette score of 0.15 for K=4 clusters. What does this tell you?**
A silhouette score of 0.15 is very weak (close to 0 means points are near cluster boundaries). Either: (1) the natural cluster structure doesn't strongly support K=4, (2) the features aren't capturing meaningful differences, (3) K is wrong (try other values), or (4) the data doesn't have clear cluster structure at all. Aim for >0.5 for meaningful business use.

**Q5: What's the difference between K-Means and KNN?**
K-Means is UNSUPERVISED — it discovers groups with no labels, using K as the number of clusters. KNN is SUPERVISED — it classifies labeled data, using K as the number of neighbors to consult. K-Means creates a model (the centroids) at training time. KNN has no model — it compares to stored training data at prediction time. Both use distances, but for completely different purposes.

</details>

---

## Go Deeper

| Resource | Why It's Worth Your Time |
|----------|--------------------------|
| [StatQuest: K-Means](https://www.youtube.com/watch?v=4b5d3muPQmA) | Best visual walkthrough of the assign-update loop and why initialization matters. 18 minutes. |
| [scikit-learn Clustering docs](https://scikit-learn.org/stable/modules/clustering.html) | Comprehensive comparison of all clustering algorithms with visual examples of when each works. |
| [K-Means++ Paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) | The 2007 paper that introduced smart initialization — explains why n_init matters and why K-Means++ is the standard. |
| *Hands-On ML* Ch. 9 — Unsupervised Learning — Aurélien Géron | Best textbook coverage: K-Means, DBSCAN, GMM, and when to use which. |
| [Kaggle: Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | Classic K-Means exercise with the annual income / spending score dataset. Build your first segmentation in 30 minutes. |
