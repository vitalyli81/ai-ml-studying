# K-Nearest Neighbors (KNN)

## What Is It?

KNN predicts by looking at the **K closest data points** and going with the majority. No training, no formulas — just "show me the nearest examples."

Think of it like moving to a new neighborhood and guessing your lifestyle will match your closest neighbors.

## Real-World Examples

- **Recommendation systems** — people who are similar to you liked these movies
- **Handwritten digit recognition** — this drawn "7" looks closest to other 7s
- **Medical diagnosis** — patients with similar symptoms had this diagnosis
- **Anomaly detection** — this data point has no close neighbors, it's an outlier

## How It Works (Step by Step)

### 1. Store All Training Data

KNN doesn't actually "learn" anything during training. It just **memorizes** all the data. This is called a "lazy learner."

### 2. When a New Point Arrives

1. **Measure the distance** from the new point to every existing point
2. **Find the K nearest neighbors**
3. **Vote** (classification) or **average** (regression)

```
New point: ?

Nearest 3 neighbors:
  Neighbor 1 (distance 0.5): Cat
  Neighbor 2 (distance 0.8): Cat
  Neighbor 3 (distance 1.2): Dog

Vote: 2 Cat vs 1 Dog → predict CAT
```

### 3. Choosing K

| Small K (1-3) | Large K (15-50) |
|---------------|-----------------|
| Sensitive to noise | Smoother boundaries |
| Captures fine details | More robust |
| Can overfit | Can underfit |
| Fast per query | Slower per query |

**Rule of thumb**: Start with K = 5. Try odd numbers (to avoid ties in binary classification). Use cross-validation to find the best K.

### 4. Distance Metrics

How do you measure "closeness"?

**Euclidean Distance** (most common — straight line):
```
d = sqrt((x1-x2)² + (y1-y2)²)
```

**Manhattan Distance** (grid-like — like walking city blocks):
```
d = |x1-x2| + |y1-y2|
```

**Cosine Similarity** (for text/high-dimensional — measures angle, not distance):
```
similarity = dot(A, B) / (|A| * |B|)
```

## Frontend Analogy

KNN is like CSS specificity matching, but for data:

```javascript
function predictKNN(newPoint, allData, k = 5) {
  // 1. Compute distances to all points
  const withDistances = allData.map(point => ({
    ...point,
    distance: euclideanDistance(newPoint, point.features)
  }));

  // 2. Sort by distance, take K nearest
  const neighbors = withDistances
    .sort((a, b) => a.distance - b.distance)
    .slice(0, k);

  // 3. Majority vote
  const votes = {};
  neighbors.forEach(n => {
    votes[n.label] = (votes[n.label] || 0) + 1;
  });

  return Object.entries(votes)
    .sort((a, b) => b[1] - a[1])[0][0];
}
```

That's the entire algorithm. No gradient descent, no loss functions.

## Important: Feature Scaling is Critical

KNN uses distances, so feature scale matters enormously:

```
Without scaling:
  Salary: [30000, 120000]   ← dominates the distance!
  Age:    [20, 65]          ← barely matters

With scaling (both 0-1):
  Salary: [0, 1]            ← equal influence
  Age:    [0, 1]            ← equal influence
```

Always use `StandardScaler` or `MinMaxScaler` before KNN.

## When to Use It

| Good For | Bad For |
|----------|---------|
| Small datasets | Large datasets (slow — compares to every point) |
| Quick baseline without tuning | High-dimensional data (curse of dimensionality) |
| Non-linear decision boundaries | When you need to understand *why* a prediction was made |
| Recommendation systems | When features have very different scales (without scaling) |

## The Curse of Dimensionality

In high dimensions (many features), **all points become roughly equidistant**. KNN breaks down because "nearest" loses meaning.

```
2D:   Nearest neighbor is clearly close
10D:  Nearest and farthest neighbors are almost the same distance
100D: KNN is essentially random
```

Fix: Reduce dimensions first with PCA, or use a different algorithm.

## Python Example

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Scale features (critical for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train (K=5)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Evaluate
accuracy = model.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy:.2%}")

# Try different K values
for k in [1, 3, 5, 7, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = knn.score(X_test_scaled, y_test)
    print(f"  K={k}: {acc:.2%}")
```

## Key Takeaway

KNN is the **simplest ML algorithm** — no training, just look at your neighbors. It's great for small datasets and quick baselines. Always scale your features, choose K wisely, and watch out for high dimensions. Think of it as "you are the average of your closest data points."
