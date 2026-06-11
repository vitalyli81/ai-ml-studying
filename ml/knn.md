# K-Nearest Neighbors (KNN)

## TL;DR

KNN predicts by finding the K most similar training examples to a new data point and taking their majority vote (classification) or average (regression). There's no training step — it just memorizes all data and computes distances at prediction time. It's the most intuitive ML algorithm: "you are who your closest neighbors are." Works well on small datasets but breaks on large or high-dimensional data. Always scale features first.

> 💡 **Key Insight:** KNN has no model — it IS the data. Every prediction requires computing distances to every training example. This makes it powerful on small datasets and unusable on large ones.

---

## The Mental Model

Think of **moving to a new city and predicting which coffee shop you'll like**.

You don't research every café in the city. You find the 5 people most similar to you (same taste in music, food, lifestyle) and ask which cafés they like. Their consensus is your prediction.

Mapping:
- "People most similar to you" → K nearest neighbors (smallest distance)
- Your profile (music taste, food preference) → feature vector
- How similar two people are → distance metric (Euclidean, Manhattan)
- Their favorite cafés → training labels
- Their consensus recommendation → the majority vote prediction
- K = 5 → polling 5 people
- K = 1 → just asking your single closest match

Change K, change how many people you trust. Change the distance metric, change your definition of "similar."

> 💻 **Frontend bridge:** KNN is `items.map(distanceTo(query)).sort().slice(0, k)` over the *entire* dataset — on every single query. It's filtering a million-row unindexed table inside the render loop: fine for a demo, fatal at scale. The grown-up fix is the same as in web dev — add an index. That's exactly what FAISS / vector databases are: a search index for nearest-neighbor queries (and the retrieval step behind RAG).

---

## Build the Intuition From Zero

KNN itself is the easiest algorithm in this folder. Two things still puzzle people: **what the distance formula's `√` and squares are actually doing, and why KNN mysteriously falls apart in "high dimensions."** Let's nail both.

### Idea 1: "Distance" is just the ruler from grade-school geometry

The scary formula `d = √((x₁−x₂)² + (y₁−y₂)²)` is the **Pythagorean theorem** — the same `a² + b² = c²` you already know. To measure how far apart two points are, you walk the horizontal gap, walk the vertical gap, and the straight-line distance is the hypotenuse:

```
        • point B (4, 4)
        │
   2.0  │  ← vertical gap (y₂−y₁) = 4−2 = 2
        │
  •─────┘
point A    horizontal gap (x₂−x₁) = 4−1 = 3
(1, 2)

  d = √(3² + 2²) = √(9+4) = √13 ≈ 3.6
```

That's all the formula says: **square each gap, add them, take the square root.** With 5 features instead of 2, you just have 5 gaps to square and add — the picture is the same, you simply can't draw it. "Nearest neighbor" = smallest `d` = the point with the least total gap across all features.

> 💡 Why square then square-root instead of just adding the gaps? Squaring makes all gaps positive (a gap of −3 and +3 are equally far) and the square root puts the answer back in the original units. It's the honest "as the crow flies" distance.

### Idea 2: Why high dimensions break KNN (the "curse")

KNN's whole bet is that *some neighbors are close and others are far*. In high dimensions that bet collapses — here's the concrete reason.

Imagine points scattered in a unit cube. In **1 dimension** (a line), two random points are often close. Add **more dimensions**, and for two points to be "close" they must happen to be close in *every single one* of those dimensions at once — which gets vanishingly unlikely:

```
 2 features:  points differ in 2 ways    → some pairs genuinely close
20 features:  must match on all 20 ways  → almost every pair is "medium-far"
200 features: nearest point is barely closer than the farthest point
```

```
distance to nearest neighbor   ≈   distance to farthest neighbor
        └──────────────── in high-D these become nearly equal ────────────────┘
```

When the closest and farthest points are about the same distance away, "the 5 nearest" is basically a random handful — and voting with random neighbors is useless. **That's the curse of dimensionality:** distance stops meaning similarity. The fix is to shrink the dimensions first (see [pca.md](pca.md)) or use a model that doesn't rely on distance at all.

Now the distance-metric and curse-of-dimensionality sections below are putting names on these two pictures.

---

## Why It Exists

### The Problem

Early classifiers were all linear — they drew straight-line boundaries. But many real-world decision boundaries are curved, irregular, and locally varying.

```
Linear classifiers: "I'll draw one straight line to separate all classes"
  Good for: globally separable data
  Bad for:  locally complex patterns

KNN: "I don't draw any line — I just look at what's nearby for each new point"
  Good for: locally varying, complex patterns
  Bad for:  large data (too slow), high dimensions (distances lose meaning)
```

### The Solution

Store all training data. At prediction time, find the closest examples and vote. No assumptions about the boundary shape — it adapts to the local structure of the data.

### What Changed

KNN is the conceptual foundation for many modern ideas: nearest-neighbor search (used in RAG retrieval), collaborative filtering (Netflix/Spotify recommendations), and anomaly detection (points with no close neighbors are outliers).

---

## Core Concepts

### 1. Distance Metrics

**One-line definition:** A formula that measures how "similar" or "different" two data points are.

**Analogy:** You can measure the "distance" between two cities on a map in different ways: straight-line (as the crow flies) or walking distance (along streets). Each is correct for different purposes.

```
Euclidean Distance (straight-line — most common):
  d = √((x₁-x₂)² + (y₁-y₂)²)
  
  Best for: continuous numeric features, when actual distance matters
  Example: geographic coordinates, physical measurements

Manhattan Distance (city-block — sum of absolute differences):
  d = |x₁-x₂| + |y₁-y₂|
  
  Best for: grid-like data, when diagonal movement isn't natural
  Example: pixel differences in images, route planning

Cosine Similarity (angle — not distance, but similarity):
  similarity = dot(A, B) / (|A| × |B|)
  
  Best for: text, high-dimensional sparse data
  Example: document similarity (same topics = small angle)
  Note: KNN usually minimizes distance, so use 1 - cosine_similarity
```

**Common misconception:** Euclidean distance works for everything. For text, Euclidean distance between TF-IDF vectors is meaningless — document length dominates. Use cosine similarity for text. For categorical features, use Hamming distance.

---

### 2. Choosing K

**One-line definition:** K is the number of neighbors to consult — balancing sensitivity (small K) against smoothness (large K).

**Analogy:** Medical second opinion. K=1 means trusting only the single closest match — fast but risky if that match is an outlier. K=100 means averaging 100 neighbors — stable but may blur important local patterns.

```
K=1:
  Pro: Captures finest local detail
  Con: Very sensitive to noise — one mislabeled neighbor flips the prediction
  
K=5 (good default):
  Pro: Balanced — local enough for detail, global enough for stability
  Con: May miss very fine-grained patterns
  
K=50:
  Pro: Very stable, robust to noise
  Con: May over-smooth — predicts the global majority in dense regions

Decision boundary:
  Small K → jagged, complex boundary (low bias, high variance)
  Large K → smooth, simple boundary (high bias, low variance)
  
              K=1              K=5             K=50
           ┌──────────┐    ┌──────────┐    ┌──────────┐
           │ × × ×  o│    │ × × ×  o│    │ × ×  o  o│
           │×   × o  │    │×   × o  │    │×   o o  o│
           │× ×   o  │    │× ×   o  │    │× o   o  o│
           └──────────┘    └──────────┘    └──────────┘
           (noisy)         (balanced)      (over-smooth)
```

**Common misconception:** The optimal K can be guessed. It can't — always use cross-validation to find the best K for your specific data.

---

### 3. Lazy Learning (No Training)

**One-line definition:** KNN stores the entire training dataset without learning any model parameters — all computation happens at prediction time.

**Analogy:** The difference between a textbook (pre-learned knowledge, instant lookup) and a library (raw books, need to search through them every time you have a question). KNN is the library — no pre-processing, but every query requires a full search.

```
Training phase:
  Decision Tree: learns rules → fast prediction (walk the tree)
  SVM: learns weights → fast prediction (one dot product)
  KNN: stores all data → prediction requires full data scan

  Training time: O(1) — nothing to learn!
  Prediction time: O(n × d) — compare to ALL n examples across ALL d features
  
  n=1M examples, d=100 features → 100M operations per prediction → SLOW
```

**Common misconception:** "Lazy" means bad. Lazy learning is excellent for small datasets and problems where the data itself changes frequently — you just add/remove examples without retraining any model.

---

### 4. The Curse of Dimensionality

**One-line definition:** In high-dimensional spaces, all points become roughly equidistant from each other, so "nearest neighbor" loses its meaning.

**Analogy:** In 2D, your "nearest neighbor" might be 10 meters away. In 1000D, everyone is roughly 100,000 units away — the closest person is barely closer than the farthest. Finding a "nearby" neighborhood becomes impossible.

```
Intuition: add dimensions, all points spread out

2D:   100 points in a 1×1 square
      Average nearest-neighbor distance: ~0.1
      Ratio (nearest/farthest): 0.05  ← clearly closest

100D: same 100 points in a 100D unit hypercube
      Average nearest-neighbor distance: ~8
      Ratio (nearest/farthest): 0.97  ← nearly all the same distance

When nearest ≈ farthest → KNN votes become random → model fails
```

Fix: Reduce dimensions with PCA or UMAP before applying KNN.

**Common misconception:** More features = more information = better KNN. In KNN, more features usually hurt because distances become meaningless. Feature selection or dimensionality reduction is critical.

---

### 5. Feature Scaling — Mandatory

**One-line definition:** Features must be on the same scale so no single feature dominates the distance calculation.

**Analogy:** If you're comparing "how similar" two people are using salary (range $0-$200K) and age (range 0-100), salary dominates 2000:1. A $1000 salary difference counts the same as the entire range of age.

```
Without scaling:
  Person A: salary=$30,000, age=25
  Person B: salary=$30,100, age=65
  Person C: salary=$60,000, age=26
  
  Euclidean distance A→B = √((100)² + (40)²) = 107    ← "closest" (wrong!)
  Euclidean distance A→C = √((30000)² + (1)²) = 30000  ← "far" (actually similar age!)
  
  Salary dominates completely. Age is ignored.

With StandardScaler (mean=0, std=1):
  Same data, now salary and age contribute equally to distances.
```

**Common misconception:** Scaling is optional if features are already in similar units. Even similar units can have very different variances. Always use StandardScaler or MinMaxScaler before KNN.

---

> 🧠 **Quick recall — answer out loud before scrolling on** (all answers are above):
> 1. Euclidean distance for 5 features — describe the formula in plain words.
> 2. K=1 vs K=50 — which is high variance, which is high bias?
> 3. Where does KNN's computation happen: training time or prediction time?
> 4. Why does "nearest neighbor" stop meaning anything at 200 features?
> 5. Unscaled salary next to age — what happens to the distance calculation?

---

## How It Actually Works (Step-by-Step)

```
Training data (stored, not learned):
  Point  | Feature 1 | Feature 2 | Label
  A      | 1.0       | 2.0       | Cat
  B      | 1.5       | 1.8       | Cat
  C      | 4.0       | 4.5       | Dog
  D      | 4.5       | 4.2       | Dog
  E      | 1.2       | 3.0       | Cat

New point: ? = [1.3, 2.5], K=3

Step 1: Compute distance from ? to ALL training points
  d(?, A) = √((1.3-1.0)² + (2.5-2.0)²) = √(0.09 + 0.25) = 0.58
  d(?, B) = √((1.3-1.5)² + (2.5-1.8)²) = √(0.04 + 0.49) = 0.73
  d(?, C) = √((1.3-4.0)² + (2.5-4.5)²) = √(7.29 + 4.00) = 3.36
  d(?, D) = √((1.3-4.5)² + (2.5-4.2)²) = √(10.24 + 2.89) = 3.62
  d(?, E) = √((1.3-1.2)² + (2.5-3.0)²) = √(0.01 + 0.25) = 0.51

Step 2: Sort by distance, take K=3 nearest
  1st: E (0.51) → Cat
  2nd: A (0.58) → Cat
  3rd: B (0.73) → Cat

Step 3: Vote
  Cat: 3 votes, Dog: 0 votes

Step 4: Predict: CAT (100% confident)
```

---

## Code in Practice

### 1. Hello World — Basic KNN

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Scale features — CRITICAL for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

print(f"Accuracy: {model.score(X_test_scaled, y_test):.2%}")
```

### 2. Practical — Finding the Best K

```python
from sklearn.model_selection import cross_val_score
import numpy as np

# Try K=1 through K=20, use cross-validation to pick best
k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # 5-fold cross-validation (no test set contamination)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"K={k:2d}: {scores.mean():.3f} ± {scores.std():.3f}")

best_k = k_values[np.argmax(cv_scores)]
print(f"\nBest K: {best_k} with CV accuracy: {max(cv_scores):.3f}")

# Train final model with best K
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train_scaled, y_train)
print(f"Test accuracy: {final_model.score(X_test_scaled, y_test):.2%}")
```

### 3. Real-World Pattern — Pipeline + Regression

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# Generate sample data: predict house price from features
np.random.seed(42)
X = np.random.randn(500, 5)   # 500 houses, 5 features
y = 200000 + 50000*X[:,0] + 30000*X[:,1] + np.random.randn(500)*10000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Pipeline ensures scaler fit only on training data (no leakage)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=7))  # Regression: averages neighbors
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(f"MAE: ${mean_absolute_error(y_test, y_pred):,.0f}")
```

---

## Gotchas & Pitfalls

```
❌ Forgetting to scale features
   Salary dominates distances, age is ignored — your "nearest neighbors" are wrong
✅ ALWAYS StandardScaler() before KNN — no exceptions

❌ Using K=1 for noisy data
   One mislabeled or outlier point completely decides the prediction
✅ Start with K=5 (odd number), find optimal K via cross-validation

❌ Fitting the scaler on the full dataset (data leakage)
   Test data statistics leaked into training → inflated accuracy
✅ Use Pipeline or: fit scaler on train only, transform both train and test

❌ Using KNN on high-dimensional data without reduction
   Distances lose meaning in >20-30 dimensions (curse of dimensionality)
✅ Apply PCA first to reduce to 10-20 meaningful dimensions

❌ Using KNN on large datasets
   KNN is O(n×d) per prediction — 1M rows means 1M distance computations per query
✅ For large data, use approximate nearest neighbors (FAISS, Annoy) or switch algorithms

❌ Using even K for binary classification
   K=4: might get 2-2 tie → random prediction
✅ Always use odd K for binary classification to avoid ties
```

---

## When to Use / When NOT to Use

### Use KNN When:
- Small dataset (<10K samples) where computation time isn't a concern
- Non-linear, locally complex decision boundaries
- Quick prototype — no training required, just store data and go
- Recommendation systems (find similar users/items)

### Don't Use KNN When:
- Large datasets — it's too slow at prediction time
- High-dimensional data — curse of dimensionality makes distances meaningless
- You need feature importance or model interpretability
- Real-time predictions at scale (millisecond latency requirements)

---

## Related Concepts

| Concept | Connection |
|---------|------------|
| PCA | Dimensionality reduction that saves KNN from the curse of dimensionality |
| Vector Databases | Approximate nearest neighbor search at scale — KNN's "grown-up" cousin for ML apps |
| Collaborative Filtering | Recommendation systems use KNN logic: find similar users, recommend what they liked |
| Distance Metrics | The choice of distance metric completely changes what KNN considers "similar" |
| Cross-Validation | Essential for finding the optimal K — never guess, always measure |

---

## Cheat Sheet

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Classification
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
])

# Regression (predicts numbers, not classes)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=5))
])

Key parameters:
  n_neighbors: K value (default 5, always tune via cross-validation)
  metric: 'euclidean' (default), 'manhattan', 'cosine'
  weights: 'uniform' (all neighbors equal) or 'distance' (closer = more weight)

How to find optimal K:
  cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=5)

Remember:
  1. Scale features — it's mandatory, not optional
  2. Use odd K to avoid ties in binary classification
  3. KNN fails in high dimensions — apply PCA first if features > 20
```

---

## Self-Check Questions

<details>
<summary>Click to reveal answers</summary>

**Q1: Why is KNN called a "lazy learner"?**
Because it does nothing during the "training" phase — it just stores all the data. All computation happens at prediction time, when it computes distances to every stored point. Compare to "eager learners" like decision trees or logistic regression, which extract patterns during training so predictions are fast (just evaluate a formula or walk a tree).

**Q2: What is the curse of dimensionality and how does it break KNN?**
In high dimensions (many features), all data points become roughly equidistant from each other. If the nearest point is almost as far as the farthest point, "nearest neighbor" is meaningless — you're essentially picking randomly. The solution is dimensionality reduction (PCA) before applying KNN.

**Q3: You're building a movie recommender. How does KNN apply?**
Represent each user as a vector of their movie ratings. To recommend movies for user A, find the K users most similar to A (smallest Euclidean or cosine distance). Take the movies those users rated highly that A hasn't seen yet. That's collaborative filtering — KNN on user profiles.

**Q4: Why should you use an odd number for K in binary classification?**
To avoid ties. With even K (e.g., K=4), you might get 2 votes for class 0 and 2 votes for class 1 — a tie with no clear winner. With odd K (e.g., K=5), one class always gets at least 3 votes, guaranteeing a majority winner.

**Q5: A new data point arrives. The 5 nearest neighbors are at distances [0.1, 0.1, 0.1, 10.0, 10.0] with labels [A, A, A, B, B]. Using KNN with K=5 (uniform weights), how might this prediction be wrong, and what fixes it?**
With uniform weights: A gets 3 votes, B gets 2 → predicts A. With distance weights: the three A neighbors at 0.1 contribute weight 10 each (1/0.1), while the two B neighbors at 10.0 contribute 0.1 each. A gets weight 30, B gets 0.2 → strongly predicts A. In this case both agree, but for boundary cases, `weights='distance'` gives more weight to closer neighbors and is generally more accurate. Set `KNeighborsClassifier(weights='distance')`.

</details>

---

## Go Deeper

| Resource | Why It's Worth Your Time |
|----------|--------------------------|
| [StatQuest: KNN](https://www.youtube.com/watch?v=HVXime0nQeI) | Visual walkthrough of how KNN makes predictions and how K affects the boundary. Clear and concise. |
| [scikit-learn KNN docs](https://scikit-learn.org/stable/modules/neighbors.html) | Official docs covering distance metrics, algorithm variants (ball-tree, KD-tree), and regression. |
| [Curse of Dimensionality (explained visually)](https://towardsdatascience.com/the-curse-of-dimensionality-50dc6e49aa1e) | The best visual explanation of why distances fail in high dimensions. |
| [FAISS by Meta](https://github.com/facebookresearch/faiss) | When KNN is too slow: FAISS implements approximate nearest neighbor search on billions of vectors. The production-scale version of KNN. |
| [Building a Recommender System](https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system) | Apply KNN to build a real movie recommender on Kaggle — connects theory directly to a real product. |
