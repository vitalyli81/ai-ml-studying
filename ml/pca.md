# PCA (Principal Component Analysis)

## TL;DR

PCA reduces the number of features in your data while keeping as much information as possible. It finds the directions where your data varies the most and projects everything onto those new axes. Use it when you have too many features slowing down models, want to visualize high-dimensional data in 2D/3D, or need to remove noise. It's a preprocessing step — you still need a classifier or regressor after PCA. Always scale your data first.

> 💡 **Key Insight:** PCA doesn't select features — it creates NEW features (principal components) that are combinations of the originals. These new features are uncorrelated with each other and ordered by how much variance they explain. You keep the most important ones and discard the rest.

---

## The Mental Model

Think of **photographing a 3D sculpture from different angles to find the most revealing shot**.

A face sculpture viewed from the front gives the most information — you see eyes, nose, mouth, all clearly separated. The top view just shows a flat oval. The front-facing photograph is the "best 2D shadow" of the 3D object.

Mapping:
- The 3D sculpture → your high-dimensional data
- Trying different camera angles → PCA searching for the best projection directions
- The most revealing angle → PC1 (the direction of maximum variance)
- Second-best angle → PC2 (perpendicular to PC1, second-most variance)
- The 2D photograph → your reduced-dimension data
- Information lost by flattening → variance NOT explained by kept components

PCA rotates the axes of your data space to align with the directions of maximum spread, then lets you discard the low-variance directions.

---

## Build the Intuition From Zero

PCA feels abstract because of three words: **"direction," "projection," and "eigenvector."** Let's make all three physical before any matrices show up.

### Idea 1: A "direction" is just a new ruler you lay across the data

Plot people by height and weight. The dots form a tilted, stretched-out blob — because tall people tend to weigh more, the cloud leans diagonally:

```
weight
  │            • •
  │         • • •
  │      • • •            ← the cloud is long in the diagonal direction,
  │   • • •                  thin across it
  │ • •
  └─────────────── height
```

A **direction** is just an arrow you could lay across this cloud, like a ruler. PCA's question: *along which ruler is the data most spread out?* Obviously the diagonal one — the cloud is longest that way. That diagonal is **PC1**. The ruler perpendicular to it (across the thin width) is **PC2**.

### Idea 2: A "projection" is the shadow of each point onto that ruler

Once you pick a ruler (direction), **projecting** a point means dropping it straight down onto that ruler and reading off where it lands — its shadow:

```
        • P                    Lay a ruler (PC1) under the cloud.
       ╱                       Each point P casts a shadow ↓ onto it.
   ───•────────── PC1 ruler    The shadow's position = that point's
      ↑ P's shadow (one number)   PC1 value: ONE number replacing two.
```

That's the whole compression: instead of storing each point's `(height, weight)` — two numbers — you store just where its shadow falls on PC1 — one number. You picked the diagonal ruler precisely because the shadows are most spread out there, so you lose the least information.

> 💡 **PCA in one line:** find the ruler the data is most spread along, replace each point with its shadow on that ruler. Add a second perpendicular ruler if one isn't enough. "Reduce dimensions" = "keep the few rulers that capture the spread, drop the rest."

### Idea 3: Eigenvectors and eigenvalues — the scary words, demystified

You don't find the best ruler by guessing. There's a matrix (the **covariance matrix**) that encodes how the data is stretched in every direction. When you "decompose" it, it hands you two things, in matched pairs:

```
eigenvector  =  a direction (a ruler)              → "spread is along HERE"
eigenvalue   =  how much spread along that ruler   → "and it's THIS much"
```

- **Eigenvector** = the *arrow* (which way the ruler points). These become your principal components.
- **Eigenvalue** = a *number* measuring how stretched the cloud is along that arrow. Big eigenvalue = long, informative direction (keep it). Tiny eigenvalue = thin, boring direction, probably noise (drop it).

So "PC1 explains 65% of variance" literally means: PC1's eigenvalue is 65% of the sum of all eigenvalues. Sort the pairs by eigenvalue, keep the top few arrows, project onto them — that *is* PCA. The matrix math is just the machine that finds the rulers for you; the idea is "longest rulers first."

Now the covariance-matrix and eigenvector sections below are naming things you can already picture.

---

## Why It Exists

### The Problem: The Curse of Dimensionality

More features = better, right? Not always. Many ML algorithms break or slow down catastrophically with many features:

```
Dataset: 1,000 medical features per patient

KNN: compares every point to every other across 1,000 dimensions
     → all distances become roughly equal → "nearest" means nothing
     
SVM: kernel computations scale badly with dimension
     → hours to train, poor generalization
     
Visualization: impossible to see structure in 1,000 dimensions
     → can't even look at your data to understand it
```

Also: many features are correlated. Height and weight. Income and spending. Temperature in Celsius and Fahrenheit. Correlated features contain **redundant information** — the model sees the same information twice and gets confused.

### The Solution

Find a lower-dimensional representation that captures most of the variance. Compress 1,000 features down to 20 principal components that retain 95% of the information.

### What Changed

PCA (invented in 1901 by Karl Pearson) became the workhorse of data preprocessing. Today it's used before neural networks, SVMs, and clustering to speed up training, improve generalization, and enable visualization.

---

## Core Concepts

### 1. Variance — What PCA Maximizes

**One-line definition:** Variance measures how spread out your data is along a direction — PCA finds the directions with the most spread.

**Analogy:** Think of rain falling on a field. Some areas have a wide spread of puddles; others are dry. PCA finds the "rainiest" direction — the axis along which the data varies most.

```
Data points: body measurements [height, weight, BMI, waist, chest]

Height varies a lot (1.5m to 2.0m) ← high variance
BMI varies a moderate amount       ← medium variance
Exact finger length varies little  ← low variance

PCA finds the combination of features that creates a new axis
with maximum variance — this becomes PC1.
```

**Common misconception:** PCA selects the features with highest variance. PCA creates NEW features as linear combinations of ALL original features. It doesn't just pick height over BMI — it might create PC1 = 0.6×height + 0.5×weight + 0.3×BMI + ...

---

### 2. Principal Components

**One-line definition:** New axes (directions) found by PCA, ordered by how much variance they explain, where each is perpendicular to all the others.

**Analogy:** Imagine a long, thin cloud of points shaped like a football. PC1 runs along the football's length (most spread). PC2 runs across its width (second most spread). PC3 would go through the thickness. They're perpendicular — independent — and ordered by importance.

```
Original features: [height, weight, BMI, waist, chest]  ← correlated

After PCA:
  PC1 = 0.58×height + 0.57×weight + 0.45×BMI + ...   ← "body size" (65% variance)
  PC2 = 0.72×height - 0.45×weight - 0.21×BMI + ...   ← "height vs weight" (20% variance)
  PC3 = ...                                             ← (10% variance)
  PC4, PC5: (tiny remaining variance — probably noise)

Keep PC1 and PC2 → retain 85% of variance, discard 15%
                 → 5 features → 2 features (60% reduction)
```

**Common misconception:** Principal components are interpretable. They're not — PC1 is a mathematical combination of all features. You can no longer say "feature 2 is most important" — instead you say "85% of the data's variance is captured by 2 components."

---

### 3. Explained Variance Ratio

**One-line definition:** The fraction of total variance in the dataset explained by each principal component — tells you how many components to keep.

**Analogy:** Compression levels on a ZIP file. At 10% compression, you keep 99% of the data. At 90% compression, you keep 50%. PCA's "explained variance" tells you how much information you retain at each compression level.

```
PCA on a 100-feature dataset:

PC1:  explains 35% of variance
PC2:  explains 20% of variance
PC3:  explains 15% of variance
PC4:  explains 10% of variance
PC5:  explains  8% of variance
...
PC100: explains 0.01% of variance

Cumulative:
  2 components: 55% variance retained
  3 components: 70% variance retained
  5 components: 88% variance retained  ← good threshold
  10 components: 95% variance retained ← common target
  100 components: 100% (no reduction)
```

**Common misconception:** You should always explain 95% of variance. For visualization, 2-3 components (even if only 60% variance) is standard. For preprocessing, 90-95% is common. For image compression, sometimes 80% is enough. The right threshold depends on your downstream task.

---

### 4. The Covariance Matrix — The Math Under the Hood

**One-line definition:** A matrix that captures how much every pair of features varies together — PCA decomposes this to find principal components.

**Analogy:** A correlation heatmap shows which features move together. PCA processes the covariance matrix (the mathematical version of that heatmap) to find the directions that explain all that correlated movement.

```
For 3 features [height, weight, BMI]:

Covariance matrix:
         height  weight   BMI
height [  4.0     3.2     2.1 ]
weight [  3.2     9.0     4.5 ]
BMI    [  2.1     4.5     2.5 ]

High values = features move together (correlated)
Diagonal = variance of each feature

PCA decomposes this into eigenvectors (directions) and eigenvalues (amounts of variance).
Eigenvectors = principal components
Eigenvalues  = variance explained by each component
```

**Common misconception:** You need to understand the math to use PCA. You don't — scikit-learn handles everything. Understanding the concept (find directions of maximum variance that are uncorrelated) is enough.

---

### 5. Scale Before PCA

**One-line definition:** PCA is variance-based — if features have different scales, the high-scale feature dominates all principal components.

**Analogy:** You're measuring the "spread" of a dataset in different units. Speed in km/h ranges from 0 to 200. Speed in m/s ranges 0 to 56. If you mix them with body temperature (36-38°C), temperature has tiny variance and PCA ignores it — even if it's the most medically relevant feature.

```
Without scaling:
  Salary: [30K, 50K, 200K] → variance = huge (millions)
  Age:    [20, 30, 65]     → variance = tiny (hundreds)
  
  PCA will find PC1 ≈ "salary direction" because salary varies so much more
  Age gets nearly ignored even if it's important

With StandardScaler first:
  Salary_scaled: [-0.8, -0.4, 1.2] → variance = ~1
  Age_scaled:    [-0.8, -0.4, 1.2] → variance = ~1
  
  Now PCA treats both equally and finds truly informative directions
```

**Common misconception:** Scaling is optional if features are similar. Even "similar" units (all in meters) can have vastly different variances. StandardScaler is ALWAYS needed before PCA.

---

## How It Actually Works (Step-by-Step)

```
Dataset: 3 features, 6 patients
[height_cm, weight_kg, age]

Step 1: Center the data (subtract mean from each feature)
  height: mean=170, weight: mean=70, age: mean=40
  Each feature now has mean=0

Step 2: Scale (divide by standard deviation)
  Each feature now has std=1 (StandardScaler does steps 1+2)

Step 3: Compute covariance matrix
  3×3 matrix showing how each pair of features varies together

Step 4: Find eigenvectors and eigenvalues
  Eigenvectors = directions of maximum variance (principal components)
  Eigenvalues  = variance explained by each direction

Step 5: Sort by eigenvalue (largest first)
  PC1 = [0.58, 0.57, 0.58] → explains 72% of variance
  PC2 = [0.71, -0.70, 0.05] → explains 25% of variance
  PC3 = [-0.40, -0.42, 0.81] → explains 3% of variance

Step 6: Project data onto top K components
  Original: 6 patients × 3 features
  After PCA(n_components=2): 6 patients × 2 principal components

Step 7: Use the 2D data for ML or visualization
  Now you can plot all 6 patients in 2D and see if they cluster
```

---

## Code in Practice

### 1. Hello World — Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target  # 4 features, 150 samples

# MUST scale before PCA
X_scaled = StandardScaler().fit_transform(X)

# See how much variance each component explains
pca_full = PCA()
pca_full.fit(X_scaled)
print("Variance per component:", pca_full.explained_variance_ratio_)
# [0.73, 0.23, 0.04, 0.00]
# 2 components explain 96% of variance!

# Reduce to 2 components
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
print(f"Shape: {X.shape} → {X_2d.shape}")  # (150, 4) → (150, 2)
print(f"Variance retained: {sum(pca.explained_variance_ratio_):.1%}")
```

### 2. Practical — Keep Enough Variance

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np

# Handwritten digits: 64 features (8x8 pixels)
digits = load_digits()
X = digits.data   # 1797 samples × 64 features

X_scaled = StandardScaler().fit_transform(X)

# Find how many components for 95% variance
pca = PCA(n_components=0.95)   # Keep components until 95% variance
X_reduced = pca.fit_transform(X_scaled)

print(f"Original: {X.shape[1]} features")
print(f"After PCA: {X_reduced.shape[1]} components")   # ~29 components
print(f"Variance retained: {sum(pca.explained_variance_ratio_):.1%}")

# Show cumulative variance by number of components
cumvar = np.cumsum(PCA().fit(X_scaled).explained_variance_ratio_)
for n in [5, 10, 20, 29, 40]:
    print(f"  {n} components: {cumvar[n-1]:.1%}")
```

### 3. Real-World Pattern — PCA then Classifier

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# PCA as a preprocessing step in a Pipeline (prevents data leakage)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=30)),   # Reduce 64 → 30 features
    ('classifier', SVC(kernel='rbf', C=10))
])

pipe.fit(X_train, y_train)
print(f"Test accuracy with PCA: {pipe.score(X_test, y_test):.2%}")

# Compare: without PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

svm_no_pca = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=10))])
svm_no_pca.fit(X_train, y_train)
print(f"Test accuracy without PCA: {svm_no_pca.score(X_test, y_test):.2%}")
# Often similar accuracy, but PCA version trains much faster
```

---

## Gotchas & Pitfalls

```
❌ Not scaling before PCA
   High-variance features (salary, pixel values) dominate all principal components
✅ StandardScaler().fit_transform(X) BEFORE PCA — every time, no exceptions

❌ Fitting PCA on the full dataset (including test data)
   Test data statistics leak into training → inflated evaluation scores
✅ Use Pipeline: PCA.fit() on training data only, .transform() on test data

❌ Using PCA to remove features for interpretability
   PC1 = "0.58×age + 0.57×weight + ..." — you can't say "feature X is important"
✅ Use feature importance (Random Forest) or Lasso if you need feature-level interpretation

❌ Keeping too few components to hit a "round" number (e.g., always 10)
   You might throw away important variance
✅ Use n_components=0.95 to automatically keep 95% of variance

❌ Applying PCA to non-linear data and expecting magic
   PCA is linear — it can't unfurl a Swiss roll or separate concentric rings
✅ For non-linear reduction, use t-SNE (visualization) or UMAP (general)

❌ Using PCA when you have very few samples
   PCA can find spurious components if n_samples < n_features
✅ Ensure you have at least 10× more samples than features before PCA
```

---

## When to Use / When NOT to Use

### Use PCA When:
- Features > 20-50 and algorithms are slow (KNN, SVM, neural networks)
- Visualizing high-dimensional data in 2D or 3D scatter plots
- Features are highly correlated (multicollinearity confuses linear models)
- You want to denoise data (small variance components are often noise)

### Don't Use PCA When:
- You need feature-level interpretation ("which original feature matters?")
- Data has non-linear structure (use t-SNE or UMAP instead)
- You have very few features (no benefit, adds complexity)
- You're using tree-based models (decision trees, Random Forest — they don't benefit from PCA)

---

## Related Concepts

| Concept | Connection |
|---------|------------|
| t-SNE | Non-linear dimensionality reduction — better for visualization, not preprocessing |
| UMAP | Faster, more scalable non-linear reduction — popular replacement for t-SNE |
| K-Means | Often paired with PCA: reduce dimensions first, then cluster |
| Feature Scaling | Mandatory prerequisite — PCA is fundamentally variance-based |
| Feature Selection | Alternative to PCA: keeps original features instead of creating new ones (interpretable) |

---

## Cheat Sheet

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Always in a Pipeline to prevent data leakage:
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=20))   # or n_components=0.95 for auto
])

X_train_reduced = pipe.fit_transform(X_train)  # fit+transform on train
X_test_reduced  = pipe.transform(X_test)        # transform only on test

# Key attributes after fitting:
pca.explained_variance_ratio_    # Variance per component (array)
pca.n_components_                 # Number of components actually kept
pca.components_                   # The PC directions (eigenvectors)

np.cumsum(pca.explained_variance_ratio_)  # Cumulative variance curve

Choosing n_components:
  Visualization → 2 or 3
  Preprocessing → use n_components=0.95 (keeps 95% of variance)
  Rule of thumb → aim for 90-95% variance retained

PCA vs t-SNE:
  PCA:   linear, fast, good for preprocessing ML pipelines
  t-SNE: non-linear, slow, only for visualization (2D/3D plots)

Remember:
  1. Scale first — always, no exceptions
  2. Use Pipeline to fit on train only
  3. PCA creates new features (not interpretable) — use feature importance instead if you need interpretability
```

---

## Self-Check Questions

<details>
<summary>Click to reveal answers</summary>

**Q1: Why must you scale features before PCA?**
PCA finds directions of maximum variance. If salary ranges from $0 to $200,000 and age from 0 to 100, salary's variance is millions of times larger than age's. Without scaling, PC1 is essentially "the salary direction" — PCA completely ignores age. StandardScaler puts all features on equal footing so PCA can find truly informative directions.

**Q2: What does it mean when someone says "PC1 explains 65% of the variance"?**
If you project all data points onto the PC1 axis, you retain 65% of the total variability in the dataset. The other 35% is spread across PC2, PC3, etc. A higher percentage means less information is lost when using that component — PC1 with 65% is very informative; PC10 with 1% is mostly noise.

**Q3: How many principal components should you keep?**
Common targets: (1) Visualization → always 2 or 3. (2) Preprocessing for ML → enough to explain 90-95% of variance (use `PCA(n_components=0.95)`). (3) Check the elbow in the explained variance curve — keep components before the curve flattens. The right answer depends on your use case and how much accuracy loss you can tolerate.

**Q4: You applied PCA and your model accuracy dropped from 92% to 88%. What might have happened?**
You either kept too few components (threw away too much information) or the features that PCA discarded were actually important for the specific patterns the model was learning. Try increasing n_components (aim for 95%+ variance). Also consider that for tree-based models (Random Forest, XGBoost), PCA typically doesn't help — those models handle high-dimensional data natively.

**Q5: What's the difference between PCA and feature selection?**
PCA creates brand-new features (linear combinations of all original features) and discards the originals. You can't say "feature X was important." Feature selection (using Lasso or tree feature importance) keeps a SUBSET of original features, discarding unimportant ones. Feature selection is interpretable ("age and income matter most"). PCA gives better compression but loses interpretability.

</details>

---

## Go Deeper

| Resource | Why It's Worth Your Time |
|----------|--------------------------|
| [StatQuest: PCA Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ) | The definitive visual explanation. Josh Starmer builds PCA from scratch using the covariance matrix. Watch this first — nothing else compares. |
| [scikit-learn PCA docs](https://scikit-learn.org/stable/modules/decomposition.html#pca) | Official docs with examples for dimensionality reduction, noise filtering, and incremental PCA for large datasets. |
| [Understanding PCA with Code](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html) | Jake VanderPlas's Python Data Science Handbook chapter — the best code-first explanation with great visualizations. |
| [t-SNE vs UMAP](https://pair-code.github.io/understanding-umap/) | When PCA isn't enough (non-linear structure), these are the next tools. This interactive guide explains both. |
| [Google's Embedding Projector](https://projector.tensorflow.org) | Interactive tool that lets you visualize high-dimensional data using PCA and t-SNE on real datasets. Best way to build intuition. |
