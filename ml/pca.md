# PCA (Principal Component Analysis)

## What Is It?

PCA **reduces the number of features** in your data while keeping as much information as possible. It finds the directions where your data varies the most and projects everything onto those directions.

Think of it like taking a 3D object and finding the best 2D shadow that preserves the most shape detail.

## Real-World Examples

- **Visualization** — project 100-dimensional data down to 2D for a scatter plot
- **Speed up ML models** — reduce features from 1000 to 50 before training
- **Image compression** — represent face images with fewer numbers
- **Noise reduction** — small components are often noise, removing them cleans the data
- **Fix KNN/SVM** — these algorithms break in high dimensions (curse of dimensionality)

## How It Works (Step by Step)

### 1. The Problem

You have data with many correlated features:

```
Height (cm): [170, 175, 160, 180, 165]
Weight (kg): [ 70,  75,  55,  85,  60]
BMI:         [ 24,  24,  21,  26,  22]
```

Height, weight, and BMI are all related. Do we really need 3 numbers? PCA says: maybe 1 or 2 is enough.

### 2. Find the Direction of Maximum Variance

PCA rotates the data to find new axes (principal components) that capture the most spread:

```
Original axes:              PCA axes:
  Weight                      PC1 (most variance)
  |    . .                        /. .
  |  . .                        /. .
  |. .                        /. .
  |_________ Height          /________ PC2 (less variance)
```

**PC1** captures the main trend (bigger people are taller AND heavier).
**PC2** captures the remaining variation (some people are tall-and-thin vs short-and-heavy).

### 3. Keep Only the Top Components

If PC1 explains 90% of the variance, you can drop PC2 and go from 2D to 1D with only 10% information loss.

```
Original: 3 features (height, weight, BMI)
After PCA: 1 feature (PC1 = "body size") — captures 95% of the variation
```

### 4. How Many Components to Keep?

Plot the **explained variance ratio**:

```
Variance Explained
100% |          ___________
     |        /
 80% |      /
     |    /         ← 2 components capture 95%
 60% |   /
     |  /
 40% | /
     |/
  0% |__________________ Components
     1    2    3    4   5
```

**Rule of thumb**: Keep enough components to explain **90-95% of the variance**.

## Frontend Analogy

PCA is like responsive image serving:

```javascript
// Original: 4000x3000 image (12M pixels)
// Medium:   1200x900  (1M pixels)  — looks almost the same!
// Small:    400x300   (120K pixels) — still recognizable

// PCA does this for data:
// Original: 100 features
// PCA 20:   20 features — retains 95% of information
// PCA 5:    5 features  — retains 80% of information
```

You're trading a little fidelity for a huge reduction in size.

## Important Notes

### 1. Scale Your Data First

PCA is based on variance. If one feature is in thousands and another in decimals, the big feature dominates.

```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)  # ALWAYS do this before PCA
```

### 2. Components Are Not Interpretable

PC1 is not "height" or "weight" — it's a **mix** of all features:

```
PC1 = 0.58 × height + 0.57 × weight + 0.58 × BMI
```

You lose the ability to say "this feature matters." It's a tradeoff: simplicity for interpretability.

### 3. PCA is Linear

It only finds straight-line relationships. For curved patterns, look into t-SNE or UMAP (for visualization) or kernel PCA (for non-linear reduction).

## When to Use It

| Good For | Bad For |
|----------|---------|
| Reducing high-dimensional data | When you need to interpret features |
| Speeding up slow algorithms (KNN, SVM) | When all features are already independent |
| Visualization (project to 2D/3D) | Non-linear relationships (use t-SNE/UMAP) |
| Removing noise | When you have very few features already |
| Fixing multicollinearity | Categorical data (use MCA instead) |

## PCA vs Feature Selection

| PCA | Feature Selection |
|-----|-------------------|
| Creates **new** features (combinations) | Keeps **existing** features |
| Not interpretable | Interpretable |
| Uses all features (mixed) | Drops some features entirely |
| Better for compression | Better when you need to explain results |

## Python Example

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np

# Load handwritten digits (8x8 images = 64 features)
digits = load_digits()
X = digits.data  # 64 features per image
print(f"Original shape: {X.shape}")  # (1797, 64)

# Step 1: Scale
X_scaled = StandardScaler().fit_transform(X)

# Step 2: See how many components we need
pca_full = PCA()
pca_full.fit(X_scaled)

# Cumulative variance explained
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
for n in [5, 10, 20, 30]:
    print(f"  {n} components: {cumulative_var[n-1]:.1%} variance explained")

# Step 3: Reduce to 20 components (captures ~90%)
pca = PCA(n_components=20)
X_reduced = pca.fit_transform(X_scaled)
print(f"\nReduced shape: {X_reduced.shape}")  # (1797, 20)
print(f"Variance retained: {sum(pca.explained_variance_ratio_):.1%}")

# Step 4: Use reduced data for faster ML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, digits.target, test_size=0.2, random_state=42
)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(f"Accuracy with 20 components: {knn.score(X_test, y_test):.2%}")
# Almost as good as using all 64 features, but 3x faster!
```

## Key Takeaway

PCA is **dimensionality reduction** — fewer features, faster models, easier visualization. Always scale first, keep enough components for 90-95% variance, and use it when you have too many features slowing things down. It's a preprocessing step, not a model — you still need a classifier/regressor after PCA.
