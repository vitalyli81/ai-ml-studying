# Support Vector Machine (SVM)

## What Is It?

SVM finds the **best boundary** (line, plane, or curved surface) that separates two classes with the **widest possible margin**. It doesn't just find *any* dividing line — it finds the one with the most breathing room between classes.

Think of it like building a wall between two groups of people, making the wall as thick as possible.

## Real-World Examples

- **Image classification** — handwritten digit recognition
- **Text classification** — sentiment analysis (positive/negative)
- **Bioinformatics** — cancer vs healthy tissue from gene data
- **Face detection** — face vs not-face in image regions

## How It Works (Step by Step)

### 1. Find the Best Separating Line

Imagine dots on a 2D plane — red class and blue class:

```
    Blue  o  o
         o    o          ← margin →
  ─────────────────── ← decision boundary (hyperplane)
         x    x          ← margin →
    Red   x  x
```

SVM picks the line that **maximizes the margin** — the distance between the line and the nearest points from each class.

### 2. Support Vectors — The Key Points

The data points **closest to the boundary** are called **support vectors**. They're the only points that actually matter for defining the boundary. Move any other point and the boundary stays the same.

```
         o              ← not a support vector (far away)
         o  ← support vector (closest to boundary)
  ────────────────────
         x  ← support vector (closest to boundary)
         x              ← not a support vector (far away)
```

### 3. The Kernel Trick (Handling Non-Linear Data)

What if a straight line can't separate the data?

```
    x x o o o x x     ← can't draw a straight line here
```

The **kernel trick** maps data into a higher dimension where it *can* be separated by a flat boundary:

```
  2D (not separable)     →     3D (separable!)
  x x o o o x x         →     x's stay low, o's rise up
                               Now a flat plane separates them
```

Common kernels:

| Kernel | Use When |
|--------|----------|
| **Linear** | Data is already linearly separable (or high-dimensional like text) |
| **RBF (Gaussian)** | Default choice. Handles most non-linear patterns |
| **Polynomial** | You expect polynomial-shaped boundaries |

### 4. Soft Margin — Allowing Some Mistakes

Real data is messy. The **C parameter** controls how strict the boundary is:

- **Large C** → narrow margin, fewer misclassifications (may overfit)
- **Small C** → wide margin, allows some misclassifications (may underfit)

```
Large C (strict):          Small C (relaxed):
  o o|x   x               o  o  |  x  x
  o  | x  x                o   |   x  x
  o  |  x x               o   x|  x      ← allows this mistake
```

## When to Use It

| Good For | Bad For |
|----------|---------|
| High-dimensional data (text, genomics) | Very large datasets (slow to train) |
| Clear margin of separation | Lots of noise/overlapping classes |
| When number of features > number of samples | When you need probability outputs |
| Binary classification | When you need interpretability |

## Important: Feature Scaling Required

SVM is **sensitive to feature scale**. Always normalize your features:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Without scaling, a feature with range [0, 1000] will dominate a feature with range [0, 1].

## Key Hyperparameters

| Parameter | What It Does | How to Tune |
|-----------|-------------|-------------|
| `C` | Strictness of margin | Start at 1.0, try 0.1, 10, 100 |
| `kernel` | Shape of boundary | Try 'rbf' first, then 'linear' |
| `gamma` | How far each point's influence reaches (RBF) | 'scale' (default) is usually fine |

## Python Example

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# IMPORTANT: Scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use same scaler!

# Train
model = SVC(kernel='rbf', C=1.0)
model.fit(X_train_scaled, y_train)

# Evaluate
accuracy = model.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy:.2%}")
print(f"Support vectors: {model.n_support_}")  # how many key points
```

## SVM vs Other Algorithms

| vs Logistic Regression | SVM maximizes margin, LR maximizes likelihood. SVM often better with small datasets. |
|---|---|
| vs Random Forest | RF handles mixed features without scaling. SVM needs scaling but works better in high dimensions. |
| vs Neural Networks | NNs need lots of data. SVM works well with smaller datasets. |

## Key Takeaway

SVM is powerful for **high-dimensional, small-to-medium datasets** where a clear boundary exists. It's the go-to for text classification and genomics. Always scale your features, try RBF kernel first, and tune C. For large datasets or when you need interpretability, consider other options.
