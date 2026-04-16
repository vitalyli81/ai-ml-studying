# Random Forest

## What Is It?

A random forest builds **many decision trees** and lets them **vote** on the answer. Each tree is slightly different, and the combined wisdom of many "okay" trees beats one single tree.

Think of it like asking 100 people a question instead of one expert — the crowd's average answer is usually better.

## Real-World Examples

- **Credit scoring** — will this person default on a loan?
- **Medical diagnosis** — does this patient have the disease?
- **Customer churn** — which customers are about to leave?
- **Feature importance** — which factors matter most for the outcome?

## How It Works (Step by Step)

### 1. Create Many Different Trees

For each tree:

1. **Random sampling (Bagging)** — take a random subset of your data (with replacement). Each tree sees different data.
2. **Random features** — at each split, only consider a random subset of features. Each tree uses different features.
3. **Grow the tree** — let it grow deep (no pruning needed).

### 2. Combine the Predictions

- **Classification** → each tree votes, **majority wins**
- **Regression** → each tree predicts a number, take the **average**

```
Tree 1: "spam"     Tree 4: "spam"     Tree 7: "not spam"
Tree 2: "spam"     Tree 5: "spam"     Tree 8: "spam"
Tree 3: "not spam" Tree 6: "spam"     Tree 9: "spam"
                                       Tree 10: "not spam"

Final vote: 7 spam vs 3 not spam → "SPAM"
```

### 3. Why Randomness Helps

Without randomness, all trees would be identical → no benefit from voting.

- **Random data** → each tree overfits to different noise → errors cancel out
- **Random features** → trees make different mistakes → diverse perspectives

It's like forming a jury: you want people with different viewpoints, not 12 clones of the same person.

## Single Tree vs Random Forest

| Single Decision Tree | Random Forest |
|---------------------|---------------|
| Fast to train | Slower (training many trees) |
| Easy to overfit | Resistant to overfitting |
| Unstable (small data change → different tree) | Stable predictions |
| Easy to visualize | Hard to visualize (100+ trees) |
| Lower accuracy | Higher accuracy |

## Key Hyperparameters

| Parameter | What It Does | Typical Values |
|-----------|-------------|----------------|
| `n_estimators` | Number of trees | 100-500 (more = better but slower) |
| `max_depth` | How deep each tree grows | None (let them grow) or 10-30 |
| `max_features` | Features considered per split | "sqrt" (classification) or "log2" |
| `min_samples_leaf` | Minimum samples in a leaf | 1-5 |

**Rule of thumb**: Start with 100 trees and default settings. Increase trees until accuracy plateaus.

## When to Use It

| Good For | Bad For |
|----------|---------|
| Almost any tabular dataset | Very large datasets (slow to train) |
| When you don't want to tune much | When you need a very fast model |
| Feature importance analysis | When you need to explain individual decisions |
| Both classification and regression | Real-time predictions with tight latency |

## Out-of-Bag (OOB) Score — Free Validation

Since each tree only sees ~63% of the data (due to random sampling), the remaining ~37% acts as a **built-in test set**. This is the OOB score — you get validation accuracy without needing a separate test set.

```python
model = RandomForestClassifier(n_estimators=100, oob_score=True)
model.fit(X, y)
print(f"OOB accuracy: {model.oob_score_:.2%}")
```

## Feature Importance

One of the best reasons to use Random Forest — it tells you **which features matter**:

```python
importances = model.feature_importances_
# Returns: [0.42, 0.31, 0.15, 0.12]
# Meaning: feature 1 is most important (42%), feature 4 least (12%)
```

How it works: features that appear in more splits and reduce impurity the most get higher importance scores.

## Python Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Train — just set n_estimators and go
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")

# Top 3 most important features
feature_importance = sorted(
    zip(wine.feature_names, model.feature_importances_),
    key=lambda x: x[1], reverse=True
)
for name, score in feature_importance[:3]:
    print(f"  {name}: {score:.2%}")
```

## Key Takeaway

Random Forest is the **Swiss Army knife of ML**. It works well on almost any tabular dataset with minimal tuning. If you're not sure which algorithm to use, start here. It's hard to go wrong with a Random Forest.
