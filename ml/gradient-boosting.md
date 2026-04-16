# Gradient Boosting

## What Is It?

Gradient Boosting builds trees **one at a time**, where each new tree **fixes the mistakes** of the previous ones. Instead of many independent trees voting (Random Forest), it's a chain of specialists — each one learning from what the team got wrong.

Think of it like editing an essay: first draft → editor fixes grammar → another editor fixes flow → another fixes facts. Each pass improves on the last.

## Real-World Examples

- **Kaggle competitions** — XGBoost/LightGBM win most tabular data competitions
- **Credit scoring** — bank loan approval decisions
- **Search ranking** — Google uses gradient boosted trees for ranking
- **Ad click prediction** — will this user click this ad?
- **Fraud detection** — real-time transaction scoring

## How It Works (Step by Step)

### 1. Start with a Simple Prediction

Make a baseline prediction (e.g., the average of all values for regression).

```
Actual:     [100, 200, 150, 300]
Prediction: [187, 187, 187, 187]   ← just the average
Errors:     [-87,  13, -37, 113]   ← residuals (what we got wrong)
```

### 2. Train a Tree on the Errors

Build a small tree that predicts the **errors** (residuals), not the original values.

```
Tree 1 learns: "These residuals follow a pattern"
Tree 1 predicts residuals: [-80, 10, -30, 100]
```

### 3. Update Predictions

Add the tree's corrections (scaled by learning rate):

```
New prediction = old prediction + learning_rate × tree's correction
               = 187 + 0.1 × [-80, 10, -30, 100]
               = [179, 188, 184, 197]

New errors: [-79, 12, -34, 103]   ← still wrong, but less wrong
```

### 4. Repeat: Train Another Tree on the Remaining Errors

```
Tree 2 learns the new residuals
Tree 3 learns the remaining residuals
...
Tree 100 learns the tiny remaining residuals

Final prediction = baseline + 0.1×tree1 + 0.1×tree2 + ... + 0.1×tree100
```

Each tree makes a **small correction**. Together, they add up to accurate predictions.

## Random Forest vs Gradient Boosting

| Random Forest | Gradient Boosting |
|---------------|-------------------|
| Trees built **independently** in parallel | Trees built **sequentially**, each fixing errors |
| Each tree is deep and complex | Each tree is shallow and simple |
| Averages many opinions | Adds up small corrections |
| Hard to overfit | Easy to overfit (needs tuning) |
| Less tuning needed | More tuning, but higher potential accuracy |
| Faster to train (parallelizable) | Slower to train (sequential) |

## The Big Three Libraries

### XGBoost (eXtreme Gradient Boosting)
- The Kaggle champion. Fast, accurate, well-optimized.
- Handles missing values automatically.

### LightGBM (Microsoft)
- Even faster than XGBoost on large datasets.
- Grows trees **leaf-wise** instead of level-wise (faster convergence).

### CatBoost (Yandex)
- Best at handling **categorical features** (no encoding needed).
- Less overfitting out of the box.

```python
# All three have similar APIs:
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
```

## Key Hyperparameters

| Parameter | What It Does | How to Tune |
|-----------|-------------|-------------|
| `n_estimators` | Number of trees | 100-1000 (use early stopping) |
| `learning_rate` | How much each tree contributes | 0.01-0.3 (lower = more trees needed) |
| `max_depth` | How deep each tree grows | 3-8 (shallow trees!) |
| `subsample` | Fraction of data per tree | 0.7-0.9 |
| `colsample_bytree` | Fraction of features per tree | 0.7-0.9 |

**The golden rule**: `learning_rate` and `n_estimators` are linked.
- Lower learning rate → need more trees → better results but slower
- Start with `learning_rate=0.1`, `n_estimators=100`, then decrease rate and increase trees

## Early Stopping (Critical Technique)

Don't guess how many trees — let the model decide:

```python
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,  # stop if no improvement for 10 rounds
)
```

This prevents overfitting and saves training time.

## When to Use It

| Good For | Bad For |
|----------|---------|
| Tabular/structured data | Images, audio, text (use deep learning) |
| When you need maximum accuracy | When training speed matters most |
| Kaggle competitions | When you need interpretability |
| Mixed feature types | Very small datasets (prone to overfit) |

## Python Example

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Further split for early stopping
X_train2, X_val, y_train2, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train with early stopping
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss',
)
model.fit(
    X_train2, y_train2,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
print(f"Trees used: {model.best_iteration + 1}")

# Feature importance
top_features = sorted(
    zip(data.feature_names, model.feature_importances_),
    key=lambda x: x[1], reverse=True
)[:5]
for name, importance in top_features:
    print(f"  {name}: {importance:.3f}")
```

## Key Takeaway

Gradient Boosting (XGBoost/LightGBM) is the **most powerful algorithm for tabular data**. It consistently wins competitions and powers production systems at scale. The tradeoff: it needs more tuning than Random Forest. Always use early stopping, start with a reasonable learning rate, and tune from there. If you're working with spreadsheet-style data and need the best accuracy, this is your algorithm.
