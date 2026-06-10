# Random Forest

## TL;DR

A Random Forest builds hundreds of slightly different decision trees and lets them vote on the answer. Each tree sees a random subset of the data AND a random subset of features — so they make different mistakes. When they vote together, the errors cancel out and accuracy improves dramatically. It's the best algorithm to try first on any tabular dataset: works well out of the box, hard to overfit, and tells you which features matter most.

> 💡 **Key Insight:** One expert is fallible. A hundred diverse experts who each make different mistakes are remarkably reliable when averaged. That's the entire insight behind Random Forest.

---

## The Mental Model

Think of **jury selection and deliberation**.

A single judge might have biases or blindspots. A jury of 12 is more reliable because their individual biases tend to cancel out. Random Forest forces diversity (different data, different features per tree), so the "jury" is even better calibrated than a random 12 people.

Mapping:
- Each juror → one decision tree
- Jurors seeing different subsets of evidence → bagging (random data sampling)
- Jurors with different backgrounds → random feature selection
- Unanimous jury verdict → majority vote across all trees
- A biased judge → a single overfit decision tree
- Jury deliberation → the final prediction aggregation

The magic: you don't need each tree to be perfect. You need them to be diverse. Diversity is what cancels out individual errors.

---

## Build the Intuition From Zero

The claim "many mediocre trees beat one good tree" sounds like hand-waving. Let's prove it to yourself, and demystify the two tricks (**bagging** and **random features**) that force the trees to disagree.

### Idea 1: Why averaging many wrong-ish guesses lands near right

Picture a jar of jellybeans with 1,000 in it. Ask one person — they guess 1,400 (off by 400). Ask 100 people: some guess high, some low, but the **errors point in random directions, so they partly cancel** when you average. The crowd's average lands around 1,020. This is the real, measurable "wisdom of crowds."

```
one guesser:        1400         → error +400
100 guessers avg:   1020         → error +20   (the highs and lows cancelled)
```

The catch that makes or breaks it: **the guessers must make *independent* mistakes.** If all 100 people copied the same person, averaging does nothing — you just get that one wrong answer 100 times. So the entire engineering problem of Random Forest is: *how do we force 100 trees to make different, independent mistakes* instead of all making the same one? Two tricks.

### Idea 2: Trick #1 — bagging (give each tree a different slice of data)

**Bagging** = "bootstrap aggregating." Each tree is trained on a random sample of the rows, **drawn with replacement** — meaning after you pick a row you toss it back, so the same row can appear twice and others not at all:

```
Original data:  [A B C D E]

Tree 1 trains on:  [A A C D E]   ← A drawn twice, B missing
Tree 2 trains on:  [B C C D D]   ← different mix
Tree 3 trains on:  [A B B C E]   ← different again
```

Because each tree sees a slightly different dataset, each grows into a slightly different shape and makes different mistakes — exactly the independence we need. (Bonus: the rows left out of a tree's sample, the "out-of-bag" rows, act as a free built-in test set.)

### Idea 3: Trick #2 — random features (forbid trees from copying each other)

Bagging alone isn't enough: if one feature is super predictive (say `income`), *every* tree will grab it for its first split and they'll all look alike. So Random Forest adds a second rule: **at each split, a tree may only consider a random handful of features**, not all of them.

```
At one split, Tree 1 is only allowed to look at: {age, zip, balance}    → splits on balance
At the same kind of split, Tree 2 may look at:    {income, age, loans}  → splits on income
   → the trees are FORCED to explore different questions → genuinely different trees
```

> 💡 **Random Forest in one line:** build many decision trees, force each to be different by (1) training it on a random resample of the rows and (2) letting it consider only a random subset of features at each split — then average their votes. The forced diversity makes their errors independent, and independent errors cancel when you average. That's the whole algorithm.

This is also the key contrast with [gradient-boosting.md](gradient-boosting.md): Random Forest's trees are built **independently and in parallel** then averaged (a jury voting at once); boosting's trees are built **sequentially**, each fixing the last one's mistakes. The sections below formalize bagging, OOB error, and feature importance.

---

## Why It Exists

### The Problem with a Single Tree

Decision trees overfit. A small change in training data can produce a completely different tree. And a deep tree memorizes noise:

```
Single decision tree on loan data:
  Training accuracy: 100%  ← memorized
  Test accuracy:      72%  ← terrible generalization
```

### The Solution: Bagging + Random Features

Train many trees, each slightly different. Average their predictions:

```
100 trees on loan data:
  Each tree: ~85% test accuracy individually
  All 100 trees voting together: 93% test accuracy
  
Why? Each tree makes different mistakes.
When 100 trees vote, wrong votes get outnumbered by right votes.
```

### What Changed

Random Forest (introduced by Leo Breiman in 2001) made ensemble methods accessible without any hyperparameter tuning expertise. It's still the baseline algorithm for tabular data competitions and production systems.

---

## Core Concepts

### 1. Bagging (Bootstrap Aggregating)

**One-line definition:** Each tree is trained on a different random sample of the training data (with replacement).

**Analogy:** Polling. Instead of asking every voter once, you run 100 polls, each asking a different random sample of 1000 people. Combining all polls gives a more accurate picture than any single poll.

```
Original data: 1000 rows

Tree 1 trains on: rows [3, 7, 7, 22, 45, 3, 180, ...] ← some rows repeated, some skipped
Tree 2 trains on: rows [1, 99, 12, 12, 54, 7, 200, ...]  ← different sample
Tree 3 trains on: rows [45, 2, 78, 7, 3, 190, 54, ...]   ← another different sample

"With replacement" = the same row can appear multiple times in one sample
                     (like putting a marble back before drawing the next one)
```

The ~37% of rows each tree DOESN'T see become its built-in "out-of-bag" test set — free validation without a separate split!

**Common misconception:** Bagging alone is enough to build a good ensemble. If you use the same features for every split, trees are too similar and still correlated. You need random features too.

---

### 2. Random Feature Selection

**One-line definition:** At each node in each tree, only a random subset of features is considered for the split.

**Analogy:** In a group brainstorm, if everyone reads the same briefing document, everyone thinks of the same ideas. But if each person gets a different random selection of facts, you get more diverse and creative solutions.

```
Dataset has 20 features. At each tree node:
  Full decision tree: considers all 20 features → picks the best
  Random Forest node: considers only √20 ≈ 4 random features → picks best of those 4

Why? If one feature (e.g., income) is very predictive, ALL trees would use it
     at the root node → all trees look the same → voting doesn't help.
     
     Forcing random features makes trees diverse → errors are uncorrelated
     → averaging works.
```

**Common misconception:** Random feature selection hurts accuracy. Each individual tree is weaker, yes. But their combination is stronger because they're diverse. Weak + diverse beats strong + correlated.

---

### 3. Voting / Averaging

**One-line definition:** For classification, take the majority vote; for regression, take the average across all trees.

```
Classification example (predict loan default):
  Tree 1: Default   Tree 6: Default
  Tree 2: Default   Tree 7: No Default
  Tree 3: No Default Tree 8: Default
  Tree 4: Default   Tree 9: Default
  Tree 5: Default   Tree 10: No Default

  Vote: 7 Default vs 3 No Default → Final: "Default"
  Confidence: 70% (7 out of 10 trees agreed)

Regression example (predict house price):
  Tree 1: $230K,  Tree 2: $245K,  Tree 3: $225K,
  Tree 4: $250K,  Tree 5: $235K

  Average: $237K → Final prediction: $237,000
```

**Common misconception:** You need the trees to agree for Random Forest to work. Disagreement between trees is EXPECTED and HEALTHY. The diversity is the point.

---

### 4. Out-of-Bag (OOB) Score — Free Validation

**One-line definition:** Each tree is tested on the ~37% of data it never saw during training, giving a free estimate of generalization accuracy.

**Analogy:** If you take a random sample of 1000 voters, about 370 weren't sampled. You can use those 370 to check how well your poll predicts their opinions — for free.

```python
model = RandomForestClassifier(n_estimators=100, oob_score=True)
model.fit(X, y)

print(model.oob_score_)  # Free validation accuracy — no test set needed

# For each sample:
# Only the trees that DIDN'T see it vote on its prediction
# The average of those predictions = OOB estimate for that sample
```

**Common misconception:** OOB score replaces cross-validation. OOB is a quick, free estimate — good for development. For final model evaluation, use a proper held-out test set.

---

### 5. Feature Importance

**One-line definition:** A score per feature measuring how much it reduced Gini impurity across all splits in all trees.

**Analogy:** In a court case, some evidence is mentioned in every argument — that evidence is "important." Feature importance measures which data features appear in the most decision nodes and drive the biggest purity gains.

```python
model.feature_importances_
# [0.03, 0.42, 0.31, 0.12, 0.08, 0.04] — 6 features
# Feature 2 most important (42% of impurity reduction attributed to it)

# More reliable than single-tree importance because it's averaged over 100 trees
```

**Common misconception:** Feature importance from Random Forest tells you the causal relationship. It tells you correlation — which features the model USED, not which features CAUSE the outcome. Causality requires controlled experiments.

---

## How It Actually Works (Step-by-Step)

```
Training phase:

Step 1: Decide: 100 trees (n_estimators=100)

Step 2: For each of 100 trees:
  a) Sample 1000 rows with replacement from 1000 training examples
     → Some rows appear 2-3 times, ~370 rows not sampled (OOB)
  b) Build a full decision tree, but at EACH node:
     → Randomly select √features (e.g., 4 of 20)
     → Find best split among only those 4 features
  c) Grow tree deep (no depth limit — each tree is intentionally high variance)

Step 3: Store all 100 trees

Prediction phase:

Step 4: New data point arrives

Step 5: Run it through all 100 trees simultaneously
  Tree 1: "Class A",  Tree 2: "Class B",  Tree 3: "Class A", ...

Step 6: Count votes
  Class A: 67 votes, Class B: 33 votes

Step 7: Predict Class A (confidence: 67%)
```

---

## Code in Practice

### 1. Hello World — Basic Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Start with 100 trees and default settings — often works great
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Test accuracy: {model.score(X_test, y_test):.2%}")
```

### 2. Practical — With OOB Score and Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,       # Free validation estimate
    n_jobs=-1,            # Use all CPU cores
    random_state=42
)
model.fit(X_train, y_train)

print(f"OOB accuracy:  {model.oob_score_:.2%}")   # Free estimate
print(f"Test accuracy: {model.score(X_test, y_test):.2%}")  # True evaluation

# Top 5 most important features
pairs = sorted(
    zip(data.feature_names, model.feature_importances_),
    key=lambda x: x[1], reverse=True
)
for name, score in pairs[:5]:
    print(f"  {name}: {score:.2%}")
```

### 3. Real-World Pattern — Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 500),    # Number of trees
    'max_depth': [None, 10, 20, 30],      # Tree depth
    'max_features': ['sqrt', 'log2'],     # Features per node
    'min_samples_leaf': randint(1, 10),   # Min samples per leaf
}

# Try 20 random combinations (faster than grid search)
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best CV F1: {search.best_score_:.3f}")
```

---

## Gotchas & Pitfalls

```
❌ Using n_estimators=10 (the old scikit-learn default)
   Too few trees → high variance, poor performance
✅ Start with n_estimators=100; increase to 300-500 for important tasks

❌ Not setting n_jobs=-1
   Random Forest training is embarrassingly parallel but uses 1 CPU by default
✅ Always set n_jobs=-1 to use all cores — 4-8× faster training

❌ Trusting feature importance for correlated features
   If two features are highly correlated, the forest splits their importance
   e.g., "height in cm" and "height in inches" both get ~50% of the true importance
✅ For correlated features, use permutation importance instead

❌ Using Random Forest for non-tabular data
   It can't learn spatial patterns in images or sequential patterns in text
✅ Random Forest is specifically for tabular (spreadsheet-style) data

❌ No train/test split because "OOB score is enough"
   OOB is an approximation. Always use a proper held-out test set for final evaluation
✅ Use train_test_split for final numbers; OOB for fast iterations during development

❌ Not scaling features... wait, you don't need to for Random Forest
   Trees use thresholds, not distances — feature scale is irrelevant
✅ No StandardScaler needed for Random Forest (unlike SVM, KNN, logistic regression)
```

---

## When to Use / When NOT to Use

### Use Random Forest When:
- Tabular data (rows = observations, columns = features)
- You don't want to tune much — it works well with defaults
- You need feature importance to understand your data
- You want an ensemble model without the complexity of Gradient Boosting

### Don't Use Random Forest When:
- You need to explain individual predictions (use a single decision tree)
- You need the absolute best accuracy on tabular data (use Gradient Boosting / XGBoost)
- You're working with images, audio, or text (use deep learning)
- You have millions of rows and speed is critical (LightGBM is faster)

---

## Related Concepts

| Concept | Connection |
|---------|------------|
| Decision Trees | Random Forest is an ensemble of decision trees — understand trees first |
| Gradient Boosting | The other major tree ensemble — sequential (corrects errors) vs parallel (averages) |
| Bagging | The sampling strategy Random Forest uses — Random Forest adds feature randomness on top |
| Feature Importance | Random Forest provides this reliably; single decision trees provide noisy estimates |
| Cross-Validation | Use with Random Forest for robust accuracy estimates across multiple data splits |

---

## Cheat Sheet

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model = RandomForestClassifier(
    n_estimators=100,     # Start here, increase if budget allows
    max_features='sqrt',  # Default: √n_features per split (classification)
    max_depth=None,       # Let trees grow fully — bagging handles overfitting
    oob_score=True,       # Free validation estimate
    n_jobs=-1,            # Use all CPU cores
    random_state=42       # Reproducibility
)

Key attributes after fitting:
  model.feature_importances_    # Which features matter most (0-1 sum=1)
  model.oob_score_              # Out-of-bag validation accuracy
  model.estimators_             # List of all individual decision trees

Classification: majority vote across trees
Regression:     average across trees (RandomForestRegressor)

Remember:
  1. Start with 100 trees; skip feature scaling — trees don't need preprocessing
  2. Set n_jobs=-1 always — it's parallel by nature
  3. Use OOB score for fast dev iteration, proper test set for final eval
```

---

## Self-Check Questions

<details>
<summary>Click to reveal answers</summary>

**Q1: Why does a Random Forest generalize better than a single decision tree?**
A single tree overfits by memorizing the training data. Each Random Forest tree memorizes *different* noise (due to random data and feature sampling). When they vote, correct predictions agree while incorrect predictions (based on different noise) cancel out. The result is a model that captures real patterns, not noise.

**Q2: What is "bagging" and what problem does it solve?**
Bagging (Bootstrap Aggregating) trains each tree on a different random sample of the data (drawn with replacement). It solves the variance problem of single decision trees — because each tree sees different data, they make different errors, and averaging reduces those errors.

**Q3: Why does Random Forest use a random subset of features at each split?**
Without it, all trees would use the same strong features at their root nodes — they'd be nearly identical, and averaging them wouldn't help. Random feature selection forces diversity: each tree develops its own "view" of the data, so their errors are uncorrelated and averaging works.

**Q4: What is the OOB score and is it a replacement for a test set?**
The OOB score estimates generalization accuracy using the ~37% of training samples each tree didn't see. It's a free, quick estimate — good for development iterations. It's NOT a replacement for a proper held-out test set for final evaluation, because the OOB estimate has higher variance.

**Q5: A client asks you to explain why a specific customer was denied a loan. You used Random Forest. What's the problem?**
Random Forest averages 100 decision trees — you can't trace any individual prediction through 100 trees simultaneously. For explainability, you'd need: (1) a single decision tree, (2) SHAP values (a post-hoc explanation method that works with Random Forest), or (3) LIME. The model is accurate but not natively interpretable at the individual prediction level.

</details>

---

## Go Deeper

| Resource | Why It's Worth Your Time |
|----------|--------------------------|
| [StatQuest: Random Forest](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) | Best visual explanation of bagging and why Random Forest beats single trees. 16 minutes. |
| [scikit-learn Random Forest docs](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees) | Official docs with parameter explanations and comparison to other ensembles. |
| [Permutation Importance (eli5)](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html) | Better than built-in feature importances for correlated features. Essential for real projects. |
| *Hands-On ML* Ch. 7 — Ensemble Learning — Aurélien Géron | Best textbook coverage of bagging, Random Forest, and boosting. Covers theory and practice together. |
| [Kaggle: Feature Importance Guide](https://www.kaggle.com/code/dansbecker/permutation-importance) | Hands-on tutorial on interpreting Random Forest feature importances correctly. |
