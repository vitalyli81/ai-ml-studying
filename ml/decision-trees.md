# Decision Trees

## TL;DR

A decision tree makes predictions by asking a series of yes/no questions — like a flowchart. Each question splits data into cleaner groups, until a final prediction is reached. The algorithm figures out the best questions automatically from data. It's the most interpretable ML model: you can read every decision it makes. But a single tree overfits easily — in practice, use Random Forest or Gradient Boosting (ensembles of many trees).

> 💡 **Key Insight:** A decision tree is just nested if/else logic that the algorithm learned from data, not rules you coded by hand.

---

## The Mental Model

Think of the game **"20 Questions"** — or a doctor using a diagnostic flowchart.

A doctor diagnoses pneumonia by asking: "Fever? → Cough? → Chest pain? → Low oxygen?" Each yes/no narrows down possibilities until a diagnosis is reached.

Mapping:
- Each question in the flowchart → a node in the tree
- "Does the patient have a fever?" → "Is feature X > threshold?"
- The doctor's final diagnosis → the leaf node (prediction)
- The flowchart itself → the decision tree model
- The doctor figuring out the best question order → the training algorithm (Gini/entropy)

The key difference: the algorithm decides what questions to ask and in what order, by finding the questions that most cleanly separate the classes.

> 💻 **Frontend bridge:** a decision tree is the nested `if/else` routing logic you'd write by hand — except *generated from data*, like a compiler emitting the conditionals for you. When you call `export_text(model)`, you are literally reading auto-generated code. Training = the machine figuring out which `if` to put at the top so the fewest branches are needed.

---

## Build the Intuition From Zero

The tree-as-flowchart idea is easy. The part that feels like magic is: **how does the algorithm know which question to ask first?** Nobody told it "ask about income before credit score." Let's derive that decision by hand — it's just "which question sorts the marbles cleanest."

### Idea 1: A good question separates the classes; a bad one doesn't

You have 10 loan applicants — 5 repaid (✓), 5 defaulted (✗) — all jumbled together. You can ask **one** yes/no question to split them. Which question is better?

```
Question A: "Income > $50k?"          Question B: "Owns a red car?"
   YES → ✓✓✓✓✗   NO → ✗✗✗✗✓             YES → ✓✓✗✗✗   NO → ✓✓✓✗✗
        (mostly repaid)(mostly default)        (still a mess)(still a mess)
   → cleanly sorted! GOOD QUESTION         → barely changed anything. USELESS
```

Question A is great because each resulting group is now *mostly one class* — you've made progress. Question B left both groups as mixed as before. **The tree's entire training job is to try every possible question and keep the one that sorts cleanest.** It's that simple — the only thing missing is a number to measure "cleanness."

### Idea 2: Gini impurity is the "messiness score"

We need to turn "how mixed is this group?" into a number so the computer can compare questions. That number is **Gini impurity**, and it has one job: **0 means pure (all one class), bigger means more mixed.**

Here's the intuition behind the formula `Gini = 1 − (p₁² + p₂² + …)`, where `p` is each class's fraction of the group:

```
all one class:   fractions (1.0, 0.0)  → 1 − (1.0² + 0²)     = 1 − 1.0  = 0.0  ✓ pure
50/50 mix:       fractions (0.5, 0.5)  → 1 − (0.5² + 0.5²)   = 1 − 0.5  = 0.5  ✗ messiest
80/20 mix:       fractions (0.8, 0.2)  → 1 − (0.64 + 0.04)   = 1 − 0.68 = 0.32 (in between)
```

Why squaring? Squaring a fraction rewards *dominance*. When one class dominates (0.8), its square (0.64) is large, so `1 − that` is small → low messiness. When it's an even mix, no class dominates, the squares stay small, and the messiness score stays high. The formula is just a clean way to say "reward groups that are lopsided toward one class."

### Idea 3: Pick the question with the biggest drop in messiness

Now combine the two ideas. To score a question, compute the messiness *before* the split, then the (weighted) messiness *after*, and take the **drop**. That drop is called **information gain** — bigger drop = better question:

```
Before:  10 applicants, 5✓/5✗   → Gini = 0.5  (maximally messy)

Question A splits into two near-pure groups → average Gini after ≈ 0.32
   drop = 0.5 − 0.32 = 0.18   ← big improvement → tree picks this

Question B splits into two still-messy groups → average Gini after ≈ 0.48
   drop = 0.5 − 0.48 = 0.02   ← barely helped → tree rejects this
```

> 💡 **The whole training algorithm in one sentence:** at each node, try every feature and every threshold, compute the drop in Gini impurity for each, and pick the split with the biggest drop. Then repeat inside each child group until the groups are pure (or you hit a depth limit). The "20 questions" order isn't designed by a human — it falls out of always asking the most-clarifying question next.

The sections below formalize Gini, entropy (a near-identical alternative), and the overfitting that happens when you let the tree keep asking questions until every leaf is perfectly pure.

---

## Why It Exists

### The Problem Before

Rules-based systems required domain experts to manually write all the if/else logic. This was slow, brittle, and couldn't scale to many features:

```javascript
// Hand-written rules — fragile, hard to maintain:
if (income > 50000) {
  if (creditScore > 700 && debtRatio < 0.4) return "APPROVE";
  else return "REVIEW";
} else {
  // ... 50 more conditions someone had to think up
}
```

### The Solution

Let the data decide the rules. Train on thousands of loan applications where you know the outcome — the algorithm finds the most useful splits automatically.

### What Changed

Decision trees made ML interpretable. Unlike "black box" models, every decision tree prediction can be traced back through a chain of human-readable rules. They also became the building block for the two most powerful tabular ML algorithms: Random Forest and Gradient Boosting.

---

## Core Concepts

### 1. Nodes and Leaves

**One-line definition:** Nodes are questions; leaves are answers.

**Analogy:** In a flowchart: diamonds (decisions) are nodes, rectangles (outcomes) are leaves.

```
                [Is income > 50k?]        ← root node (first question)
                /                \
             Yes                  No
              |                    |
  [Credit score > 700?]    [Has collateral?]   ← internal nodes
       /          \             /          \
     Yes          No          Yes          No
      |            |           |            |
   APPROVE      REVIEW      REVIEW       DENY    ← leaf nodes (predictions)
```

**Common misconception:** Deeper trees are better. Deeper trees memorize training data (overfit). A shallow tree (depth 3-5) often generalizes better.

---

### 2. Gini Impurity — Choosing the Best Split

**One-line definition:** A score that measures how "mixed" a group is — 0 means perfectly pure (one class only), higher values mean more mixed.

**Analogy:** Imagine sorting colored marbles into buckets. Gini impurity measures how mixed each bucket is. A bucket with all red marbles (Gini = 0) is perfect. A bucket with an even split of classes is maximally impure.

```
Formula: Gini = 1 - (p₁² + p₂² + ...)
  where p₁, p₂... = fraction of each class in the group

Max Gini depends on the number of classes K:
  2 classes → max Gini = 0.5   (50/50 split)
  3 classes → max Gini ≈ 0.67  (even thirds)
  K classes → max Gini = 1 - 1/K

Example:
  Group A: 10 spam, 0 not-spam
  Gini A = 1 - (1² + 0²) = 0       ← perfectly pure ✓

  Group B: 5 spam, 5 not-spam
  Gini B = 1 - (0.5² + 0.5²) = 0.5 ← maximally mixed for 2 classes ✗

The tree picks the split that produces the lowest weighted Gini
across both resulting groups.
```

**Common misconception:** You need to understand the math to use decision trees. No — scikit-learn handles all of this. Understanding the concept (lower = purer = better) is enough.

---

### 3. Information Gain / Entropy

**One-line definition:** An alternative to Gini — measures how much a split reduces uncertainty (disorder) in the data.

**Analogy:** Entropy is borrowed from physics — the "disorder" of a system. A sorted deck of cards has low entropy. A shuffled deck has high entropy. A good split turns a disordered group into two more ordered ones.

```
Entropy = -Σ p × log₂(p)

  All one class:    entropy = 0   (no disorder)
  50/50 split:      entropy = 1   (maximum disorder)

Information Gain = entropy_before - weighted_entropy_after_split

Higher gain = better split
```

**Common misconception:** Gini and entropy are very different. In practice, they give almost identical trees. Gini is faster to compute (no log), so it's the scikit-learn default. Don't overthink the choice.

---

### 4. Overfitting in Decision Trees

**One-line definition:** A tree that's too deep memorizes the training data exactly but fails on new data.

**Analogy:** A student who memorizes all past exam questions word-for-word but can't handle any new question. They "learned" the noise, not the pattern.

```
Overfit tree (too deep):
  "If age=27 AND salary=52,341 AND zip=90210 → APPROVE"
  → Only works for that exact person. Useless for others.

Good tree (constrained depth):
  "If income > 50K AND credit_score > 700 → APPROVE"
  → Generalizes to any person matching this pattern.
```

**Prevention hyperparameters:**

| Parameter | What It Limits |
|-----------|---------------|
| `max_depth` | Tree can grow at most this many levels deep |
| `min_samples_split` | A node must have at least N samples to split |
| `min_samples_leaf` | Each leaf must have at least N samples |
| `max_features` | Only consider a subset of features per split |

**Common misconception:** More data fixes overfitting in decision trees. Not automatically — without constraining the tree, it will still memorize the data, just with more examples.

---

### 5. Feature Importance

**One-line definition:** A score for each feature measuring how much it contributed to making pure splits across all nodes.

**Analogy:** In 20 Questions, the best first question (e.g., "Is it a living thing?") eliminates half the possibilities. Feature importance measures which features act like that best first question.

```python
model.feature_importances_
# Returns: [0.04, 0.53, 0.39, 0.04]
# feature 2 (petal_length) is most important (53% of splits used it)
```

**Common misconception:** Feature importance from a single tree is reliable. Single-tree importances are noisy. Importances from Random Forest (many trees) are much more stable.

---

> 🧠 **Quick recall — answer out loud before scrolling on** (all answers are above):
> 1. Gini of a group with 10 spam / 0 ham? With 5 / 5?
> 2. In one sentence: how does the tree choose which question to ask at a node?
> 3. Gini vs entropy — what's the practical difference?
> 4. Name the first two hyperparameters you'd set to stop a tree from overfitting.
> 5. Why is a single tree's feature importance unreliable?

---

## How It Actually Works (Step-by-Step)

```
Dataset: predict if a loan will default

  Income  | CreditScore | Defaulted?
  ─────────────────────────────────
  30K     | 600         | Yes
  60K     | 750         | No
  45K     | 680         | No
  25K     | 550         | Yes
  80K     | 800         | No
  35K     | 650         | Yes

Step 1: Start at root — find the best split for ALL data
  Try: Income > 40K
    Left (<=40K): [Yes, Yes, Yes] → Gini = 0.0 (pure! all default)
    Right (>40K): [No, No, No]   → Gini = 0.0 (pure! none default)
  → This is a perfect split! Gini gain = maximum

Step 2: Build tree
  [Income > 40K?]
      /        \
    ≤40K       >40K
     |            |
   Default    No Default    ← both leaves are pure, done!

Step 3: Evaluate on new data
  New: Income=55K → Right branch → "No Default"
  New: Income=28K → Left branch → "Default"
```

Real datasets are never this clean — the algorithm tries every possible threshold for every feature and picks the one that reduces Gini impurity the most.

---

## Code in Practice

### 1. Hello World — Iris Classification

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

# Limit depth to prevent overfitting
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Read the actual tree (this is the ML model — readable!)
print(export_text(model, feature_names=list(iris.feature_names)))
```

### 2. Practical — Train/Test with Depth Tuning

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Try different depths to find the sweet spot
for depth in [1, 2, 3, 5, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Depth {str(depth):5}: train={train_acc:.2%}, test={test_acc:.2%}")

# Depth=None: train=100%, test=93% ← overfitting (big gap)
# Depth=3:    train=98%,  test=97% ← sweet spot (small gap)
```

### 3. Real-World Pattern — Feature Importance + Visualization

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

data = load_breast_cancer()
model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
model.fit(data.data, data.target)

# Feature importance
importances = sorted(
    zip(data.feature_names, model.feature_importances_),
    key=lambda x: x[1], reverse=True
)
print("Top 5 features:")
for name, score in importances[:5]:
    print(f"  {name}: {score:.3f}")

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=data.feature_names,
          class_names=data.target_names, filled=True)
plt.savefig("decision_tree.png", dpi=100, bbox_inches="tight")
```

---

## Gotchas & Pitfalls

```
❌ Letting the tree grow without constraints
   It will achieve 100% training accuracy but fail miserably on new data
✅ Always set max_depth=3-5 or min_samples_leaf=5 as a starting point

❌ Not splitting into train and test sets
   You'll never see the overfitting if you measure accuracy on training data
✅ Always evaluate on a held-out test set; watch for large train/test accuracy gap

❌ Using a single decision tree in production
   It's unstable — small changes in data can produce a completely different tree
✅ Use Random Forest or Gradient Boosting for production; single trees are for exploration

❌ Not scaling features... wait, actually you don't need to for decision trees
   Unlike linear models, trees only care about whether a feature is > a threshold
✅ Decision trees are scale-invariant — no StandardScaler needed

❌ Treating feature importance from a single tree as ground truth
   Single-tree importances are noisy and biased toward high-cardinality features
✅ Use Random Forest importances (averaged over 100+ trees) for reliability

❌ Ignoring class imbalance
   A tree might just predict the majority class to minimize impurity
✅ Use class_weight='balanced' parameter for imbalanced datasets
```

---

## When to Use / When NOT to Use

### Use Decision Trees When:
- You need to explain every single prediction to a human (compliance, medicine)
- Quick exploration: "what splits in the data explain the outcome?"
- You want to visualize and understand the rules the model learned
- You have mixed numeric + categorical features (trees handle both natively)

### Don't Use Decision Trees When:
- You need the best possible accuracy (use Random Forest or Gradient Boosting)
- Your data has many features (trees become unstable and overfit)
- Predictions must be stable across slightly different datasets
- You're doing regression with smooth continuous outputs (use linear regression)

---

## Related Concepts

| Concept | Connection |
|---------|------------|
| Random Forest | An ensemble of hundreds of decision trees — fixes overfitting by averaging |
| Gradient Boosting | Sequences of small decision trees that each correct the previous one's errors |
| Gini / Entropy | The "score" that tells each tree node which question to ask |
| Feature Engineering | Good features make trees split more cleanly (lower impurity faster) |
| Feature Importance | Decision trees provide this for free — used by Random Forest too |

---

## Cheat Sheet

```python
from sklearn.tree import DecisionTreeClassifier, export_text

model = DecisionTreeClassifier(
    max_depth=4,          # Limits depth → prevents overfitting
    min_samples_leaf=5,   # Each leaf needs 5+ samples
    criterion='gini',     # 'gini' (default) or 'entropy'
    class_weight='balanced',  # For imbalanced classes
    random_state=42
)
model.fit(X_train, y_train)

model.feature_importances_      # Which features matter most
export_text(model, feature_names=[...])  # Read the actual tree

How the algorithm picks splits:
  For every feature, try every threshold
  Pick the split that gives the lowest weighted Gini impurity
  Repeat at each child node until stopping criteria met

Remember:
  1. Single tree = interpretable but unstable → use for exploration
  2. max_depth=3-5 is the first thing to tune
  3. For production accuracy, use Random Forest or Gradient Boosting
```

---

## Self-Check Questions

<details>
<summary>Click to reveal answers</summary>

**Q1: Why does a decision tree without any constraints achieve 100% training accuracy?**
Without constraints, the tree keeps splitting until every leaf contains exactly one training example (or all examples in that leaf are the same class). It memorizes every data point. This is perfect overfitting — it fails on any new data.

**Q2: What does Gini impurity of 0 mean, and what does 0.5 mean?**
Gini = 0 means the node is perfectly pure — all examples belong to the same class. Gini = 0.5 is the maximum for a 2-class problem — exactly 50% class 0, 50% class 1 (for K classes the max is 1 − 1/K). The tree tries to find splits that move groups toward Gini = 0.

**Q3: Why doesn't a decision tree need feature scaling?**
Decision trees make decisions based on thresholds: "Is feature X > some value?" The absolute scale doesn't matter — whether income is in dollars ($50,000) or thousands ($50), the tree finds the right threshold. Unlike distance-based algorithms (KNN, SVM), the scale is irrelevant.

**Q4: How do you detect overfitting in a decision tree?**
Compare training accuracy to test accuracy. A large gap (e.g., train=100%, test=70%) signals overfitting — the model memorized training data but doesn't generalize. Reduce max_depth or increase min_samples_leaf to fix it.

**Q5: When would you prefer a single decision tree over Random Forest?**
When you need to explain every prediction to a human — e.g., "we denied your loan because income < $40K AND credit score < 700." Random Forest can't do this (it's an average of 100 trees). A single tree is your only interpretable option at the cost of lower accuracy.

</details>

---

## Go Deeper

| Resource | Why It's Worth Your Time |
|----------|--------------------------|
| [StatQuest: Decision Trees](https://www.youtube.com/watch?v=_L39rN6gz7Y) | Visual explanation of Gini impurity and tree building. The clearest 20-minute intro available. |
| [scikit-learn Decision Tree docs](https://scikit-learn.org/stable/modules/tree.html) | Official docs with great visualizations using plot_tree(). Covers both classification and regression trees. |
| [Visualizing Decision Trees](https://mljar.com/blog/visualize-decision-tree/) | Shows multiple ways to visualize and interpret decision trees — critical for using them in practice. |
| *Hands-On ML* Ch. 6 — Aurélien Géron | Best book chapter on decision trees. Covers CART algorithm, regularization, and regression trees in one place. |
| [Kaggle: Decision Trees Tutorial](https://www.kaggle.com/learn/intro-to-machine-learning) | Kaggle's intro course uses decision trees as the starting point. Hands-on and practical. |
