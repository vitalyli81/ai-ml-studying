# Decision Trees

## What Is It?

A decision tree makes predictions by asking a series of **yes/no questions** about the data, like a flowchart. Each question splits the data into smaller groups until you reach an answer.

Think of it like the game "20 Questions" — each question narrows down the possibilities.

## Real-World Examples

- Should I **approve this loan**? (income > 50k? → credit score > 700? → ...)
- What **species** is this flower? (petal length > 2.5cm? → petal width > 1.8cm? → ...)
- Will a customer **buy this product**? (age > 30? → visited site > 3 times? → ...)

## How It Works (Step by Step)

### 1. The Structure

```
                [Is income > 50k?]
                /                \
             Yes                  No
              |                    |
    [Credit score > 700?]    [Has collateral?]
       /          \             /          \
     Yes          No          Yes          No
      |            |           |            |
   APPROVE      REVIEW      REVIEW       DENY
```

- **Root node** — the first question (top)
- **Internal nodes** — questions in the middle
- **Leaf nodes** — final predictions (bottom)
- **Branches** — yes/no paths

### 2. How Does It Pick the Best Question?

The tree tries every possible question and picks the one that **splits the data most cleanly** — separating classes as purely as possible.

Two common measures:

**Gini Impurity** (used by default in scikit-learn):
```
Gini = 1 - (p1² + p2²)
```
- Gini = 0 → perfectly pure (all same class) 
- Gini = 0.5 → maximum impurity (50/50 split)

**Information Gain / Entropy**:
```
Entropy = -p1*log(p1) - p2*log(p2)
```
- Entropy = 0 → perfectly pure
- Entropy = 1 → maximum disorder

In practice, both give similar results. Don't overthink the choice.

### 3. When Does It Stop Splitting?

The tree stops when:
- A node is **pure** (all same class)
- It reaches the **max depth** you set
- A node has **too few samples** to split further
- Further splitting doesn't improve enough

## Frontend Analogy

A decision tree is like **nested if/else statements**:

```javascript
function shouldApproveLoan(applicant) {
  if (applicant.income > 50000) {
    if (applicant.creditScore > 700) {
      return "APPROVE";
    } else {
      return "REVIEW";
    }
  } else {
    if (applicant.hasCollateral) {
      return "REVIEW";
    } else {
      return "DENY";
    }
  }
}
```

That's literally what a decision tree learns — but it figures out the conditions and thresholds automatically from the data.

## When to Use It

| Good For | Bad For |
|----------|---------|
| When you need to **explain** decisions | High-dimensional data (many features) |
| Mixed feature types (numbers + categories) | When you need the highest accuracy |
| Quick prototyping and exploration | Stable predictions (small data changes → different tree) |
| Understanding which features matter most | Smooth, continuous predictions |

## The Big Problem: Overfitting

A decision tree will happily memorize your training data if you let it grow deep enough. It creates super-specific rules that don't generalize.

```
Overfit tree: "If age=27 AND salary=52341 AND zip=90210 → YES"
Good tree:    "If age > 25 AND salary > 50000 → YES"
```

### How to Fix Overfitting

| Technique | What It Does |
|-----------|-------------|
| `max_depth` | Limit how deep the tree grows |
| `min_samples_split` | Require minimum samples to split a node |
| `min_samples_leaf` | Require minimum samples in each leaf |
| **Pruning** | Grow full tree, then remove branches that don't help |

Or better yet — use **Random Forests** (next doc), which combine many trees to fix this problem.

## For Classification AND Regression

- **Classification tree** — leaf nodes vote on a class (majority wins)
- **Regression tree** — leaf nodes average the values

```python
from sklearn.tree import DecisionTreeClassifier   # for categories
from sklearn.tree import DecisionTreeRegressor     # for numbers
```

## Python Example

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris

# Load famous iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train with max_depth to prevent overfitting
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# See the actual tree (human-readable!)
tree_rules = export_text(model, feature_names=iris.feature_names)
print(tree_rules)

# Feature importance — which features matter most?
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"  {name}: {importance:.2f}")
```

## Key Takeaway

Decision trees are the **most interpretable ML model** — you can read and explain every decision. They're great for understanding your data, but a single tree tends to overfit. In practice, you'll usually use ensemble methods (Random Forest, Gradient Boosting) that combine many trees for better accuracy.
