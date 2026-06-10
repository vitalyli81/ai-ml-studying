"""Learn more Python by building a decision tree.

Part 3 — no gradient descent this time. Trees ask questions instead. New Python:
  STEP 2: Gini impurity                 (learn: collections.Counter)
  STEP 3: finding the best question     (learn: nested loops, float("inf") sentinel)
  STEP 4: growing the tree              (learn: RECURSION + self-referencing dataclass)
  STEP 5: overfitting, measured         (learn: the depth sweep, train/test gap)
  STEP 6: sklearn trees you can READ    (learn: export_text, sorted(key=lambda))

Theory companion: ../../ml/decision-trees.md (same loan data, same Gini numbers).

Run from python/ml-practice/:
    uv run decision-trees/decision_trees.py
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — The loan data from ml/decision-trees.md (Income > 40K splits it perfectly)
# ──────────────────────────────────────────────────────────────────────────────

FEATURES = ["income_k", "credit_score"]
X_LOANS = np.array([
    [30, 600],
    [60, 750],
    [45, 680],
    [25, 550],
    [80, 800],
    [35, 650],
])
Y_LOANS = np.array([1, 0, 0, 1, 0, 1])  # 1 = defaulted, 0 = repaid


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Gini impurity: the "messiness score" (0 = pure, 0.5 = 50/50 mess)
# ──────────────────────────────────────────────────────────────────────────────

def gini(labels: np.ndarray) -> float:
    """1 − Σ(fraction of each class)². Counter does the counting."""
    counts = Counter(labels)                  # {1: 3, 0: 3} — a dict of tallies
    n = len(labels)
    return 1.0 - sum((count / n) ** 2 for count in counts.values())


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — The best question: try every feature, every threshold, keep the
#          split with the lowest weighted Gini. (The whole training algorithm.)
# ──────────────────────────────────────────────────────────────────────────────

def best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    """Returns (feature_index, threshold, weighted_gini) of the best question."""
    best = (0, 0.0, float("inf"))             # sentinel: anything beats infinity

    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])     # sorted unique values
        # Candidate thresholds = midpoints between consecutive values
        midpoints = (values[:-1] + values[1:]) / 2
        for threshold in midpoints:
            mask = X[:, feature] <= threshold        # boolean mask → two groups
            left, right = y[mask], y[~mask]          # ~ flips the mask
            weighted = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
            if weighted < best[2]:
                best = (feature, float(threshold), weighted)
    return best


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Grow the tree: recursion. A node either asks a question (and has two
#          child nodes) or is a leaf that predicts. Children are TreeNodes too.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TreeNode:
    prediction: int | None = None          # set on leaves
    feature: int | None = None             # set on question nodes…
    threshold: float | None = None
    left: "TreeNode | None" = None         # ← the type refers to ITSELF
    right: "TreeNode | None" = None


def majority(labels: np.ndarray) -> int:
    """The most common label — what a leaf predicts."""
    return int(Counter(labels).most_common(1)[0][0])


def build_tree(X: np.ndarray, y: np.ndarray, depth: int = 0, max_depth: int = 3) -> TreeNode:
    # BASE CASES — when to stop asking questions and just answer:
    if len(set(y)) == 1:                          # pure: everyone has the same label
        return TreeNode(prediction=int(y[0]))
    if depth >= max_depth:                        # depth limit: anti-overfitting
        return TreeNode(prediction=majority(y))

    feature, threshold, weighted = best_split(X, y)
    if weighted >= gini(y):                       # no question helps → give up, answer
        return TreeNode(prediction=majority(y))

    # RECURSIVE CASE — split, then grow each side the same way
    mask = X[:, feature] <= threshold
    return TreeNode(
        feature=feature,
        threshold=threshold,
        left=build_tree(X[mask], y[mask], depth + 1, max_depth),
        right=build_tree(X[~mask], y[~mask], depth + 1, max_depth),
    )


def predict_one(node: TreeNode, row: np.ndarray) -> int:
    """Walk the tree: recursion again, one branch per question."""
    if node.prediction is not None:               # reached a leaf
        return node.prediction
    branch = node.left if row[node.feature] <= node.threshold else node.right
    return predict_one(branch, row)


def print_tree(node: TreeNode, names: list[str], depth: int = 0) -> None:
    """Recursive pretty-printer — indentation = string multiplication."""
    pad = "    " * depth
    if node.prediction is not None:
        label = "DEFAULT" if node.prediction == 1 else "repaid"
        print(f"{pad}→ predict {label}")
    else:
        print(f"{pad}[{names[node.feature]} <= {node.threshold:g}?]")
        print(f"{pad}  yes:")
        print_tree(node.left, names, depth + 1)
        print(f"{pad}  no:")
        print_tree(node.right, names, depth + 1)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5/6 — Realistic data, the overfitting sweep, and sklearn's readable trees
# ──────────────────────────────────────────────────────────────────────────────

def make_loans(n: int = 300, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """300 synthetic applicants; default risk depends on income, credit, debt."""
    rng = np.random.default_rng(seed)
    income = rng.uniform(20, 100, size=n)         # $K
    credit = rng.uniform(500, 850, size=n)
    debt_ratio = rng.uniform(0.0, 1.0, size=n)
    z = -0.06 * income - 0.012 * credit + 3.0 * debt_ratio + 9.0
    p_default = 1.0 / (1.0 + np.exp(-z))
    defaulted = (rng.random(n) < p_default).astype(int)
    return np.column_stack([income, credit, debt_ratio]), defaulted


def main() -> None:
    print("STEP 2 — Gini sanity checks (numbers from ml/decision-trees.md):")
    print(f"  all one class  {gini(np.array([1, 1, 1, 1])):.2f}   (pure)")
    print(f"  50/50 mix      {gini(np.array([1, 1, 0, 0])):.2f}   (maximally messy)")
    print(f"  80/20 mix      {gini(np.array([1, 1, 1, 1, 0])):.2f}   (in between)\n")

    print("STEP 3 — best first question for the 6 loan applicants:")
    feature, threshold, weighted = best_split(X_LOANS, Y_LOANS)
    print(f"  [{FEATURES[feature]} <= {threshold:g}?]  → weighted Gini {weighted:.2f}")
    print("  → a PERFECT split (Gini 0), exactly the theory doc's answer\n")

    print("STEP 4 — the recursive tree, grown and printed:")
    tree = build_tree(X_LOANS, Y_LOANS, max_depth=3)
    print_tree(tree, FEATURES)
    applicants = {"$55K, 720 credit": [55, 720], "$28K, 590 credit": [28, 590]}
    for desc, row in applicants.items():
        verdict = "DEFAULT" if predict_one(tree, np.array(row)) else "repaid"
        print(f"  new applicant {desc} → {verdict}")
    print()

    print("STEP 5 — the overfitting sweep on 300 noisy applicants:")
    X, y = make_loans()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    names = ["income_k", "credit_score", "debt_ratio"]

    depths: list[int | None] = [1, 2, 3, 5, None]
    train_scores, test_scores = [], []
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))
        gap = train_scores[-1] - test_scores[-1]
        print(f"  depth {str(depth):>4}: train={train_scores[-1]:.0%}  "
              f"test={test_scores[-1]:.0%}  gap={gap:+.0%}")
    print("  → unlimited depth memorizes the training set; the gap is the tell\n")

    print("STEP 6 — the depth-3 tree is a model you can READ:")
    best = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
    print(export_text(best, feature_names=names))

    ranked = sorted(zip(names, best.feature_importances_),
                    key=lambda pair: pair[1], reverse=True)
    print("  feature importance (fraction of impurity reduction):")
    for name, score in ranked:
        print(f"    {name:>14}: {score:.0%}")

    # Plot: the tree itself + the overfitting curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_tree(best, feature_names=names, class_names=["repaid", "default"],
              filled=True, fontsize=7, ax=ax1)
    ax1.set_title("The depth-3 tree (sklearn plot_tree)")

    labels = [str(d) for d in depths]
    ax2.plot(labels, train_scores, "o-", label="train accuracy")
    ax2.plot(labels, test_scores, "o-", label="test accuracy")
    ax2.set_xlabel("max_depth")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Overfitting: train keeps climbing, test does not")
    ax2.legend()

    fig.tight_layout()
    out = "decision-trees/tree_plot.png"
    fig.savefig(out, dpi=120)
    print(f"\n  plot saved → {out}")


if __name__ == "__main__":
    main()
