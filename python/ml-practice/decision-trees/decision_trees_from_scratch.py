"""Decision trees FROM SCRATCH — sklearn fully benched.

Part 3 (decision_trees.py) already built gini/best_split/build_tree by hand;
sklearn only supplied the workflow. This file replaces that too:
  - train/test split          → our own shuffle-and-cut
  - the depth sweep           → our tree, our accuracy
  - feature_importances_      → accumulated impurity drops (defaultdict)
  - export_text / plot_tree   → recursion utilities + a matplotlib bar chart

New Python:
  - importing from YOUR OWN module (and why `if __name__ == "__main__"` exists)
  - collections.defaultdict
  - max() over recursive calls (tree_depth)

Run from python/ml-practice/:
    uv run decision-trees/decision_trees_from_scratch.py
"""

from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt

# Part 3's building blocks — imported like any library, because your own files
# ARE modules. Note: decision_trees.py's main() does NOT run here; its
# `if __name__ == "__main__":` guard exists precisely for this moment.
from decision_trees import TreeNode, best_split, gini, majority, make_loans, predict_one

FEATURE_NAMES = ["income_k", "credit_score", "debt_ratio"]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — The workflow pieces sklearn was doing: split + accuracy
# (Same patterns you built in linear-regression/from-scratch.md and Part 2.)
# ──────────────────────────────────────────────────────────────────────────────

def train_test_split(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.3, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(y))
    cut = int(len(y) * (1 - test_ratio))
    train, test = shuffled[:cut], shuffled[cut:]
    return X[train], X[test], y[train], y[test]


def accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float((np.asarray(actual) == np.asarray(predicted)).mean())


def predict(node: TreeNode, X: np.ndarray) -> np.ndarray:
    """Batch prediction: one recursive walk per row, via a list comprehension."""
    return np.array([predict_one(node, row) for row in X])


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — build_tree with a twist: record each split's impurity drop.
# That running total per feature IS feature importance.
# ──────────────────────────────────────────────────────────────────────────────

def build_tree(
    X: np.ndarray,
    y: np.ndarray,
    depth: int = 0,
    max_depth: int | None = 3,
    importance: dict[int, float] | None = None,
) -> TreeNode:
    """Same recursion as Part 3 + an accumulator threaded through every call.

    `importance` is a defaultdict the CALLER owns. Every recursive call writes
    into the same dict (dicts are passed by reference, like JS objects), so
    when recursion finishes, the caller holds the totals.
    """
    if importance is None:
        importance = defaultdict(float)  # missing keys auto-start at 0.0

    if len(set(y)) == 1:
        return TreeNode(prediction=int(y[0]))
    if max_depth is not None and depth >= max_depth:
        return TreeNode(prediction=majority(y))

    feature, threshold, weighted = best_split(X, y)
    if weighted >= gini(y):
        return TreeNode(prediction=majority(y))

    # This split reduced messiness by (gini before − weighted gini after),
    # credited to the feature it used, weighted by how many rows it affected.
    importance[feature] += len(y) * (gini(y) - weighted)

    mask = X[:, feature] <= threshold
    return TreeNode(
        feature=feature,
        threshold=threshold,
        left=build_tree(X[mask], y[mask], depth + 1, max_depth, importance),
        right=build_tree(X[~mask], y[~mask], depth + 1, max_depth, importance),
    )


def normalized_importance(importance: dict[int, float]) -> dict[str, float]:
    """Raw impurity-drop totals → fractions that sum to 1 (sklearn's convention)."""
    total = sum(importance.values())
    return {FEATURE_NAMES[f]: drop / total for f, drop in importance.items()}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Recursion utilities: measure the tree itself
# ──────────────────────────────────────────────────────────────────────────────

def count_nodes(node: TreeNode) -> int:
    if node.prediction is not None:        # a leaf counts as 1
        return 1
    return 1 + count_nodes(node.left) + count_nodes(node.right)


def tree_depth(node: TreeNode) -> int:
    if node.prediction is not None:        # a leaf adds no depth
        return 0
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — The overfitting sweep, 100% our own code
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    X, y = make_loans()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print("THE SWEEP — every number below comes from code in this repo, not sklearn:")
    print(f"  {'max_depth':>9} | {'train':>5} | {'test':>5} | {'gap':>4} | {'nodes':>5}")
    depths: list[int | None] = [1, 2, 3, 5, None]
    train_scores, test_scores = [], []
    for depth in depths:
        tree = build_tree(X_train, y_train, max_depth=depth)
        tr = accuracy(y_train, predict(tree, X_train))
        te = accuracy(y_test, predict(tree, X_test))
        train_scores.append(tr)
        test_scores.append(te)
        print(f"  {str(depth):>9} | {tr:>5.0%} | {te:>5.0%} | {tr - te:>+4.0%} "
              f"| {count_nodes(tree):>5}")
    print("  → same story as sklearn's sweep: unlimited depth buys train accuracy")
    print("    with NODES (memorized noise), and test accuracy pays for it\n")

    print("FEATURE IMPORTANCE — from the accumulator, no sklearn:")
    importance: dict[int, float] = defaultdict(float)
    tree3 = build_tree(X_train, y_train, max_depth=3, importance=importance)
    ranked = sorted(normalized_importance(importance).items(),
                    key=lambda pair: pair[1], reverse=True)
    for name, frac in ranked:
        bar = "█" * round(frac * 40)              # a terminal bar chart
        print(f"  {name:>14}: {frac:>4.0%} {bar}")
    print(f"\n  depth-3 tree: {count_nodes(tree3)} nodes, "
          f"actual depth {tree_depth(tree3)}, "
          f"test accuracy {accuracy(y_test, predict(tree3, X_test)):.0%}")

    # Plot: the overfitting curve + the importance bars — matplotlib only
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = [str(d) for d in depths]
    ax1.plot(labels, train_scores, "o-", label="train accuracy")
    ax1.plot(labels, test_scores, "o-", label="test accuracy")
    ax1.set_xlabel("max_depth")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Overfitting sweep — our tree, our split, our accuracy")
    ax1.legend()

    names = [name for name, _ in ranked]
    fracs = [frac for _, frac in ranked]
    ax2.barh(names[::-1], fracs[::-1], color="seagreen")   # [::-1] = reversed
    ax2.set_xlabel("Fraction of total impurity reduction")
    ax2.set_title("Feature importance, accumulated during recursion")

    fig.tight_layout()
    out = "decision-trees/from_scratch_plot.png"
    fig.savefig(out, dpi=120)
    print(f"  plot saved → {out}")


if __name__ == "__main__":
    main()
