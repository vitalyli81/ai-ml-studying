"""Learn more Python by building a random forest.

Part 4 — the fix for Part 3's overfitting. One deep tree memorizes noise
(+29% train/test gap); a jury of diverse trees cancels it out. New Python:
  STEP 0: importing from a SIBLING folder    (learn: pathlib, __file__, sys.path)
  STEP 1: bootstrap sampling + OOB rows      (learn: sets and set operations)
  STEP 2: the forest                         (learn: lists of trees, voting via mean)
  STEP 3: out-of-bag scoring                 (learn: defaultdict of votes)
  STEP 4: tree vs forest, timed              (learn: time.perf_counter)
  STEP 5: sklearn's version                  (n_jobs=-1, oob_score=True)

Theory companion: ../../ml/random-forest.md (bagging, random features, OOB).

Run from python/ml-practice/:
    uv run random-forest/random_forest.py
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ──────────────────────────────────────────────────────────────────────────────
# STEP 0 — Import Part 3's building blocks from a SIBLING folder.
# Python imports search sys.path; a script's own folder is on it, but
# ../decision-trees is not. pathlib + __file__ fix that explicitly.
# ──────────────────────────────────────────────────────────────────────────────

sys.path.append(str(Path(__file__).parent.parent / "decision-trees"))
from decision_trees import TreeNode, gini, majority, make_loans, predict_one

FEATURE_NAMES = ["income_k", "credit_score", "debt_ratio"]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Bootstrap sampling: rows WITH replacement; the left-out rows (OOB)
# fall out of a set difference.
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_indices(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw n row indices with replacement; also return the rows never drawn."""
    sampled = rng.integers(0, n, size=n)              # repeats allowed — that's the point
    oob = np.array(sorted(set(range(n)) - set(sampled.tolist())))   # set difference!
    return sampled, oob


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — A tree that only considers a RANDOM SUBSET of features per split
# (trick #2 from the theory doc — bagging alone isn't enough diversity)
# ──────────────────────────────────────────────────────────────────────────────

def best_split_random(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_features: int
) -> tuple[int, float, float]:
    """Part 3's best_split, but each call may only look at a random
    handful of features — so the trees are FORCED to differ."""
    allowed = rng.choice(X.shape[1], size=max_features, replace=False)
    best = (0, 0.0, float("inf"))
    for feature in allowed:
        values = np.unique(X[:, feature])
        for threshold in (values[:-1] + values[1:]) / 2:
            mask = X[:, feature] <= threshold
            left, right = y[mask], y[~mask]
            weighted = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
            if weighted < best[2]:
                best = (int(feature), float(threshold), weighted)
    return best


def build_random_tree(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
    max_features: int, depth: int = 0, max_depth: int | None = None,
) -> TreeNode:
    if len(set(y)) == 1:
        return TreeNode(prediction=int(y[0]))
    if max_depth is not None and depth >= max_depth:
        return TreeNode(prediction=majority(y))

    feature, threshold, weighted = best_split_random(X, y, rng, max_features)
    if weighted >= gini(y):
        return TreeNode(prediction=majority(y))

    mask = X[:, feature] <= threshold
    return TreeNode(
        feature=feature, threshold=threshold,
        left=build_random_tree(X[mask], y[mask], rng, max_features, depth + 1, max_depth),
        right=build_random_tree(X[~mask], y[~mask], rng, max_features, depth + 1, max_depth),
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — The forest: many diverse trees + a majority vote + free OOB score
# ──────────────────────────────────────────────────────────────────────────────

class ScratchRandomForest:
    def __init__(self, n_trees: int = 50, max_features: int | None = None, seed: int = 0):
        self.n_trees = n_trees
        self.max_features = max_features      # None → round(sqrt(n_features))
        self.seed = seed
        self.trees_: list[TreeNode] = []
        self.oob_sets_: list[np.ndarray] = []
        self.oob_score_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScratchRandomForest":
        rng = np.random.default_rng(self.seed)
        k = self.max_features or round(np.sqrt(X.shape[1]))
        self.trees_, self.oob_sets_ = [], []

        for _ in range(self.n_trees):
            sampled, oob = bootstrap_indices(len(y), rng)
            tree = build_random_tree(X[sampled], y[sampled], rng, k)  # grown DEEP
            self.trees_.append(tree)
            self.oob_sets_.append(oob)

        self.oob_score_ = self._oob_score(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Every tree votes on every row; majority wins.
        votes is (n_trees × n_rows); mean over axis=0 = fraction voting 1."""
        votes = np.stack([[predict_one(t, row) for row in X] for t in self.trees_])
        return (votes.mean(axis=0) >= 0.5).astype(int)

    def _oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Each row is judged ONLY by the trees that never saw it —
        a free validation estimate, no test set spent."""
        ballots: dict[int, list[int]] = defaultdict(list)
        for tree, oob in zip(self.trees_, self.oob_sets_):
            for i in oob:
                ballots[i].append(predict_one(tree, X[i]))
        verdicts = {i: int(np.mean(v) >= 0.5) for i, v in ballots.items()}
        return float(np.mean([y[i] == pred for i, pred in verdicts.items()]))


# ──────────────────────────────────────────────────────────────────────────────
# Putting it together
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = np.random.default_rng(42)

    print("STEP 1 — bootstrap sampling (n=10, watch the repeats and gaps):")
    sampled, oob = bootstrap_indices(10, rng)
    print(f"  sampled rows: {sorted(sampled.tolist())}")
    print(f"  never drawn (out-of-bag): {oob.tolist()}")
    fractions = [len(bootstrap_indices(300, rng)[1]) / 300 for _ in range(500)]
    print(f"  OOB fraction over 500 draws of n=300: {np.mean(fractions):.1%} "
          f"(theory says 1/e ≈ 36.8%)\n")

    X, y = make_loans()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print("STEP 4 — one deep tree vs the jury (same 300 applicants as Part 3):")
    t0 = time.perf_counter()
    lone_tree = DecisionTreeClassifier(max_depth=None, random_state=42).fit(X_train, y_train)
    t_tree = time.perf_counter() - t0

    t0 = time.perf_counter()
    forest = ScratchRandomForest(n_trees=50, seed=0).fit(X_train, y_train)
    t_forest = time.perf_counter() - t0

    rows = [
        ("single deep tree", lone_tree.score(X_train, y_train),
         lone_tree.score(X_test, y_test), t_tree),
        ("scratch forest (50)", np.mean(forest.predict(X_train) == y_train),
         np.mean(forest.predict(X_test) == y_test), t_forest),
    ]
    print(f"  {'model':>20} | {'train':>5} | {'test':>5} | {'gap':>4} | {'fit time':>8}")
    for name, tr, te, secs in rows:
        print(f"  {name:>20} | {tr:>5.0%} | {te:>5.0%} | {tr - te:>+4.0%} | {secs:>7.2f}s")
    print(f"  forest OOB estimate: {forest.oob_score_:.0%} "
          f"(vs actual test {np.mean(forest.predict(X_test) == y_test):.0%} — "
          "free validation, no test set spent)\n")

    print("STEP 5 — more trees, better jury. One run of each size is too noisy")
    print("to trust (a single tree can get lucky), so: 5 seeds each, mean ± spread:")
    sizes = [1, 5, 10, 25, 50]
    size_scores = []
    for n in sizes:
        accs = [float(np.mean(ScratchRandomForest(n_trees=n, seed=s)
                              .fit(X_train, y_train).predict(X_test) == y_test))
                for s in range(5)]
        size_scores.append(float(np.mean(accs)))
        print(f"  {n:>3} trees → test {np.mean(accs):.0%} ± {np.std(accs):.1%}")
    print("  → bigger juries are better AND more consistent (smaller ±)\n")

    print("STEP 6 — sklearn's forest (parallel, optimized — same idea):")
    sk = RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs=-1,
                                random_state=42).fit(X_train, y_train)
    print(f"  test {sk.score(X_test, y_test):.0%}, OOB {sk.oob_score_:.0%}")
    ranked = sorted(zip(FEATURE_NAMES, sk.feature_importances_),
                    key=lambda p: p[1], reverse=True)
    print("  importances: " + ", ".join(f"{n} {v:.0%}" for n, v in ranked))

    # Plot: the jury effect — individual trees are mediocre, the vote is not
    single_accs = []
    for tree, _ in zip(forest.trees_, forest.oob_sets_):
        preds = np.array([predict_one(tree, row) for row in X_test])
        single_accs.append(float(np.mean(preds == y_test)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(single_accs, bins=12, color="tan", edgecolor="black",
             label="individual trees (50)")
    ax1.axvline(rows[1][2], color="green", linewidth=2.5,
                label=f"the forest's vote ({rows[1][2]:.0%})")
    ax1.axvline(rows[0][2], color="red", linestyle="--", linewidth=2,
                label=f"single deep tree ({rows[0][2]:.0%})")
    ax1.set_xlabel("Test accuracy")
    ax1.set_ylabel("Number of trees")
    ax1.set_title("Mediocre jurors, strong jury")
    ax1.legend()

    ax2.plot([str(s) for s in sizes], size_scores, "o-")
    ax2.set_xlabel("Number of trees")
    ax2.set_ylabel("Test accuracy")
    ax2.set_title("More trees help — with diminishing returns")

    fig.tight_layout()
    out = "random-forest/forest_plot.png"
    fig.savefig(out, dpi=120)
    print(f"\n  plot saved → {out}")


if __name__ == "__main__":
    main()
