"""Learn more Python by building gradient boosting.

Part 5 — the forest's rival. Random forest grows trees in parallel and averages;
boosting grows them ONE AT A TIME, each fitting what's still wrong (the residuals).
  STEP 1: the regression tree            (Part 3's tree, two lines swapped)
  STEP 2: the boosting loop              (predict → residual → fit tree → repeat)
  STEP 3: the theory doc's trace, live   (watch residuals shrink round by round)
  STEP 4: early stopping                 (learn: optional kwargs, slicing as rollback)
  STEP 5: the learning-rate tradeoff     (small steps win, given enough of them)
  STEP 6: sklearn's GradientBoosting     (and why production uses XGBoost)

Theory companion: ../../ml/gradient-boosting.md

Run from python/ml-practice/:
    uv run gradient-boosting/gradient_boosting.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — A REGRESSION tree: Part 3's classifier with two lines swapped.
#   messiness:  gini      → variance
#   leaf value: majority  → mean
# Same recursion, same best-split search. That's the whole difference.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RegNode:
    value: float | None = None             # leaves predict a NUMBER now
    feature: int | None = None
    threshold: float | None = None
    left: "RegNode | None" = None
    right: "RegNode | None" = None


def best_split_reg(X: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    """Lowest weighted VARIANCE instead of lowest weighted gini."""
    best = (0, 0.0, float("inf"))
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        for threshold in (values[:-1] + values[1:]) / 2:
            mask = X[:, feature] <= threshold
            left, right = y[mask], y[~mask]
            weighted = (len(left) * np.var(left) + len(right) * np.var(right)) / len(y)
            if weighted < best[2]:
                best = (feature, float(threshold), float(weighted))
    return best


def build_reg_tree(X: np.ndarray, y: np.ndarray, depth: int = 0, max_depth: int = 2) -> RegNode:
    if depth >= max_depth or len(y) <= 2 or np.var(y) == 0.0:
        return RegNode(value=float(y.mean()))          # leaf = the group's MEAN

    feature, threshold, weighted = best_split_reg(X, y)
    if weighted >= np.var(y):
        return RegNode(value=float(y.mean()))

    mask = X[:, feature] <= threshold
    return RegNode(
        feature=feature, threshold=threshold,
        left=build_reg_tree(X[mask], y[mask], depth + 1, max_depth),
        right=build_reg_tree(X[~mask], y[~mask], depth + 1, max_depth),
    )


def tree_predict(node: RegNode, X: np.ndarray) -> np.ndarray:
    def one(n: RegNode, row: np.ndarray) -> float:
        if n.value is not None:
            return n.value
        return one(n.left if row[n.feature] <= n.threshold else n.right, row)
    return np.array([one(node, row) for row in X])


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — The boosting machine: start with the mean, then repeatedly fit a
# small tree to WHAT'S STILL WRONG and add a fraction of its correction.
# ──────────────────────────────────────────────────────────────────────────────

class ScratchGradientBoosting:
    def __init__(self, n_trees: int = 300, lr: float = 0.1,
                 max_depth: int = 2, patience: int | None = None):
        self.n_trees = n_trees
        self.lr = lr
        self.max_depth = max_depth
        self.patience = patience            # None = early stopping off
        self.f0_: float = 0.0
        self.trees_: list[RegNode] = []
        self.train_rmse_: list[float] = []
        self.val_rmse_: list[float] = []
        self.best_iteration_: int | None = None

    def fit(self, X, y, X_val=None, y_val=None, verbose: bool = False):
        """X_val/y_val are OPTIONAL keyword args — pass them to get a per-round
        validation curve and (with patience) early stopping."""
        self.f0_ = float(y.mean())                     # round 0: predict the average
        pred = np.full(len(y), self.f0_)
        val_pred = np.full(len(y_val), self.f0_) if X_val is not None else None

        self.trees_, self.train_rmse_, self.val_rmse_ = [], [], []
        best_val, best_iter, rounds_since_best = float("inf"), 0, 0

        for round_no in range(1, self.n_trees + 1):    # enumerate-style, from 1
            residuals = y - pred                       # what's still wrong
            tree = build_reg_tree(X, residuals, max_depth=self.max_depth)
            self.trees_.append(tree)

            pred = pred + self.lr * tree_predict(tree, X)   # a FRACTION of the fix
            self.train_rmse_.append(rmse(y, pred))

            if X_val is not None:
                val_pred = val_pred + self.lr * tree_predict(tree, X_val)
                val_score = rmse(y_val, val_pred)
                self.val_rmse_.append(val_score)

                if val_score < best_val:
                    best_val, best_iter, rounds_since_best = val_score, round_no, 0
                else:
                    rounds_since_best += 1
                if self.patience is not None and rounds_since_best >= self.patience:
                    if verbose:
                        print(f"    early stop at round {round_no} "
                              f"(best was round {best_iter})")
                    break

            if verbose and round_no <= 3:
                print(f"    round {round_no}: residual RMSE {rmse(y, y - residuals):.0f} "
                      f"→ {self.train_rmse_[-1]:.0f}")

        if X_val is not None and self.patience is not None:
            self.best_iteration_ = best_iter
            self.trees_ = self.trees_[:best_iter]      # slicing = roll back the model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.full(len(X), self.f0_)
        for tree in self.trees_:
            pred = pred + self.lr * tree_predict(tree, X)
        return pred


# ──────────────────────────────────────────────────────────────────────────────
# Data: 300 houses with a NON-linear price process (a straight line can't win)
# ──────────────────────────────────────────────────────────────────────────────

def make_houses(n: int = 300, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sqft = rng.uniform(500, 2500, size=n)
    age = rng.uniform(0, 50, size=n)
    renovated = rng.integers(0, 2, size=n)             # 0/1
    price = (
        60_000
        + 170 * sqft
        - 1_200 * age
        + 55_000 * renovated
        + 90_000 * (sqft > 1800)                       # a step: big-house premium
        + 1_000 * np.sqrt(sqft) * renovated            # an interaction
        + rng.normal(0, 18_000, size=n)
    )
    return np.column_stack([sqft, age, renovated]), price


def main() -> None:
    # STEP 3 — the theory doc's hand-trace, executed (5 houses, lr=0.1)
    print("STEP 3 — the theory doc's trace, live (5 houses, stumps, lr=0.1):")
    sqft5 = np.array([[800.0], [900.0], [1000.0], [1200.0], [1500.0]])
    price5 = np.array([150_000.0, 200_000.0, 250_000.0, 300_000.0, 350_000.0])
    print(f"    round 0: predict the mean for everyone → {price5.mean():,.0f}")
    tiny = ScratchGradientBoosting(n_trees=3, lr=0.1, max_depth=1)
    tiny.fit(sqft5, price5, verbose=True)
    print("    → every round, a small tree eats a bite of the leftover error\n")

    # The real workflow
    X, y = make_houses()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42)

    print("STEP 4 — early stopping on 300 houses (n_trees=300 offered, "
          "patience=20):")
    model = ScratchGradientBoosting(n_trees=300, lr=0.1, patience=20)
    model.fit(X_train, y_train, X_val, y_val, verbose=True)
    print(f"    kept {len(model.trees_)} trees "
          f"→ test RMSE ${rmse(y_test, model.predict(X_test)):,.0f}\n")

    print("STEP 5 — head-to-head on the same split:")
    deep_tree = DecisionTreeRegressor(max_depth=None, random_state=42)
    deep_tree.fit(X_train, y_train)
    sk = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1,
                                   max_depth=2, n_iter_no_change=20,
                                   random_state=42).fit(X_train, y_train)
    rows = [
        ("single deep tree", rmse(y_test, deep_tree.predict(X_test))),
        ("scratch boosting", rmse(y_test, model.predict(X_test))),
        ("sklearn boosting", rmse(y_test, sk.predict(X_test))),
        ("noise floor (σ)", 18_000.0),
    ]
    for name, score in rows:
        print(f"    {name:>17}: ${score:>7,.0f} test RMSE")
    print("    → boosting cuts the lone tree's error by ~25% and closes in on "
          "the noise floor\n")

    # STEP 6 — the learning-rate tradeoff (full curves, no early stop)
    print("STEP 6 — learning rate: big confident steps vs many small ones:")
    curves: dict[float, list[float]] = {}
    for lr in [1.0, 0.3, 0.1]:
        m = ScratchGradientBoosting(n_trees=300, lr=lr, patience=None)
        m.fit(X_train, y_train, X_val, y_val)
        curves[lr] = m.val_rmse_
        best_round = int(np.argmin(m.val_rmse_)) + 1
        print(f"    lr={lr:<4} best val RMSE ${min(m.val_rmse_):>7,.0f} "
              f"at round {best_round:>3}, final ${m.val_rmse_[-1]:>7,.0f}")
    print("    → lr=1.0 bottoms out early then OVERFITS; lr=0.1 keeps improving\n")

    # Plot: validation curves per lr + train-vs-val for lr=0.1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for lr, curve in curves.items():
        ax1.plot(range(1, len(curve) + 1), curve, label=f"lr={lr}")
    ax1.set_xlabel("Boosting round")
    ax1.set_ylabel("Validation RMSE ($)")
    ax1.set_title("Small learning rate wins — given enough rounds")
    ax1.legend()

    full = ScratchGradientBoosting(n_trees=300, lr=0.1, patience=None)
    full.fit(X_train, y_train, X_val, y_val)
    rounds = range(1, len(full.val_rmse_) + 1)
    ax2.plot(rounds, full.train_rmse_, label="train RMSE")
    ax2.plot(rounds, full.val_rmse_, label="validation RMSE")
    stop = int(np.argmin(full.val_rmse_)) + 1
    ax2.axvline(stop, color="gray", linestyle="--",
                label=f"early stop would cut here ({stop})")
    ax2.set_xlabel("Boosting round")
    ax2.set_ylabel("RMSE ($)")
    ax2.set_title("Train keeps falling forever; validation knows when to stop")
    ax2.legend()

    fig.tight_layout()
    out = "gradient-boosting/boosting_plot.png"
    fig.savefig(out, dpi=120)
    print(f"    plot saved → {out}")


if __name__ == "__main__":
    main()
