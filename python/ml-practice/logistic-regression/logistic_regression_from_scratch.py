"""Logistic regression FROM SCRATCH — sklearn fully benched.

Part 2 (logistic_regression.py) built the single-feature model and metrics;
sklearn supplied the multi-feature model, the split, and the report. Not anymore:
  LogisticRegression      → multi-feature gradient descent with L2 regularization
  train_test_split        → our own STRATIFIED shuffle-and-cut
  classification_report   → our own per-class table

This also resolves Part 2's cliffhanger: on separable data our confidence grew
without bound, and we said "sklearn's C parameter stops that." Here we BUILD
the thing that stops it.

New Python / NumPy:
  - the @ operator (matrix × vector), .T transpose
  - axis=0 reductions (per-column mean/std)
  - np.where, np.concatenate, np.logspace

Run from python/ml-practice/:
    uv run logistic-regression/logistic_regression_from_scratch.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt

# Part 2's building blocks — your own module, imported (its __main__ guard
# keeps its demo from running).
from logistic_regression import (
    HOURS,
    PASSED,
    accuracy,
    make_students,
    precision,
    recall,
    sigmoid,
)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Stratified train/test split (what sklearn's stratify=y was doing)
# ──────────────────────────────────────────────────────────────────────────────

def stratified_split(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.25, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle-and-cut *within each class*, so train and test keep the same
    pass/fail ratio. A plain split can hand you a test set that's 70% passes
    by bad luck — stratification removes that luck."""
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]        # indices of THIS class's rows
        idx = rng.permutation(idx)
        cut = int(len(idx) * (1 - test_ratio))
        train_parts.append(idx[:cut])
        test_parts.append(idx[cut:])

    train = rng.permutation(np.concatenate(train_parts))   # glue + reshuffle
    test = rng.permutation(np.concatenate(test_parts))
    return X[train], X[test], y[train], y[test]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Multi-feature logistic regression with L2 regularization
# ──────────────────────────────────────────────────────────────────────────────

class LogisticRegressionScratch:
    """Part 2's model, upgraded twice:

    1. MULTI-FEATURE — the weights are a vector now; `X @ w` computes every
       student's score in one matrix-vector multiply.
    2. L2 REGULARIZATION — `+ l2 * w` in the gradient constantly shrinks the
       weights toward 0, so confidence can't grow without bound. This is the
       from-scratch version of sklearn's C (note: C = 1/regularization, so
       BIG C = weak penalty, small l2 here = weak penalty).
    """

    def __init__(self, lr: float = 0.5, epochs: int = 20_000, l2: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        mu = X.mean(axis=0)                 # axis=0 → one mean PER COLUMN
        sigma = X.std(axis=0)
        X_std = (X - mu) / sigma            # 2D − 1D: broadcasts row by row

        w = np.zeros(X.shape[1])            # one weight per feature
        b = 0.0
        n = len(y)

        for _ in range(self.epochs):
            p = sigmoid(X_std @ w + b)      # (n×k) @ (k,) → (n,) scores at once
            error = p - y
            grad_w = X_std.T @ error / n + self.l2 * w   # ← the regularizer
            grad_b = error.mean()
            w -= self.lr * grad_w
            b -= self.lr * grad_b

        # De-standardize back to raw feature units
        self.coef_ = w / sigma
        self.intercept_ = float(b - np.sum(w * mu / sigma))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.coef_ + self.intercept_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — classification_report, from scratch
# ──────────────────────────────────────────────────────────────────────────────

def f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    p, r = precision(actual, predicted), recall(actual, predicted)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def classification_report_scratch(
    actual: np.ndarray, predicted: np.ndarray, names: tuple[str, ...]
) -> str:
    """Per-class table. The trick: our precision/recall only know binary 0/1 —
    so for each class, RELABEL the problem ('is it this class, yes/no?') and
    the binary metrics work unchanged."""
    lines = [f"{'':>10} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>9}"]
    for cls, name in enumerate(names):
        a = (actual == cls).astype(int)        # this class vs everything else
        p = (predicted == cls).astype(int)
        lines.append(f"{name:>10} {precision(a, p):>9.2f} {recall(a, p):>9.2f} "
                     f"{f1(a, p):>9.2f} {int(a.sum()):>9}")
    lines.append(f"{'accuracy':>10} {'':>9} {'':>9} "
                 f"{accuracy(actual, predicted):>9.2f} {len(actual):>9}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Putting it together
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("STEP 2 — Part 2's cliffhanger, resolved. Same 6 students, with and")
    print("without the L2 leash (P(pass) after 20,000 epochs):\n")
    hours_2d = HOURS.reshape(-1, 1)            # the model wants 2D now
    print(f"  {'l2':>6} | {'coef':>7} | {'P(3h)':>6} | {'P(4h)':>6} | {'P(5h)':>6}")
    for l2 in [0.0, 0.01, 0.1]:
        m = LogisticRegressionScratch(l2=l2).fit(hours_2d, PASSED)
        probs = m.predict_proba(np.array([[3.0], [4.0], [5.0]]))
        print(f"  {l2:>6} | {m.coef_[0]:>7.2f} | {probs[0]:>6.2f} "
              f"| {probs[1]:>6.2f} | {probs[2]:>6.2f}")
    print("  → l2=0: weights (and confidence) explode on separable data.")
    print("    A little l2 = calibrated probabilities, like the theory doc's.\n")

    print("FULL WORKFLOW — 200 students, zero sklearn:")
    X, y = make_students()
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"  stratification check: train {y_train.mean():.0%} pass, "
          f"test {y_test.mean():.0%} pass (matched ratios)\n")

    model = LogisticRegressionScratch(l2=0.001).fit(X_train, y_train)
    print(f"  learned: study={model.coef_[0]:.2f}, sleep={model.coef_[1]:.2f}, "
          f"bias={model.intercept_:.2f}   (true: 1.1, 0.6, -9.0)\n")

    print(classification_report_scratch(
        y_test, model.predict(X_test), names=("fail", "pass")))

    # Plot: what L2 does to the curve, and to the weights
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    curve_x = np.linspace(0, 9, 200).reshape(-1, 1)
    for l2, style in [(0.0, "r-"), (0.01, "g--"), (0.1, "b-.")]:
        m = LogisticRegressionScratch(l2=l2).fit(hours_2d, PASSED)
        ax1.plot(curve_x, m.predict_proba(curve_x), style, label=f"l2={l2}")
    ax1.scatter(HOURS, PASSED, s=90, color="black", zorder=3, label="the 6 students")
    ax1.set_xlabel("Hours studied")
    ax1.set_ylabel("P(pass)")
    ax1.set_title("L2 tames the cliff: same data, three leash lengths")
    ax1.legend()

    # 30 points from 0.0001 to 1.0. Why stop at 1.0? The update includes
    # `w -= lr * l2 * w`; once lr×l2 gets large the decay step ITSELF
    # diverges — weights oscillate and overflow. (See exercise 4.)
    l2_values = np.logspace(-4, 0, 30)
    coef_sizes = [abs(LogisticRegressionScratch(epochs=5_000, l2=v)
                      .fit(hours_2d, PASSED).coef_[0]) for v in l2_values]
    ax2.plot(l2_values, coef_sizes, "o-", markersize=3)
    ax2.set_xscale("log")
    ax2.set_xlabel("l2 strength (log scale)")
    ax2.set_ylabel("|weight|")
    ax2.set_title("Stronger penalty → smaller weights → humbler model")

    fig.tight_layout()
    out = "logistic-regression/from_scratch_plot.png"
    fig.savefig(out, dpi=120)
    print(f"\n  plot saved → {out}")


if __name__ == "__main__":
    main()
