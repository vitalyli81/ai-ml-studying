"""Learn more Python by building logistic regression.

Builds on linear-regression/ — same learning loop, different loss. New here:
  STEP 1-2: the sigmoid                  (learn: one function for scalars AND arrays)
  STEP 3:   log loss                     (learn: np.clip, why not MSE)
  STEP 4:   the classifier               (learn: boolean masks, .astype, defaults)
  STEP 5:   metrics from scratch         (learn: dicts as counters, conditional expressions)
  STEP 6:   sklearn + threshold tuning   (learn: 2D column slicing, np.linspace)

Theory companion: ../../ml/logistic-regression.md (same 6 students, same math).

Run from python/ml-practice/:
    uv run logistic-regression/logistic_regression.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — The data: 6 students from ml/logistic-regression.md
# ──────────────────────────────────────────────────────────────────────────────

HOURS = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 8.0])
PASSED = np.array([0, 0, 0, 1, 1, 1])  # 0 = fail, 1 = pass


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — The sigmoid: squash any number into (0, 1)
# ──────────────────────────────────────────────────────────────────────────────

def sigmoid(z: float | np.ndarray) -> np.ndarray:
    """σ(z) = 1 / (1 + e^(-z)).

    Because np.exp broadcasts, ONE definition works for a single number
    AND a whole array — no overloads, no separate loop version.
    """
    return 1.0 / (1.0 + np.exp(-z))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Log loss: the score that punishes confident wrong answers
# ──────────────────────────────────────────────────────────────────────────────

def log_loss(actual: np.ndarray, prob: np.ndarray) -> float:
    """-(y·log(p) + (1-y)·log(1-p)), averaged.

    np.clip pins probabilities into [1e-15, 1-1e-15] first — log(0) is -inf,
    and one perfectly-confident prediction would otherwise nuke the average.
    """
    p = np.clip(prob, 1e-15, 1 - 1e-15)
    return float(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — The classifier: same training loop as linear regression, new loss
# ──────────────────────────────────────────────────────────────────────────────

class ScratchLogisticRegression:
    """Linear regression's loop with a sigmoid on top.

    The gradient works out to the SAME shape as linear regression's:
    mean(error × x) — except 'error' is now (probability − label).
    """

    def __init__(self, lr: float = 0.5, epochs: int = 5_000):
        self.lr = lr
        self.epochs = epochs
        self.coef_: float | None = None
        self.intercept_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ScratchLogisticRegression":
        mu, sigma = x.mean(), x.std()
        x_std = (x - mu) / sigma  # scale → gradient descent converges fast

        w, b = 0.0, 0.0
        for _ in range(self.epochs):
            p = sigmoid(w * x_std + b)
            error = p - y                      # how over/under-confident per student
            w -= self.lr * (error * x_std).mean()
            b -= self.lr * error.mean()

        self.coef_ = float(w / sigma)          # back to raw "per hour" units
        self.intercept_ = float(b - w * mu / sigma)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """The probability of passing — the model's real output."""
        return sigmoid(self.coef_ * x + self.intercept_)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Probabilities → hard 0/1 labels, via a boolean mask."""
        return (self.predict_proba(x) >= threshold).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Metrics from scratch: the accuracy trap, exposed
# ──────────────────────────────────────────────────────────────────────────────

def accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction correct. (actual == predicted) is a boolean array;
    .mean() treats True as 1 and False as 0 — accuracy in one expression."""
    return float((actual == predicted).mean())


def confusion_counts(actual: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    """Count TP/FP/TN/FN with a dict as a counter."""
    counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for a, p in zip(actual, predicted):
        key = ("T" if a == p else "F") + ("P" if p == 1 else "N")
        counts[key] += 1
    return counts


def precision(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Of everything I flagged positive, how much really was?"""
    c = confusion_counts(actual, predicted)
    flagged = c["TP"] + c["FP"]
    return c["TP"] / flagged if flagged else 0.0   # conditional expression

def recall(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Of all actual positives, how many did I catch?"""
    c = confusion_counts(actual, predicted)
    positives = c["TP"] + c["FN"]
    return c["TP"] / positives if positives else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Realistic data + the sklearn workflow
# ──────────────────────────────────────────────────────────────────────────────

def make_students(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """200 students: pass probability driven by study + sleep hours."""
    rng = np.random.default_rng(seed)
    study = rng.uniform(0, 10, size=n)
    sleep = rng.uniform(3, 9, size=n)
    p_pass = sigmoid(1.1 * study + 0.6 * sleep - 9.0)   # the true process
    passed = (rng.random(n) < p_pass).astype(int)       # probabilistic outcomes
    return np.column_stack([study, sleep]), passed


def main() -> None:
    print("STEP 2 — the sigmoid squashes any score into a probability:")
    for z in [-4.0, -1.0, 0.0, 1.0, 4.0]:
        print(f"  sigmoid({z:+.0f}) = {sigmoid(z):.3f}")
    print()

    print("STEP 4 — train on the 6 students from ml/logistic-regression.md:")
    model = ScratchLogisticRegression().fit(HOURS, PASSED)
    print(f"  P(pass) = sigmoid({model.coef_:.2f} × hours + {model.intercept_:.2f})")
    for h in [3.0, 4.0, 5.0]:
        p = model.predict_proba(np.array([h]))[0]
        verdict = "PASS" if p >= 0.5 else "FAIL"        # conditional expression
        print(f"  {h:.0f} hours → P(pass) = {p:.2f} → {verdict}")
    print("  → same shape as the theory doc: ~3h fails, ~5h passes, 4h borderline\n")

    print("STEP 5 — the accuracy trap (why ml/model-evaluation.md exists):")
    sick = np.array([1] * 5 + [0] * 95)                  # 5% positive, like fraud
    lazy = np.zeros(100, dtype=int)                      # model that always says 0
    print(f"  'always predict 0' on 95/5 data → accuracy={accuracy(sick, lazy):.0%}, "
          f"recall={recall(sick, lazy):.0%}  ← 95% accurate, catches nothing\n")

    print("STEP 6 — sklearn on 200 students (study + sleep hours):")
    X, y = make_students()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    clf = LogisticRegression().fit(X_train, y_train)
    print(f"  learned weights: study={clf.coef_[0][0]:.2f}, sleep={clf.coef_[0][1]:.2f}, "
          f"bias={clf.intercept_[0]:.2f}   (true: 1.1, 0.6, -9.0)")
    print(classification_report(y_test, clf.predict(X_test), target_names=["fail", "pass"]))

    # Threshold tuning — the probability is the product; 0.5 is just a default
    probs = clf.predict_proba(X_test)[:, 1]             # column 1 = P(pass)
    print("  threshold sweep (our own metrics, sklearn's probabilities):")
    for t in [0.3, 0.5, 0.7]:
        preds = (probs >= t).astype(int)
        print(f"    threshold {t:.1f} → precision {precision(y_test, preds):.2f}, "
              f"recall {recall(y_test, preds):.2f}")
    print("  → lower threshold catches more passes (recall↑) at the cost of "
          "false alarms (precision↓)\n")

    # Plot: the scratch model's sigmoid + the precision/recall tradeoff
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    curve_x = np.linspace(0, 9, 200)
    ax1.plot(curve_x, model.predict_proba(curve_x), "r-", linewidth=2,
             label="P(pass) — scratch model")
    ax1.scatter(HOURS, PASSED, s=90, color="black", zorder=3, label="the 6 students")
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="threshold 0.5")
    ax1.set_xlabel("Hours studied")
    ax1.set_ylabel("P(pass)")
    ax1.set_title("Logistic regression = a sigmoid fitted to 0/1 outcomes")
    ax1.legend()

    thresholds = np.linspace(0.05, 0.95, 50)
    ax2.plot(thresholds, [precision(y_test, (probs >= t).astype(int)) for t in thresholds],
             label="precision")
    ax2.plot(thresholds, [recall(y_test, (probs >= t).astype(int)) for t in thresholds],
             label="recall")
    ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Decision threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("The threshold trades precision against recall")
    ax2.legend()

    fig.tight_layout()
    out = "logistic-regression/logistic_plot.png"
    fig.savefig(out, dpi=120)
    print(f"  plot saved → {out}")


if __name__ == "__main__":
    main()
