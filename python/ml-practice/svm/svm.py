"""Learn more Python by building a Support Vector Machine.

Part 6 — back to gradients, but with a new question. Logistic regression asked
"what probability?"; SVM asks "where is the WIDEST lane between the classes?"
  STEP 1: feature scaling          (the theory doc's email numbers, reproduced)
  STEP 2: the ±1 trick + hinge loss (learn: np.where as ternary, np.maximum)
  STEP 3: a scratch linear SVM      (subgradient descent; margin width = 2/||w||)
  STEP 4: kernels as VALUES         (learn: dicts of functions, Callable, assert)
  STEP 5: the kernel trick, visible (rings: linear fails → lift by hand → RBF)
  STEP 6: the C knob, measured      (support vectors counted at C = 0.01/1/100)

Theory companion: ../../ml/svm.md

Run from python/ml-practice/:
    uv run svm/svm.py
"""

from collections.abc import Callable

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ──────────────────────────────────────────────────────────────────────────────
# The theory doc's 4 emails: [word_count, exclamation_count], 1 = spam
# ──────────────────────────────────────────────────────────────────────────────

EMAILS = np.array([[350.0, 12.0], [200.0, 1.0], [500.0, 8.0], [300.0, 2.0]])
SPAM = np.array([1, 0, 1, 0])


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    return (X - mu) / sigma, mu, sigma


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Hinge loss: zero penalty OUTSIDE the lane, growing penalty inside.
# Labels must be ±1 (not 0/1!) so that  y * score > 1  means "comfortably right".
# ──────────────────────────────────────────────────────────────────────────────

def hinge_loss(scores: np.ndarray, y_pm: np.ndarray) -> float:
    """mean(max(0, 1 - y*score)) — np.maximum is ELEMENTWISE max, not Python max."""
    return float(np.maximum(0.0, 1.0 - y_pm * scores).mean())


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — A linear SVM is gradient descent on  lam/2·||w||² + hinge.
# Shrinking w widens the lane (width = 2/||w||); hinge pushes points out of it.
# ──────────────────────────────────────────────────────────────────────────────

class ScratchLinearSVM:
    def __init__(self, lr: float = 0.1, c: float = 1.0, epochs: int = 2000):
        self.lr = lr
        self.c = c                      # the C knob: high C = violations cost more
        self.epochs = epochs
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y_pm: np.ndarray):
        n, n_features = X.shape
        self.w, self.b = np.zeros(n_features), 0.0
        lam = 1.0 / self.c              # weak regularization when C is big

        for _ in range(self.epochs):
            margins = y_pm * (X @ self.w + self.b)
            mask = margins < 1          # ONLY lane-violators get a say (subgradient)
            grad_w = lam * self.w - (y_pm[mask] @ X[mask]) / n
            grad_b = -y_pm[mask].sum() / n
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def score_fn(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b      # sign = class, magnitude = confidence

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score_fn(X) >= 0).astype(int)

    def margin_width(self) -> float:
        return 2.0 / float(np.linalg.norm(self.w))

    def support_mask(self, X: np.ndarray, y_pm: np.ndarray) -> np.ndarray:
        # GD settles with the lane LEANING on its closest points (margin ≈ 1.03,
        # not exactly 1.0) — so "on the edge" needs a tolerance, not equality
        return y_pm * self.score_fn(X) <= 1.1


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Kernels are similarity FUNCTIONS — and functions are values in Python.
# ──────────────────────────────────────────────────────────────────────────────

def rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float = 0.5) -> float:
    """exp(-gamma * distance²): 1.0 = identical points, →0 = unrelated points."""
    return float(np.exp(-gamma * np.sum((a - b) ** 2)))


KERNELS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "linear": lambda a, b: float(a @ b),
    "rbf": rbf_kernel,
}


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 data — a ring of one class around the other: NO straight line works.
# ──────────────────────────────────────────────────────────────────────────────

def make_rings(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    half = n // 2
    radius = np.concatenate([rng.uniform(0.0, 1.0, half),     # inner blob
                             rng.uniform(1.6, 2.4, half)])    # outer ring
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    X = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    y = np.concatenate([np.zeros(half, dtype=int), np.ones(half, dtype=int)])
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 data — two heavily overlapping blobs: a perfect split does NOT exist,
# so the C tradeoff (wide lane vs training mistakes) actually matters.
# ──────────────────────────────────────────────────────────────────────────────

def make_blobs(n: int = 300, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    half = n // 2
    X = np.vstack([rng.normal(loc=[-1.0, 0.0], scale=1.0, size=(half, 2)),
                   rng.normal(loc=[+1.0, 0.0], scale=1.0, size=(half, 2))])
    y = np.concatenate([np.zeros(half, dtype=int), np.ones(half, dtype=int)])
    return X, y


def accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float((actual == predicted).mean())


def main() -> None:
    # STEP 1 — scale the 4 emails; reproduce the theory doc's table exactly
    print("STEP 1 — StandardScaler by hand (the theory doc's email numbers):")
    X_scaled, mu, sigma = standardize(EMAILS)
    print(f"    word_count_scaled: {np.round(X_scaled[:, 0], 1)}")
    print(f"    excl_count_scaled: {np.round(X_scaled[:, 1], 1)}")
    print("    → matches the doc: [0.1 -1.3 1.5 -0.3] and [1.4 -1.1 0.5 -0.8]\n")

    # STEP 2 — the ±1 trick and the hinge
    print("STEP 2 — labels become ±1, and the hinge loss only punishes the lane:")
    y_pm = np.where(SPAM == 1, 1, -1)          # vectorized ternary: 0/1 → -1/+1
    print(f"    labels 0/1 → ±1: {SPAM} → {y_pm}")
    for score, label in [(2.0, 1), (0.4, 1), (-0.7, 1)]:
        h = max(0.0, 1.0 - label * score)
        where = "outside the lane — free" if h == 0 else (
            "inside the lane" if score > 0 else "wrong side — expensive")
        print(f"    score {score:+.1f}, true +1 → hinge {h:.1f}   ({where})")
    print()

    # STEP 3 — train the scratch SVM on the 4 scaled emails
    print("STEP 3 — scratch linear SVM on the 4 scaled emails (c=100):")
    model = ScratchLinearSVM(lr=0.1, c=100.0, epochs=2000)
    model.fit(X_scaled, y_pm)
    print(f"    w = {np.round(model.w, 2)}, b = {model.b:+.2f}")
    print(f"    final hinge loss: {hinge_loss(model.score_fn(X_scaled), y_pm):.3f}"
          f"  → margin width 2/||w|| = {model.margin_width():.2f}")
    sv = model.support_mask(X_scaled, y_pm)
    print(f"    support vectors (on the lane edge): {np.round(X_scaled[sv], 1).tolist()}")

    new_email = (np.array([400.0, 5.0]) - mu) / sigma   # the doc's Step 5 email
    score = float(model.score_fn(new_email[None, :])[0])
    print(f"    new email scaled → {np.round(new_email, 1)}, "
          f"score {score:+.2f} → {'SPAM' if score >= 0 else 'not spam'}")

    # STEP 4 — sklearn agreement + kernels as dictionary values
    print("\nSTEP 4 — sklearn's exact answer, then kernels as plain functions:")
    sk = SVC(kernel="linear", C=100.0).fit(X_scaled, SPAM)
    sk_score = float(sk.decision_function(new_email[None, :])[0])
    print(f"    sklearn w = {np.round(sk.coef_[0], 2)}, b = {sk.intercept_[0]:+.2f}, "
          f"support vectors per class: {sk.n_support_}")
    print(f"    same new email, three solvers: doc -0.04, scratch {score:+.2f}, "
          f"sklearn {sk_score:+.4f}")
    print("    → all ≈ 0: the point sits ON the boundary. The sign is noise; "
          "the tiny magnitude is the real answer")

    a, b_pt, c_pt = np.array([1.0, 2.0]), np.array([1.0, 3.0]), np.array([9.0, 9.0])
    k_near = KERNELS["rbf"](a, b_pt)           # look the FUNCTION up, then call it
    k_far = KERNELS["rbf"](a, c_pt)
    assert round(k_near, 2) == 0.61            # the theory doc's arithmetic, verified
    assert k_far < 1e-20
    print(f"    K([1,2],[1,3]) = {k_near:.2f} (similar), "
          f"K([1,2],[9,9]) = {k_far:.1e} (unrelated) — asserts passed\n")

    # STEP 5 — the kernel trick made visible on the rings
    print("STEP 5 — the ring dataset: lift it yourself, then let RBF do it:")
    X_ring, y_ring = make_rings()
    y_ring_pm = np.where(y_ring == 1, 1, -1)

    X_ring_std, _, _ = standardize(X_ring)
    flat = ScratchLinearSVM(c=1.0).fit(X_ring_std, y_ring_pm)
    print(f"    linear SVM on raw 2D rings:        "
          f"{accuracy(y_ring, flat.predict(X_ring_std)):.0%}  (a line can't cut a ring)")

    lifted = np.column_stack([X_ring, (X_ring ** 2).sum(axis=1)])  # height = dist²
    X_lift_std, _, _ = standardize(lifted)
    lifted_model = ScratchLinearSVM(c=1.0).fit(X_lift_std, y_ring_pm)
    print(f"    SAME linear SVM + your 3rd axis:   "
          f"{accuracy(y_ring, lifted_model.predict(X_lift_std)):.0%}  (flat plane in 3D)")

    rbf_svc = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X_ring, y_ring)
    print(f"    sklearn RBF on raw 2D (no lift!):  "
          f"{accuracy(y_ring, rbf_svc.predict(X_ring)):.0%}  (the kernel lifts implicitly)\n")

    # STEP 6 — the C knob on data with NO perfect answer
    print("STEP 6 — C on overlapping blobs (wide lane vs training mistakes):")
    X_blob, y_blob = make_blobs()
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X_blob, y_blob, test_size=0.25, random_state=42)
    for c_value in [0.01, 1.0, 100.0]:
        svc = SVC(kernel="rbf", C=c_value, gamma="scale").fit(Xb_train, yb_train)
        n_sv = int(svc.n_support_.sum())
        print(f"    C={c_value:<6} train {accuracy(yb_train, svc.predict(Xb_train)):.0%}  "
              f"test {accuracy(yb_test, svc.predict(Xb_test)):.0%}  "
              f"support vectors {n_sv:>3}/{len(yb_train)}")
    print("    → low C: the lane swallows nearly ALL the data (underfit); "
          "high C buys train accuracy that test never sees\n")

    # Plot: the widest lane (left) and the RBF ring boundary (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    spam_mask = SPAM == 1
    ax1.scatter(*X_scaled[spam_mask].T, marker="x", s=120, color="crimson", label="spam")
    ax1.scatter(*X_scaled[~spam_mask].T, marker="o", s=80, color="steelblue", label="not spam")
    ax1.scatter(*X_scaled[sv].T, s=300, facecolors="none", edgecolors="black",
                label="support vectors")
    ax1.scatter(*new_email, marker="*", s=200, color="darkorange", label="new email")
    x1_line = np.linspace(-1.6, 1.8, 50)
    for level, style in [(0.0, "-"), (1.0, "--"), (-1.0, "--")]:
        x2_line = (level - model.b - model.w[0] * x1_line) / model.w[1]
        ax1.plot(x1_line, x2_line, style, color="gray")
    ax1.set_xlabel("word_count (scaled)")
    ax1.set_ylabel("excl_count (scaled)")
    ax1.set_title("The widest lane: boundary (solid) and margin edges (dashed)")
    ax1.legend(loc="lower right")

    # meshgrid → ravel → predict → reshape: the standard boundary-plot dance
    xx, yy = np.meshgrid(np.linspace(-2.7, 2.7, 300), np.linspace(-2.7, 2.7, 300))
    grid = np.column_stack([xx.ravel(), yy.ravel()])     # (90000, 2) of pixels
    zz = rbf_svc.decision_function(grid).reshape(xx.shape)
    ax2.contourf(xx, yy, zz, levels=[zz.min(), 0.0, zz.max()],
                 colors=["lightsteelblue", "mistyrose"], alpha=0.6)
    ax2.contour(xx, yy, zz, levels=[0.0], colors="gray")
    inner = y_ring == 0
    ax2.scatter(*X_ring[inner].T, marker="o", s=25, color="steelblue", label="inner")
    ax2.scatter(*X_ring[~inner].T, marker="x", s=35, color="crimson", label="ring")
    ax2.set_title("RBF kernel: a 'straight' boundary in lifted space is a ring here")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    out = "svm/svm_plot.png"
    fig.savefig(out, dpi=120)
    print(f"    plot saved → {out}")


if __name__ == "__main__":
    main()
