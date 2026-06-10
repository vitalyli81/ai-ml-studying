"""Linear regression FROM SCRATCH — NumPy + pandas allowed, scikit-learn banned.

Everything sklearn did for us in linear_regression.py is implemented by hand here:
  STEP 2: explore the data with pandas      (learn: DataFrame, describe, corr)
  STEP 3: the metrics — MSE, MAE, R²        (learn: NumPy reductions; what R² IS)
  STEP 4: the closed-form solution          (learn: slope = cov/var, derived)
  STEP 5: train/test split                  (learn: permutation + fancy indexing)
  STEP 6: gradient descent w/ early stop    (learn: @dataclass, break, loss history)

Theory companion: ../../ml/linear-regression.md
sklearn version (for comparison after): linear_regression.py

Run from python/ml-practice/:
    uv run linear-regression/linear_regression_from_scratch.py
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Generate data with seeded randomness (reproducible "houses")
# ──────────────────────────────────────────────────────────────────────────────

# The same 5 verification houses as ml/linear-regression.md
SQFT_5 = np.array([600.0, 800.0, 1000.0, 1200.0, 1500.0])
PRICE_5 = np.array([150_000.0, 200_000.0, 250_000.0, 280_000.0, 350_000.0])


def make_dataset(n: int = 200, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic houses: true process is 218·sqft + 23,607, plus noise."""
    rng = np.random.default_rng(seed)  # seeded → same "random" data every run
    sqft = rng.uniform(500, 2500, size=n)
    price = 218.0 * sqft + 23_607 + rng.normal(0, 25_000, size=n)
    return sqft, price


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Metrics from scratch (what sklearn.metrics was doing)
# ──────────────────────────────────────────────────────────────────────────────

def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean squared error — big misses hurt quadratically."""
    return float(np.mean((actual - predicted) ** 2))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error — same units as the target, robust to outliers."""
    return float(np.mean(np.abs(actual - predicted)))


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """R² = 1 − (error left over) / (total variation in the data).

    'What fraction of the price variation does my line explain?'
    1.0 = perfect; 0.0 = no better than always predicting the mean.
    """
    ss_residual = np.sum((actual - predicted) ** 2)
    ss_total = np.sum((actual - actual.mean()) ** 2)
    return float(1 - ss_residual / ss_total)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — The closed-form solution (what sklearn's LinearRegression computes)
# ──────────────────────────────────────────────────────────────────────────────

def fit_closed_form(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least squares has an exact answer for one feature:

        slope = covariance(x, y) / variance(x)
        intercept = mean(y) − slope · mean(x)   (the line passes through the means)

    No iteration, no learning rate — calculus already solved this problem.
    Gradient descent (STEP 6) earns its keep on problems with NO closed form.
    """
    x_dev = x - x.mean()  # deviations from the mean
    y_dev = y - y.mean()
    slope = np.sum(x_dev * y_dev) / np.sum(x_dev**2)
    intercept = y.mean() - slope * x.mean()
    return float(slope), float(intercept)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train/test split from scratch (what sklearn.model_selection did)
# ──────────────────────────────────────────────────────────────────────────────

def train_test_split(
    x: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle indices, cut once, index both arrays with the same shuffle."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(x))      # e.g. [183, 7, 42, ...] — shuffled indices
    cut = int(len(x) * (1 - test_ratio))
    train_idx, test_idx = shuffled[:cut], shuffled[cut:]   # slicing
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]  # fancy indexing


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Gradient descent with convergence detection (a @dataclass model)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScratchLinearRegression:
    """Linear regression trained by gradient descent.

    @dataclass auto-generates __init__ and __repr__ from these fields —
    compare with the hand-written class in linear_regression.py.
    """

    lr: float = 0.1
    max_epochs: int = 10_000
    tol: float = 1e-12               # stop when MSE improves less than this (relative)
    slope_: float | None = None      # trailing _ = learned during fit()
    intercept_: float | None = None
    history_: list[float] = field(default_factory=list, repr=False)  # MSE per epoch
    epochs_run_: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ScratchLinearRegression":
        # Standardize x so one learning rate works regardless of feature scale
        mu, sigma = x.mean(), x.std()
        x_std = (x - mu) / sigma

        w, b = 0.0, 0.0
        prev_loss = float("inf")
        self.history_ = []

        for epoch in range(1, self.max_epochs + 1):
            errors = y - (w * x_std + b)
            w -= self.lr * (-2 * (errors * x_std).mean())
            b -= self.lr * (-2 * errors.mean())

            loss = mse(y, w * x_std + b)
            self.history_.append(loss)

            # Converged? Stop early instead of burning the remaining epochs.
            # (Scale the threshold by `loss`, not `prev_loss` — prev starts at
            # inf, and tol × inf = inf would "converge" instantly on epoch 1.)
            if abs(prev_loss - loss) <= self.tol * max(loss, 1.0):
                break
            prev_loss = loss

        self.epochs_run_ = epoch
        # Trained on standardized x — convert the weights back to raw units
        self.slope_ = float(w / sigma)
        self.intercept_ = float(b - w * mu / sigma)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.slope_ is None:
            raise RuntimeError("Call fit() before predict()")
        return self.slope_ * x + self.intercept_


# ──────────────────────────────────────────────────────────────────────────────
# Putting it all together
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # STEP 2 — look at the data BEFORE modeling (pandas)
    sqft, price = make_dataset()
    df = pd.DataFrame({"sqft": sqft, "price": price})

    print("STEP 2 — explore with pandas:")
    print(df.head(3).round(0).to_string(index=False))
    print(f"\n  {len(df)} houses | mean price ${df['price'].mean():,.0f} "
          f"| sqft↔price correlation: {df['sqft'].corr(df['price']):.3f}\n")

    # STEP 4 — closed form on the 5 verification houses (matches the theory doc)
    print("STEP 4 — closed form on the 5 houses from ml/linear-regression.md:")
    slope, intercept = fit_closed_form(SQFT_5, PRICE_5)
    print(f"  price ≈ {slope:.1f} × sqft + {intercept:,.0f}   ← the exact answer\n")

    # STEP 6 — gradient descent should land on the SAME line, iteratively
    print("STEP 6 — gradient descent on the same 5 houses:")
    tiny = ScratchLinearRegression().fit(SQFT_5, PRICE_5)
    print(f"  {tiny}")  # the @dataclass wrote this __repr__ for us
    print(f"  price ≈ {tiny.slope_:.1f} × sqft + {tiny.intercept_:,.0f} "
          f"(converged in {tiny.epochs_run_} epochs)")
    print("  → same line as the closed form. GD works; it's just the slow road.\n")

    # The full workflow on realistic data — split, fit on train, judge on test
    print("FULL WORKFLOW — 200 noisy houses, all from-scratch parts:")
    x_train, x_test, y_train, y_test = train_test_split(sqft, price)
    model = ScratchLinearRegression().fit(x_train, y_train)
    preds = model.predict(x_test)

    report = {                                  # a dict — Python's object literal
        "slope": round(model.slope_, 1),
        "intercept": round(model.intercept_),
        "epochs_to_converge": model.epochs_run_,
        "test_mse": round(mse(y_test, preds)),
        "test_mae": round(mae(y_test, preds)),
        "test_r2": round(r_squared(y_test, preds), 3),
    }
    for key, value in report.items():           # .items() = Object.entries()
        print(f"  {key:>20}: {value:,}" if isinstance(value, int)
              else f"  {key:>20}: {value}")

    print(f"\n  predict(1100 sqft) → ${model.predict(np.array([1100.0]))[0]:,.0f}")

    # Plot: the fit + the loss curve (watch convergence happen)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(x_train, y_train, s=12, alpha=0.4, label="train")
    ax1.scatter(x_test, y_test, s=18, alpha=0.7, color="orange", label="test (held out)")
    line_x = np.array([500.0, 2500.0])
    ax1.plot(line_x, model.predict(line_x), "r-", linewidth=2,
             label=f"fit: {model.slope_:.0f}·x + {model.intercept_:,.0f}")
    ax1.set_xlabel("Square feet")
    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"From-scratch fit (test R² = {report['test_r2']})")
    ax1.legend()

    ax2.plot(model.history_)
    ax2.set_yscale("log")                       # loss falls over orders of magnitude
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE (log scale)")
    ax2.set_title(f"Gradient descent converging ({model.epochs_run_} epochs)")

    fig.tight_layout()
    out = "linear-regression/from_scratch_plot.png"
    fig.savefig(out, dpi=120)
    print(f"  plot saved → {out}")


if __name__ == "__main__":
    main()
