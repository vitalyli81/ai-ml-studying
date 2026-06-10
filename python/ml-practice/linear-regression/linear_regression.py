"""Learn Python by building linear regression — three ways.

The same model, built three times:
  STEP 1-3: pure Python      (learn: lists, functions, loops, zip)
  STEP 4:   NumPy            (learn: arrays, vectorization, broadcasting)
  STEP 5:   a class          (learn: classes, methods, the sklearn API shape)
  STEP 6:   scikit-learn     (learn: the real-world workflow + plotting)

Theory companion: ../../ml/linear-regression.md (same dataset, same math).

Run from python/ml-practice/:
    uv run linear-regression/linear_regression.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window needed
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — The data, as plain Python lists
# (Same 5 houses as ml/linear-regression.md — true fit ≈ 218 × sqft + 23,607)
# ──────────────────────────────────────────────────────────────────────────────

SQFT = [600.0, 800.0, 1000.0, 1200.0, 1500.0]
PRICE = [150_000.0, 200_000.0, 250_000.0, 280_000.0, 350_000.0]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Pure-Python building blocks: functions, list comprehensions, zip
# ──────────────────────────────────────────────────────────────────────────────

def mean(values: list[float]) -> float:
    """Average of a list. sum() and len() are built-ins — no imports needed."""
    return sum(values) / len(values)


def predict(x: list[float], w: float, b: float) -> list[float]:
    """The model itself: y = w*x + b for every x. One list comprehension."""
    return [w * xi + b for xi in x]


def mse(actual: list[float], predicted: list[float]) -> float:
    """Mean squared error — how wrong are we, on average (squared)?"""
    return mean([(a - p) ** 2 for a, p in zip(actual, predicted)])


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Gradient descent in pure Python: the learning loop
# ──────────────────────────────────────────────────────────────────────────────

def train_pure_python(
    x: list[float],
    y: list[float],
    lr: float = 1e-8,  # tiny: raw sqft values make gradients huge (see STEP 4)
    epochs: int = 200,
) -> tuple[float, float]:
    """Start with a terrible line (w=0, b=0) and nudge it downhill."""
    w, b = 0.0, 0.0
    for epoch in range(epochs):
        preds = predict(x, w, b)
        errors = [yi - pi for yi, pi in zip(y, preds)]

        # The gradients tell each parameter which way to move (and how hard)
        grad_w = -2 * mean([e * xi for e, xi in zip(errors, x)])
        grad_b = -2 * mean(errors)

        w -= lr * grad_w  # step downhill, opposite the gradient
        b -= lr * grad_b

        if epoch % 50 == 0:
            print(f"  epoch {epoch:>3}: w={w:7.2f}  b={b:.4f}  mse={mse(y, preds):,.0f}")
    return w, b


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — NumPy: vectorize, standardize, converge in a fraction of the steps
# ──────────────────────────────────────────────────────────────────────────────

def train_numpy(
    x: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 500
) -> tuple[float, float]:
    """Same algorithm, two upgrades:

    1. Vectorized — `errors * x` multiplies whole arrays at once (no loops).
    2. Standardized — scaling x to mean 0 / std 1 tames the gradients, so we
       can use lr=0.1 instead of 1e-8 and converge in hundreds of steps.
    """
    mu, sigma = x.mean(), x.std()
    x_std = (x - mu) / sigma  # broadcasting: one expression, whole array

    w, b = 0.0, 0.0
    for _ in range(epochs):
        errors = y - (w * x_std + b)
        w -= lr * (-2 * (errors * x_std).mean())
        b -= lr * (-2 * errors.mean())

    # We trained on standardized x — convert weights back to raw-sqft units
    slope = w / sigma
    intercept = b - w * mu / sigma
    return slope, intercept


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Wrap it in a class with the scikit-learn API shape (fit / predict)
# ──────────────────────────────────────────────────────────────────────────────

class ScratchLinearRegression:
    """Our model, packaged the way every sklearn estimator is packaged."""

    def __init__(self, lr: float = 0.1, epochs: int = 500):
        self.lr = lr
        self.epochs = epochs
        self.coef_: float | None = None       # sklearn convention: trailing _
        self.intercept_: float | None = None  # = "learned during fit()"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ScratchLinearRegression":
        self.coef_, self.intercept_ = train_numpy(x, y, self.lr, self.epochs)
        return self  # returning self enables model.fit(x, y).predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict()")
        return self.coef_ * x + self.intercept_

    def __repr__(self) -> str:  # what print(model) shows
        return f"ScratchLinearRegression(lr={self.lr}, epochs={self.epochs})"


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — The real-world workflow: scikit-learn + train/test split + a plot
# ──────────────────────────────────────────────────────────────────────────────

def realistic_dataset(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """200 synthetic houses around the same trend, with realistic noise."""
    rng = np.random.default_rng(seed)
    sqft = rng.uniform(500, 2500, size=n)
    noise = rng.normal(0, 25_000, size=n)
    price = 218.0 * sqft + 23_607 + noise
    return sqft, price


def main() -> None:
    print("STEP 3 — pure-Python gradient descent (raw features, tiny lr):")
    w, b = train_pure_python(SQFT, PRICE)
    print(f"  after 200 epochs: price ≈ {w:.1f} × sqft + {b:.2f}")
    print("  → it learns, but slowly, and b has barely moved. NumPy next.\n")

    x = np.array(SQFT)
    y = np.array(PRICE)

    print("STEP 4 — NumPy gradient descent (standardized, lr=0.1, 500 epochs):")
    slope, intercept = train_numpy(x, y)
    print(f"  price ≈ {slope:.1f} × sqft + {intercept:,.0f}")
    print("  → matches the closed-form answer from ml/linear-regression.md\n")

    print("STEP 5 — same thing, as a class with the sklearn API shape:")
    scratch = ScratchLinearRegression().fit(x, y)
    print(f"  {scratch}")
    print(f"  predict(1100 sqft) → ${scratch.predict(np.array([1100.0]))[0]:,.0f}\n")

    print("STEP 6 — scikit-learn on 200 noisy houses (the real workflow):")
    sqft, price = realistic_dataset()
    X = sqft.reshape(-1, 1)  # sklearn wants 2D: (n_samples, n_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, price, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))

    print(f"  sklearn:  price ≈ {model.coef_[0]:.1f} × sqft + {model.intercept_:,.0f}")
    print(f"  R² on the held-out test set: {r2:.3f}")
    print(f"  predict(1100 sqft) → ${model.predict([[1100.0]])[0]:,.0f}\n")

    # Plot: the data, our scratch model, and sklearn's fit
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sqft, price, s=12, alpha=0.4, label="houses (synthetic)")
    line_x = np.array([500.0, 2500.0])
    ax.plot(line_x, scratch.predict(line_x), "g--", linewidth=2,
            label=f"scratch GD (5 houses): {scratch.coef_:.0f}·x + {scratch.intercept_:,.0f}")
    ax.plot(line_x, model.predict(line_x.reshape(-1, 1)), "r-", linewidth=2,
            label=f"sklearn (200 houses): {model.coef_[0]:.0f}·x + {model.intercept_:,.0f}")
    ax.scatter(SQFT, PRICE, s=80, color="black", zorder=3, label="original 5 houses")
    ax.set_xlabel("Square feet")
    ax.set_ylabel("Price ($)")
    ax.set_title("Linear regression: from-scratch gradient descent vs scikit-learn")
    ax.legend()
    fig.tight_layout()
    out = "linear-regression/regression_plot.png"
    fig.savefig(out, dpi=120)
    print(f"  plot saved → {out}")


if __name__ == "__main__":
    main()
