"""Learn more Python by building PCA.

Part 10 — the last algorithm of the classical track, and the only one that
predicts NOTHING: PCA re-describes your data with fewer numbers. Find the
directions the data is most spread along, keep those, drop the rest.
  STEP 1: the tilted blob            (find the diagonal ruler by hand)
  STEP 2: the eigen-machinery        (np.cov + np.linalg.eigh on real iris)
  STEP 3: sklearn agreement          (...up to SIGN — eigenvectors don't care)
  STEP 4: the unscaled disaster      (salary hijacks PC1, measured)
  STEP 5: digits 64 → enough-for-95% (np.cumsum + np.searchsorted, timed)
  STEP 6: reconstruction             (the JPEG quality slider, on real images)

Theory companion: ../../ml/pca.md

Run from python/ml-practice/:
    uv run pca/pca.py
"""

from time import perf_counter

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# ──────────────────────────────────────────────────────────────────────────────
# The whole algorithm: covariance matrix → eigendecomposition → sort → project.
# np.linalg.eigh is for symmetric matrices (a covariance matrix always is):
# it's faster than eig, guarantees real numbers, and returns eigenvalues
# in ASCENDING order — hence the argsort[::-1] to put the longest ruler first.
# ──────────────────────────────────────────────────────────────────────────────

def pca_scratch(X_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = np.cov(X_std.T)                          # how every pair co-varies
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]          # biggest spread first
    return eigenvalues[order], eigenvectors[:, order]   # columns = components


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0        # constant features (blank pixels): don't divide by 0
    return (X - mu) / sigma, mu, sigma


def make_people(n: int = 200, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height = rng.normal(170, 10, size=n)
    weight = 0.9 * (height - 170) + 70 + rng.normal(0, 4, size=n)  # correlated!
    return np.column_stack([height, weight])


def main() -> None:
    # STEP 1 — the doc's tilted blob: which ruler is the data longest along?
    print("STEP 1 — 200 people, height vs weight (the doc's tilted blob):")
    people = make_people()
    people_std, _, _ = standardize(people)
    eigenvalues, components = pca_scratch(people_std)
    ratio = eigenvalues / eigenvalues.sum()
    print(f"    PC1 direction {np.round(components[:, 0], 2)} carries "
          f"{ratio[0]:.0%} of the spread")
    print(f"    PC2 direction {np.round(components[:, 1], 2)} carries "
          f"{ratio[1]:.0%}")
    shadow = people_std[0] @ components[:, 0]      # project = one dot product
    print(f"    person 0: (height, weight) = {np.round(people_std[0], 2)} "
          f"→ shadow on PC1 = {shadow:.2f}")
    print("    → the diagonal 'body size' ruler, found by eigh — two numbers "
          "become one\n")

    # STEP 2 — the eigen-machinery on iris (the series' first REAL dataset)
    print("STEP 2 — iris (150 real flowers, 4 features), scratch PCA:")
    iris = load_iris()
    X_iris, _, _ = standardize(iris.data)
    eigenvalues, components = pca_scratch(X_iris)
    ratio = eigenvalues / eigenvalues.sum()
    print(f"    explained variance ratio: {np.round(ratio, 2)}")
    assert np.allclose(np.round(ratio, 2), [0.73, 0.23, 0.04, 0.01])
    print(f"    PC1+PC2 = {ratio[:2].sum():.0%} — the doc's '2 components "
          "explain 96%', from your own eigh\n")

    # STEP 3 — sklearn agreement... up to sign
    print("STEP 3 — sklearn PCA on the same flowers:")
    sk = PCA().fit(X_iris)
    assert np.allclose(ratio, sk.explained_variance_ratio_)
    ours = X_iris @ components[:, :2]
    theirs = sk.transform(X_iris)[:, :2]
    flipped = int((np.sign(ours[0]) != np.sign(theirs[0])).sum())
    print(f"    variance ratios: identical (allclose). Projections: "
          f"{flipped} of 2 components came out sign-FLIPPED")
    signs = np.sign((ours * theirs).sum(axis=0))   # align each column's sign
    assert np.allclose(ours * signs, theirs)
    print("    → an eigenvector and its negative are the SAME ruler pointing "
          "the other way;")
    print("      align signs, then scratch == sklearn exactly\n")

    # STEP 4 — the unscaled disaster, measured
    print("STEP 4 — skip the scaler and salary hijacks PC1 (the doc's gotcha):")
    rng = np.random.default_rng(7)
    age = rng.uniform(20, 65, size=300)
    salary = 2_000 * age + rng.normal(0, 15_000, size=300)
    jobs = np.column_stack([salary, age])
    raw_vals, raw_vecs = pca_scratch(jobs - jobs.mean(axis=0))   # centered only
    raw_ratio = raw_vals / raw_vals.sum()
    print(f"    unscaled: PC1 = {np.round(raw_vecs[:, 0], 4)} carrying "
          f"{raw_ratio[0]:.2%}  ← pure salary, age invisible")
    jobs_std, _, _ = standardize(jobs)
    std_vals, std_vecs = pca_scratch(jobs_std)
    std_ratio = std_vals / std_vals.sum()
    print(f"    scaled:   PC1 = {np.round(std_vecs[:, 0], 2)} carrying "
          f"{std_ratio[0]:.0%}  ← both features get a vote\n")

    # STEP 5 — digits: how many of 64 pixel-features carry 95% of the variance?
    print("STEP 5 — handwritten digits (1,797 images, 64 pixels each):")
    digits = load_digits()
    X_dig, mu_dig, sigma_dig = standardize(digits.data)
    eigenvalues_dig, components_dig = pca_scratch(X_dig)
    cumvar = np.cumsum(eigenvalues_dig / eigenvalues_dig.sum())
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1   # first index where ≥ 0.95
    for n in [5, 10, 20, n95, 60]:
        marker = "  ← 95% reached" if n == n95 else ""
        print(f"    {n:>2} components: {cumvar[n - 1]:.1%} of variance{marker}")

    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42)
    Xtr_std, mu, sigma = standardize(X_train)
    Xte_std = (X_test - mu) / sigma                # fit on train, transform test

    t0 = perf_counter()
    plain = SVC(kernel="rbf", C=10).fit(Xtr_std, y_train)
    acc_plain = plain.score(Xte_std, y_test)
    t_plain = perf_counter() - t0

    t0 = perf_counter()
    vals_tr, comps_tr = pca_scratch(Xtr_std)       # PCA fitted on TRAIN only
    Ztr, Zte = Xtr_std @ comps_tr[:, :n95], Xte_std @ comps_tr[:, :n95]
    reduced = SVC(kernel="rbf", C=10).fit(Ztr, y_train)
    acc_reduced = reduced.score(Zte, y_test)
    t_reduced = perf_counter() - t0

    print(f"    SVC on all 64 features:    {acc_plain:.1%} in {t_plain:.2f}s")
    print(f"    SVC on {n95} components:     {acc_reduced:.1%} in {t_reduced:.2f}s "
          "(including the PCA itself)")
    print("    → same accuracy story as the doc: keep 95% of the variance, "
          "lose ~nothing downstream\n")

    # STEP 6 — reconstruction: project down, multiply back up
    print("STEP 6 — the JPEG slider: one digit rebuilt from k components:")
    digit = X_dig[0]                               # standardized '0' image
    panels = []
    for k in [64, n95, 10, 3]:
        z = digit @ components_dig[:, :k]          # compress: 64 → k numbers
        back = z @ components_dig[:, :k].T         # decompress: k → 64
        image = (back * sigma_dig + mu_dig).reshape(8, 8)   # un-standardize
        panels.append(image)
        err = float(np.abs(back - digit).mean())
        print(f"    k={k:>2}: stored {k:>2}/64 numbers, "
              f"mean reconstruction error {err:.3f}")
    print("    → 3 numbers still look like a zero: most pixels were redundant\n")

    # Plot: the rulers (left), the variance curve (middle), the slider (right)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    eigenvalues_p, components_p = pca_scratch(people_std)
    ax1.scatter(*people_std.T, s=14, alpha=0.5, color="steelblue")
    for j, color in [(0, "crimson"), (1, "darkorange")]:
        arrow = components_p[:, j] * 2.2 * np.sqrt(eigenvalues_p[j])
        ax1.annotate("", xy=arrow, xytext=(0, 0),
                     arrowprops=dict(color=color, width=2, headwidth=10))
        ax1.text(*(arrow * 1.25), f"PC{j + 1}", color=color, fontsize=12,
                 ha="center")
    ax1.set_xlabel("height (standardized)")
    ax1.set_ylabel("weight (standardized)")
    ax1.set_title("The rulers eigh found: long PC1, thin PC2")
    ax1.set_aspect("equal")

    ax2.plot(range(1, 65), cumvar, "o-", markersize=3, color="steelblue")
    ax2.axhline(0.95, color="gray", linestyle="--")
    ax2.axvline(n95, color="crimson", linestyle="--",
                label=f"{n95} components → 95%")
    ax2.set_xlabel("number of components kept")
    ax2.set_ylabel("cumulative variance retained")
    ax2.set_title("Digits: the compression curve")
    ax2.legend(loc="lower right")

    strip = np.hstack([np.pad(p, 1, constant_values=16) for p in panels])
    ax3.imshow(strip, cmap="gray_r")
    ax3.set_xticks([]), ax3.set_yticks([])
    ax3.set_title(f"Same digit from 64 / {n95} / 10 / 3 components")

    fig.tight_layout()
    out = "pca/pca_plot.png"
    fig.savefig(out, dpi=120)
    print(f"    plot saved → {out}")


if __name__ == "__main__":
    main()
