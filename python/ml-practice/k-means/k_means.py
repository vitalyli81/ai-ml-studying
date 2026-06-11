"""Learn more Python by building K-Means.

Part 9 — the first UNSUPERVISED algorithm. Notice what's missing from this file:
there is no y. No labels, no accuracy, no train/test split — the data sorts
itself, and judging the result becomes your job (inertia, elbow, silhouette).
  STEP 1: the 1D dance               (the doc's 1,2,9,10 → centers 1.5 / 9.5)
  STEP 2: the 6-customer trace       (every doc distance reproduced, asserted)
  STEP 3: inertia only falls         (and bad starts trap — n_init by hand)
  STEP 4: elbow + scratch silhouette (verified against sklearn with allclose)
  STEP 5: sklearn agreement          (same clusters, up to label renaming)
  STEP 6: profiling with pandas      (groupby — clusters are useless unnamed)

New Python: GENERATORS (yield), while True + break, the walrus operator :=,
np.array_equal, and pandas groupby.

Theory companion: ../../ml/k-means.md

Run from python/ml-practice/:
    uv run k-means/k_means.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.append(str(Path(__file__).parent.parent / "knn"))   # Part 4's trick,
from knn import pairwise_distances                           # Part 7's function


# ──────────────────────────────────────────────────────────────────────────────
# The scratch model. fit_steps is a GENERATOR: `yield` hands each round's state
# to the caller and PAUSES — the loop resumes only when the caller asks for
# the next round. You can watch the dance happen, one frame at a time.
# ──────────────────────────────────────────────────────────────────────────────

class ScratchKMeans:
    def __init__(self, k: int = 3, tol: float = 1e-9, seed: int | None = None):
        self.k = k
        self.tol = tol
        self.seed = seed

    def fit_steps(self, X: np.ndarray, init=None):
        rng = np.random.default_rng(self.seed)
        if init is None:                      # classic init: K random data points
            centroids = X[rng.choice(len(X), size=self.k, replace=False)].astype(float)
        else:
            centroids = np.array(init, dtype=float)

        while True:                           # loop until the dance stops
            dists = pairwise_distances(X, centroids)          # (n, k)
            labels = dists.argmin(axis=1)                     # ASSIGN
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                for j in range(self.k)])                      # UPDATE (mean wins)

            yield centroids.copy(), labels                    # hand out this frame

            # walrus := computes the shift AND names it, inside the if
            if (shift := float(np.linalg.norm(new_centroids - centroids))) < self.tol:
                break
            centroids = new_centroids

        self.centroids_, self.labels_ = new_centroids, labels
        self.inertia_ = float(((X - new_centroids[labels]) ** 2).sum())

    def fit(self, X: np.ndarray, init=None):
        for _ in self.fit_steps(X, init):     # drain the generator: run the dance
            pass                              # to the end without watching it
        return self


def inertia_of(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    return float(((X - centroids[labels]) ** 2).sum())


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — silhouette from scratch: for each point, a = "how far from my own
# cluster", b = "how far from the best OTHER cluster", score = (b-a)/max(a,b).
# ──────────────────────────────────────────────────────────────────────────────

def silhouette_scratch(X: np.ndarray, labels: np.ndarray) -> float:
    dists = pairwise_distances(X, X)                  # all pairs at once (Part 7)
    scores = []
    for i in range(len(X)):
        own = labels == labels[i]
        own[i] = False                                # don't count yourself
        a = dists[i][own].mean()
        b = min(dists[i][labels == other].mean()
                for other in set(labels) - {labels[i]})
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


# ──────────────────────────────────────────────────────────────────────────────
# 210 customers, NO labels — three blobs the algorithm must discover itself.
# ──────────────────────────────────────────────────────────────────────────────

def make_customers(n: int = 210, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    third = n // 3
    centers = np.array([[25.0, 75.0], [55.0, 30.0], [85.0, 70.0]])
    return np.vstack([rng.normal(c, 8.0, size=(third, 2)) for c in centers])


def main() -> None:
    # STEP 1 — the doc's 1D dance: points 1,2,9,10, deliberately bad start 3,6
    print("STEP 1 — the 1D dance (points 1,2,9,10; centers start at 3 and 6):")
    line = np.array([[1.0], [2.0], [9.0], [10.0]])
    km1d = ScratchKMeans(k=2)
    for round_no, (centers, labels) in enumerate(km1d.fit_steps(line, init=[[3.0], [6.0]]),
                                                 start=1):
        groups = [sorted(line[labels == j, 0].tolist()) for j in range(2)]
        print(f"    round {round_no}: centers {centers[:, 0].tolist()} "
              f"→ groups {groups[0]} | {groups[1]}")
    assert km1d.centroids_[:, 0].tolist() == [1.5, 9.5]
    print(f"    converged at {km1d.centroids_[:, 0].tolist()} — the doc's answer, "
          "and it can never un-converge\n")

    # STEP 2 — the doc's 6-customer walkthrough, every distance checked
    print("STEP 2 — the doc's 6 customers, K=2, init A=(15,39) B=(70,85):")
    customers6 = np.array([[15.0, 39.0], [25.0, 60.0], [60.0, 5.0],
                           [70.0, 85.0], [40.0, 40.0], [50.0, 80.0]])
    init = [[15.0, 39.0], [70.0, 85.0]]
    d = pairwise_distances(customers6, np.array(init))
    expected = np.array([[0.0, 71.7], [23.3, 51.5], [56.4, 80.6],
                         [71.7, 0.0], [25.0, 54.1], [53.9, 20.6]])
    assert np.allclose(np.round(d, 1), expected)      # the doc's table, verified
    for point, row in zip(customers6, d):
        side = "A" if row[0] < row[1] else "B"
        print(f"    ({point[0]:>4.0f},{point[1]:>3.0f}) → "
              f"dist to A {row[0]:>5.1f}, to B {row[1]:>5.1f} → cluster {side}")

    km6 = ScratchKMeans(k=2).fit(customers6, init=init)
    assert np.allclose(km6.centroids_, [[35.0, 36.0], [60.0, 82.5]])
    print(f"    update → A {km6.centroids_[0].tolist()}, "
          f"B {km6.centroids_[1].tolist()} — then nobody switches: converged\n")

    # STEP 3 — inertia only falls, but WHERE it lands depends on the start
    print("STEP 3 — inertia per round (one run), then 10 random starts:")
    X = make_customers()
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)      # scale first — distances!

    frames = list(ScratchKMeans(k=3, seed=3).fit_steps(X_std))
    inertias = [inertia_of(X_std, c, lab) for c, lab in frames]
    print(f"    seed 3:  {' → '.join(f'{i:.0f}' for i in inertias)}  "
          "(falls every round — the convergence guarantee)")
    assert all(a >= b for a, b in zip(inertias, inertias[1:]))

    runs = [ScratchKMeans(k=3, seed=s).fit(X_std) for s in range(10)]
    finals = sorted(round(m.inertia_) for m in runs)
    best = min(runs, key=lambda m: m.inertia_)        # n_init=10, by hand
    print(f"    10 seeds' final inertias: {finals}")
    print(f"    → same data, different starts, different traps. "
          f"Keep the best: {best.inertia_:.0f} (that's all n_init=10 is)\n")

    # STEP 4 — choosing K: the elbow, then silhouette as tie-breaker
    print("STEP 4 — the elbow (inertia always falls; the QUESTION is how fast):")
    elbow = {}
    for k in range(1, 9):
        elbow[k] = min(ScratchKMeans(k=k, seed=s).fit(X_std).inertia_
                       for s in range(10))
        drop = f"  (−{elbow[k - 1] - elbow[k]:>3.0f})" if k > 1 else ""
        print(f"    K={k}: inertia {elbow[k]:>6.1f}{drop}")
    print("    → big drops until K=3, crumbs after: the elbow says 3")

    print("    silhouette (scratch vs sklearn):")
    for k in [2, 3, 4, 5]:
        labels_k = min((ScratchKMeans(k=k, seed=s).fit(X_std) for s in range(10)),
                       key=lambda m: m.inertia_).labels_
        ours = silhouette_scratch(X_std, labels_k)
        theirs = float(silhouette_score(X_std, labels_k))
        assert np.allclose(ours, theirs)
        marker = "  ← best" if k == 3 else ""
        print(f"    K={k}: {ours:.3f}{marker}")
    print("    → both judges agree on K=3, and scratch == sklearn (allclose)\n")

    # STEP 5 — sklearn, and agreement up to label renaming
    print("STEP 5 — sklearn KMeans(n_clusters=3, n_init=10):")
    sk = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_std)
    print(f"    inertia: scratch best {best.inertia_:.2f}  "
          f"sklearn {sk.inertia_:.2f}")
    # cluster NUMBERS are arbitrary — map each scratch cluster to the sklearn
    # label its members most often get, then compare assignments
    mapping = {j: Counter(sk.labels_[best.labels_ == j]).most_common(1)[0][0]
               for j in range(3)}
    remapped = np.array([mapping[j] for j in best.labels_])
    agreement = float((remapped == sk.labels_).mean())
    print(f"    assignments agree (after renaming): {agreement:.0%}")
    assert agreement == 1.0
    print("    → cluster 2 ≠ cluster 2 across runs — only the GROUPING is real, "
          "names are arbitrary\n")

    # STEP 6 — clusters are useless until you profile and name them
    print("STEP 6 — profiling with pandas groupby (give the clusters names):")
    df = pd.DataFrame(X, columns=["income_k", "spending_score"])
    df["cluster"] = best.labels_
    profile = df.groupby("cluster").mean().round(0)
    profile["size"] = df.groupby("cluster").size()
    print(profile.to_string())
    for cluster_id, row in profile.iterrows():
        if row["income_k"] > 70:
            name = "VIPs (high income, high spending)"
        elif row["spending_score"] > 50:
            name = "young spenders (modest income, spends anyway)"
        else:
            name = "careful savers (mid income, low spending)"
        print(f"    cluster {cluster_id} → {name}")
    print()

    # Plot: the dance from a bad start (left), elbow + silhouette (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bad_init = [[20.0, 60.0], [30.0, 80.0], [80.0, 75.0]]   # two in one blob
    viz = ScratchKMeans(k=3)
    history = [c for c, _ in viz.fit_steps(X, init=bad_init)]
    colors = ["steelblue", "darkorange", "seagreen"]
    for j in range(3):
        members = X[viz.labels_ == j]
        ax1.scatter(*members.T, s=18, color=colors[j], alpha=0.5)
        path = np.array([h[j] for h in history])
        ax1.plot(*path.T, "k--", linewidth=1)
        ax1.scatter(*path[0], marker="s", s=60, color=colors[j], edgecolor="black")
        ax1.scatter(*viz.centroids_[j], marker="X", s=200, color=colors[j],
                    edgecolor="black")
    ax1.set_xlabel("income (k$)")
    ax1.set_ylabel("spending score")
    ax1.set_title(f"The dance: bad start (squares) → centroids walk home (X), "
                  f"{len(history)} rounds")

    ks = list(elbow.keys())
    ax2.plot(ks, [elbow[k] for k in ks], "o-", color="steelblue", label="inertia")
    ax2.axvline(3, color="gray", linestyle="--", label="elbow at K=3")
    ax2.set_xlabel("K (number of clusters)")
    ax2.set_ylabel("inertia (lower = tighter)", color="steelblue")
    ax2.set_title("The elbow: pay for clusters until they stop buying tightness")
    ax2.legend()

    fig.tight_layout()
    out = "k-means/kmeans_plot.png"
    fig.savefig(out, dpi=120)
    print(f"    plot saved → {out}")


if __name__ == "__main__":
    main()
