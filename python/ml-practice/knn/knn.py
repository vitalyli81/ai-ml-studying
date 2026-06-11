"""Learn more Python by building K-Nearest Neighbors.

Part 7 — the model that doesn't train. No gradients, no recursion, no trees:
KNN just memorizes the data and answers every query with a distance scan.
  STEP 1: the ruler                  (Euclidean distance = Pythagoras, asserted)
  STEP 2: the doc's cat/dog vote     (learn: np.argsort, enumerate(start=1))
  STEP 3: ScratchKNN                 (learn: broadcasting with None, lazy economics)
  STEP 4: the scaling disaster       (the doc's salary/age numbers, reproduced)
  STEP 5: the curse, measured        (nearest ≈ farthest as dimensions grow)
  STEP 6: scratch vs sklearn         (they must agree EXACTLY — KNN has no luck)

Theory companion: ../../ml/knn.md

Run from python/ml-practice/:
    uv run knn/knn.py
"""

from collections import Counter
from time import perf_counter
from typing import Literal

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Distance is grade-school geometry: square the gaps, add, square-root.
# ──────────────────────────────────────────────────────────────────────────────

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum()))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — All query-to-point distances at once, with broadcasting.
# queries (q, d) and points (n, d) can't subtract — until None adds the axes:
#   queries[:, None, :] is (q, 1, d);  points[None, :, :] is (1, n, d)
#   → the subtraction broadcasts to (q, n, d): every query minus every point.
# ──────────────────────────────────────────────────────────────────────────────

def pairwise_distances(queries: np.ndarray, points: np.ndarray) -> np.ndarray:
    diff = queries[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))            # (q, n): one row per query


class ScratchKNN:
    def __init__(self, k: int = 5,
                 weights: Literal["uniform", "distance"] = "uniform"):
        self.k = k
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_, self.y_ = X, y     # "training" = remembering. That's ALL of it.
        return self

    def predict(self, queries: np.ndarray) -> np.ndarray:
        dists = pairwise_distances(queries, self.X_)
        nearest = np.argsort(dists, axis=1)[:, :self.k]   # indices, not values!
        predictions = []
        for query_idx, neighbor_idx in enumerate(nearest):
            labels = self.y_[neighbor_idx]
            if self.weights == "distance":
                weight = 1.0 / (dists[query_idx, neighbor_idx] + 1e-12)
                votes: dict = {}
                for label, w in zip(labels, weight):
                    votes[label] = votes.get(label, 0.0) + w
                predictions.append(max(votes, key=votes.get))
            else:
                predictions.append(Counter(labels).most_common(1)[0][0])
        return np.array(predictions)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 data — two interleaved half-moons: no single line separates them,
# but every point's NEIGHBORHOOD is pure. KNN's home turf.
# ──────────────────────────────────────────────────────────────────────────────

def make_moons_scratch(n: int = 300, noise: float = 0.22,
                       seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    half = n // 2
    angle = rng.uniform(0.0, np.pi, half)
    top = np.column_stack([np.cos(angle), np.sin(angle)])
    bottom = np.column_stack([1.0 - np.cos(angle), 0.5 - np.sin(angle)])
    X = np.vstack([top, bottom]) + rng.normal(0.0, noise, size=(n, 2))
    y = np.concatenate([np.zeros(half, dtype=int), np.ones(half, dtype=int)])
    return X, y


def accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float((actual == predicted).mean())


def main() -> None:
    # STEP 1 — the doc's Pythagoras example, locked in with an assert
    print("STEP 1 — distance is Pythagoras (the doc's A=(1,2), B=(4,4)):")
    d = euclidean(np.array([1.0, 2.0]), np.array([4.0, 4.0]))
    assert round(d, 1) == 3.6                  # √13 — the doc's arithmetic
    print(f"    gaps (3, 2) → √(9+4) = √13 = {d:.2f} — assert passed\n")

    # STEP 2 — the doc's 5-pet walkthrough: distances, argsort, vote
    print("STEP 2 — the doc's cat/dog table, query [1.3, 2.5], K=3:")
    pets = np.array([[1.0, 2.0], [1.5, 1.8], [4.0, 4.5], [4.5, 4.2], [1.2, 3.0]])
    pet_labels = np.array(["Cat", "Cat", "Dog", "Dog", "Cat"])
    names = ["A", "B", "C", "D", "E"]
    query = np.array([1.3, 2.5])

    dists = np.sqrt(((pets - query) ** 2).sum(axis=1))   # one row vs all rows
    print(f"    distances: {np.round(dists, 2)}  (doc: 0.58 0.73 3.36 3.62 0.51)")
    order = np.argsort(dists)                  # SORT THE INDICES, not the values
    for rank, idx in enumerate(order[:3], start=1):      # ranks start at 1
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank, "th")
        print(f"    {rank}{suffix} nearest: {names[idx]} ({dists[idx]:.2f}) "
              f"→ {pet_labels[idx]}")
    vote = Counter(str(label) for label in pet_labels[order[:3]])
    print(f"    vote: {dict(vote)} → predict {vote.most_common(1)[0][0]} "
          "(the doc's answer, 3-0)\n")

    # STEP 3 — the class, and the economics of laziness
    print("STEP 3 — lazy learning: fit() is free, predict() pays for everything:")
    rng = np.random.default_rng(0)
    queries = rng.normal(size=(200, 2))
    for n in [500, 2_000, 8_000]:
        X_big = rng.normal(size=(n, 2))
        y_big = (X_big.sum(axis=1) > 0).astype(int)
        t0 = perf_counter()
        model = ScratchKNN(k=5).fit(X_big, y_big)
        t_fit = perf_counter() - t0
        t0 = perf_counter()
        model.predict(queries)
        t_predict = perf_counter() - t0
        print(f"    n={n:>5}: fit {t_fit * 1000:6.2f} ms   "
              f"predict(200 queries) {t_predict * 1000:6.2f} ms")
    print("    → fit is O(1) — it just stores. Predict scales with n: "
          "the cost moved to query time\n")

    print("    the doc's Q5: neighbors at [0.1, 0.1, 0.1, 10, 10], "
          "labels [A, A, A, B, B]:")
    q5_dists = np.array([0.1, 0.1, 0.1, 10.0, 10.0])
    q5_labels = np.array(["A", "A", "A", "B", "B"])
    weights = 1.0 / q5_dists
    tally: dict = {}
    for label, w in zip(q5_labels, weights):
        tally[label] = tally.get(label, 0.0) + w
    print(f"    uniform: A 3 votes, B 2 votes — distance-weighted: "
          f"A {tally['A']:.0f}, B {tally['B']:.1f} → "
          f"{max(tally, key=tally.get)} either way, but never a coin flip\n")

    # STEP 4 — the scaling disaster, with the doc's exact people
    print("STEP 4 — unscaled salary drowns age (the doc's three people):")
    people = np.array([[30_000.0, 25.0], [30_100.0, 65.0], [60_000.0, 26.0]])
    d_ab = euclidean(people[0], people[1])
    d_ac = euclidean(people[0], people[2])
    print(f"    raw:    A→B {d_ab:>8,.0f}  A→C {d_ac:>8,.0f}  "
          "→ B 'closest' (a 40-year age gap, ignored!)")
    scaled = (people - people.mean(axis=0)) / people.std(axis=0)
    d_ab_s = euclidean(scaled[0], scaled[1])
    d_ac_s = euclidean(scaled[0], scaled[2])
    print(f"    scaled: A→B {d_ab_s:>8.2f}  A→C {d_ac_s:>8.2f}  "
          "→ C closest: a $30K gap and a 40-year gap now weigh what they should\n")

    # STEP 5 — the curse of dimensionality, as an experiment
    print("STEP 5 — the curse: 100 random points in a unit cube, one query:")
    rng = np.random.default_rng(42)
    for n_dims in [2, 20, 200]:
        cloud = rng.uniform(size=(100, n_dims))
        probe = rng.uniform(size=n_dims)
        cloud_dists = np.sqrt(((cloud - probe) ** 2).sum(axis=1))
        ratio = cloud_dists.min() / cloud_dists.max()
        print(f"    {n_dims:>3} dims: nearest {cloud_dists.min():5.2f}, "
              f"farthest {cloud_dists.max():5.2f}, ratio {ratio:.2f}")
    print("    → the ratio marches toward 1.0: nearest ≈ farthest, "
          "so 'nearest' stops meaning 'similar'\n")

    # STEP 6 — scratch vs sklearn on the moons (exact agreement required)
    print("STEP 6 — half-moons: scratch vs sklearn, then CV picks K:")
    X, y = make_moons_scratch()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scratch = ScratchKNN(k=5).fit(X_train, y_train)
    sk = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    scratch_pred = scratch.predict(X_test)
    sk_pred = sk.predict(X_test)
    agree = accuracy(scratch_pred, sk_pred)
    print(f"    scratch K=5 test {accuracy(y_test, scratch_pred):.0%}   "
          f"sklearn K=5 test {accuracy(y_test, sk_pred):.0%}   "
          f"predictions agree: {agree:.0%}")
    assert agree == 1.0     # no randomness anywhere — same K, same data, same answer

    print("    cross-validation over odd K (the doc: never guess K, measure it):")
    cv_results = []
    for k in [1, 3, 5, 9, 15, 25, 51]:
        scores = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                                 X_train, y_train, cv=5)
        cv_results.append((k, scores.mean(), scores.std()))
        print(f"      K={k:<3} CV accuracy {scores.mean():.0%} ± {scores.std():.1%}")
    best_k = max(cv_results, key=lambda row: row[1])[0]
    best = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
    print(f"    best K = {best_k} → test {accuracy(y_test, best.predict(X_test)):.0%}\n")

    # Plot — the doc's K=1 / K=5 / K=50 ASCII picture, drawn for real
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 250),
                         np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 250))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    captions = {1: "jagged — memorizes every noise point",
                5: "balanced — the doc's default",
                50: "over-smooth — local detail blurred away"}
    for ax, k in zip(axes, [1, 5, 50]):
        knn_k = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        zz = knn_k.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                    colors=["lightsteelblue", "mistyrose"], alpha=0.7)
        top = y_train == 0
        ax.scatter(*X_train[top].T, marker="o", s=22, color="steelblue")
        ax.scatter(*X_train[~top].T, marker="x", s=30, color="crimson")
        ax.set_title(f"K={k}: train {accuracy(y_train, knn_k.predict(X_train)):.0%}, "
                     f"test {accuracy(y_test, knn_k.predict(X_test)):.0%}\n"
                     f"({captions[k]})")
    fig.tight_layout()
    out = "knn/knn_plot.png"
    fig.savefig(out, dpi=120)
    print(f"    plot saved → {out}")


if __name__ == "__main__":
    main()
