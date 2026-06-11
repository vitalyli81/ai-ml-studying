"""Learn more Python by building Model Evaluation.

Part 11 — the capstone. No new model: this part builds the JUDGES — the
metrics, splits, and curves that decide whether any of Parts 1-10 actually
worked. The narrative is the theory doc's loan-default walkthrough, executed.
  STEP 1: the accuracy trap          (one confusion matrix, four honest lenses)
  STEP 2: k-fold CV from scratch     (a generator of folds == sklearn, asserted)
  STEP 3: ROC + AUC from scratch     (argsort + cumsum + np.trapezoid)
  STEP 4: when ROC-AUC lies          (1% fraud: PR-AUC is the honest headline)
  STEP 5: threshold tuning           ("business says recall >= 80%" — find the price)
  STEP 6: grid search demystified    (itertools.product x your own CV + ONE test look)

Theory companion: ../../ml/model-evaluation.md

Run from python/ml-practice/:
    uv run model-evaluation/model_evaluation.py   (~15s: it fits ~30 forests)
"""

import itertools

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, \
    train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# The four lenses (Part 2 built these once for one model — now they're tools)
# ──────────────────────────────────────────────────────────────────────────────

def confusion(actual: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    return {"TP": int(((actual == 1) & (predicted == 1)).sum()),
            "FP": int(((actual == 0) & (predicted == 1)).sum()),
            "TN": int(((actual == 0) & (predicted == 0)).sum()),
            "FN": int(((actual == 1) & (predicted == 0)).sum())}


def accuracy(actual, predicted) -> float:
    return float((actual == predicted).mean())


def precision(actual, predicted) -> float:
    c = confusion(actual, predicted)
    return c["TP"] / (c["TP"] + c["FP"]) if c["TP"] + c["FP"] else 0.0


def recall(actual, predicted) -> float:
    c = confusion(actual, predicted)
    return c["TP"] / (c["TP"] + c["FN"]) if c["TP"] + c["FN"] else 0.0


def f1(actual, predicted) -> float:
    p, r = precision(actual, predicted), recall(actual, predicted)
    return 2 * p * r / (p + r) if p + r else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — K-fold from scratch: a GENERATOR of (train_idx, test_idx) pairs.
# Replicates sklearn's KFold exactly: contiguous blocks, the first n % k
# folds one row bigger. Five yields = the doc's five-judge panel.
# ──────────────────────────────────────────────────────────────────────────────

def kfold_indices(n: int, k: int = 5):
    sizes = np.full(k, n // k)
    sizes[: n % k] += 1
    start = 0
    for size in sizes:
        test = np.arange(start, start + size)
        train = np.concatenate([np.arange(0, start), np.arange(start + size, n)])
        yield train, test
        start += size


def cross_val_f1(model_factory, X: np.ndarray, y: np.ndarray, k: int = 5) -> list[float]:
    scores = []
    for train_idx, test_idx in kfold_indices(len(X), k):
        model = model_factory().fit(X[train_idx], y[train_idx])
        scores.append(f1(y[test_idx], model.predict(X[test_idx])))
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — ROC from scratch: sort by score (most confident first), walk down
# the list, count TPs and FPs as you go. np.cumsum IS that walk.
# ──────────────────────────────────────────────────────────────────────────────

def roc_curve_scratch(actual: np.ndarray, scores: np.ndarray):
    order = np.argsort(scores)[::-1]               # most confident first
    hits = actual[order]                           # 1 where a positive, 0 where not
    tpr = np.cumsum(hits) / hits.sum()             # recall, growing down the list
    fpr = np.cumsum(1 - hits) / (1 - hits).sum()   # false alarms, growing too
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr])


def auc_scratch(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapezoid(tpr, fpr))           # area = integral, one call


def pr_curve_scratch(actual: np.ndarray, scores: np.ndarray):
    order = np.argsort(scores)[::-1]
    hits = actual[order]
    tps = np.cumsum(hits)
    prec = tps / np.arange(1, len(hits) + 1)       # purity of the catch so far
    rec = tps / hits.sum()                         # completeness of the catch
    return prec, rec


def average_precision_scratch(actual: np.ndarray, scores: np.ndarray) -> float:
    prec, rec = pr_curve_scratch(actual, scores)
    rec_steps = np.diff(np.concatenate([[0.0], rec]))   # how much recall each
    return float((rec_steps * prec).sum())              # prediction added


# ──────────────────────────────────────────────────────────────────────────────
# The doc's walkthrough data: loans with an ~8% default rate (imbalanced!)
# ──────────────────────────────────────────────────────────────────────────────

def make_loans(n: int = 4_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    income = rng.normal(60, 18, n)                 # k$
    debt_ratio = rng.uniform(0.0, 0.6, n)
    credit = rng.normal(650, 70, n)
    late_payments = rng.poisson(1.0, n)
    z = (-3.65 - 0.02 * (income - 60) + 4.0 * (debt_ratio - 0.3)
         - 0.012 * (credit - 650) + 0.55 * late_payments + rng.normal(0, 0.6, n))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
    X = np.column_stack([income, debt_ratio, credit, late_payments])
    return X, y


def main() -> None:
    X, y = make_loans()
    print(f"the doc's Step 1 — check balance first: {len(y):,} loans, "
          f"{y.mean():.0%} defaulted → imbalanced\n")

    # the doc's Step 2: stratified 70/15/15, test kept pristine
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42)

    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    logit = LogisticRegression(max_iter=2000).fit((X_train - mu) / sigma, y_train)
    val_probs = logit.predict_proba((X_val - mu) / sigma)[:, 1]
    val_pred = (val_probs >= 0.5).astype(int)

    # STEP 1 — the accuracy trap, on the baseline model
    print("STEP 1 — baseline logistic, validation set, threshold 0.5:")
    c = confusion(y_val, val_pred)
    print(f"    confusion: TP={c['TP']}  FP={c['FP']}  FN={c['FN']}  TN={c['TN']}")
    for name, fn, sk_fn in [("accuracy", accuracy, accuracy_score),
                            ("precision", precision, precision_score),
                            ("recall", recall, recall_score),
                            ("f1", f1, f1_score)]:
        ours = fn(y_val, val_pred)
        assert np.isclose(ours, sk_fn(y_val, val_pred))   # formulas verified
        print(f"    {name:>9}: {ours:.2f}")
    print(f"    → the doc's trap, live: {accuracy(y_val, val_pred):.0%} accuracy "
          f"while missing {1 - recall(y_val, val_pred):.0%} of all defaults.")
    print("      (always predicting 'repaid' would score "
          f"{1 - y_val.mean():.0%} — accuracy can't see rare classes)\n")

    # STEP 2 — five judges instead of one: scratch CV == sklearn CV
    print("STEP 2 — 5-fold cross-validation from scratch (RandomForest, F1):")
    factory = lambda: RandomForestClassifier(n_estimators=60, random_state=42)
    ours = cross_val_f1(factory, X_temp, y_temp)
    theirs = cross_val_score(factory(), X_temp, y_temp,
                             cv=KFold(n_splits=5), scoring="f1")
    assert np.allclose(ours, theirs)               # same folds, same scores
    print(f"    fold F1s: {np.round(ours, 3)}")
    print(f"    F1 = {np.mean(ours):.2f} ± {np.std(ours):.2f}  "
          "(scratch folds == sklearn KFold, asserted)")

    sub = slice(0, 600)                            # small data → wobbly folds
    plain = [y_temp[sub][test].mean()
             for _, test in kfold_indices(600, 5)]
    strat = [y_temp[sub][test].mean()
             for _, test in StratifiedKFold(5).split(X_temp[sub], y_temp[sub])]
    print(f"    fold default-rates on 600 rows — plain: "
          f"{[f'{r:.0%}' for r in plain]}")
    print(f"                               stratified: "
          f"{[f'{r:.0%}' for r in strat]}")
    print("    → plain folds wobble on rare classes; stratified pins the ratio\n")

    # STEP 3 — ROC + AUC from scratch
    print("STEP 3 — ROC and AUC, built from argsort + cumsum:")
    fpr, tpr = roc_curve_scratch(y_val, val_probs)
    auc = auc_scratch(fpr, tpr)
    assert np.isclose(auc, roc_auc_score(y_val, val_probs))
    print(f"    AUC = {auc:.3f} (scratch == sklearn roc_auc_score, asserted)")
    print("    → reading: ~the probability a random defaulter is ranked above "
          "a random repayer.")
    print("      Doc's ladder: >0.9 excellent | 0.8-0.9 good | 0.7-0.8 fair | "
          "0.5 coin flip\n")

    # STEP 4 — when ROC-AUC lies: make defaults rare (1%) and compare headlines
    print("STEP 4 — the imbalance lie (defaults thinned 8x):")
    keep = (y_val == 0) | (np.arange(len(y_val)) % 8 == 0)
    y_rare, probs_rare = y_val[keep], val_probs[keep]
    ap_rare = average_precision_scratch(y_rare, probs_rare)
    assert np.isclose(ap_rare, average_precision_score(y_rare, probs_rare))
    print(f"    at {y_rare.mean():.1%} positives: "
          f"ROC-AUC = {roc_auc_score(y_rare, probs_rare):.2f}  "
          f"but PR-AUC = {ap_rare:.2f} (scratch == sklearn, asserted)")
    print(f"    (same model at 8% positives had PR-AUC "
          f"{average_precision_scratch(y_val, val_probs):.2f})")
    print("    → ROC-AUC barely notices rarity; precision collapses. "
          "For heavy imbalance, PR-AUC is the headline\n")

    # STEP 5 — the doc's Step 9: "business says recall must be > 80%"
    print("STEP 5 — threshold tuning: buy recall ≥ 80%, pay in precision:")
    forest = RandomForestClassifier(n_estimators=150, random_state=42)
    forest.fit(X_train, y_train)
    probs = forest.predict_proba(X_val)[:, 1]
    rows = [(t, precision(y_val, (probs >= t).astype(int)),
             recall(y_val, (probs >= t).astype(int)))
            for t in [0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]]
    chosen = max((row for row in rows if row[2] >= 0.80),
                 key=lambda row: row[0])           # highest threshold that qualifies
    for t, p, r in rows:
        mark = "  ← chosen" if t == chosen[0] else ""
        print(f"    threshold {t:.2f}: precision {p:.2f}  recall {r:.2f}{mark}")
    print("    → same model, new contract: every flag is now only "
          f"{chosen[1]:.0%} likely real — that's what 80% recall costs here\n")

    # STEP 6 — grid search demystified: product × CV, then ONE look at test
    print("STEP 6 — GridSearchCV by hand: itertools.product × your own CV:")
    results = []
    for depth, n_trees in itertools.product([4, 8], [50, 150]):
        make = lambda: RandomForestClassifier(max_depth=depth, n_estimators=n_trees,
                                              random_state=42)
        results.append(((depth, n_trees), float(np.mean(cross_val_f1(make, X_temp, y_temp)))))
        print(f"    max_depth={depth}, n_estimators={n_trees:<4} → CV F1 "
              f"{results[-1][1]:.3f}")
    configs, scores = zip(*results)                # unzip pairs into two tuples
    best_i = int(np.argmax(scores))
    best_depth, best_trees = configs[best_i]
    print(f"    best: max_depth={best_depth}, n_estimators={best_trees} "
          "(that's all GridSearchCV does)")

    final = RandomForestClassifier(max_depth=best_depth, n_estimators=best_trees,
                                   random_state=42).fit(X_temp, y_temp)
    train_f1 = f1(y_temp, final.predict(X_temp))
    test_f1 = f1(y_test, final.predict(X_test))    # the ONE test-set touch
    print(f"    final verdict — CV promised F1 {scores[best_i]:.2f}; the ONE "
          f"test look: {test_f1:.2f} (train was {train_f1:.2f})")
    print(f"    → train ≫ test = variance, the doc's diagnosis. And with only "
          f"{int(y_test.sum())} positive test rows, F1-at-0.5 is a noisy judge —")
    print("      picking the best of 4 configs by CV inflates hopes a little; "
          "the once-touched test set exists to catch exactly that\n")

    # Plot: the ROC curve (left), precision/recall vs threshold (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(fpr, tpr, color="steelblue", label=f"logistic (AUC={auc:.2f})")
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="coin flip (AUC=0.50)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (recall)")
    ax1.set_title("ROC: every threshold at once")
    ax1.legend(loc="lower right")

    thresholds = np.linspace(0.01, 0.7, 80)
    pr_pairs = [(precision(y_val, (probs >= t).astype(int)),
                 recall(y_val, (probs >= t).astype(int))) for t in thresholds]
    precisions, recalls = zip(*pr_pairs)
    ax2.plot(thresholds, precisions, color="steelblue", label="precision")
    ax2.plot(thresholds, recalls, color="crimson", label="recall")
    ax2.axhline(0.80, color="gray", linestyle=":", label="recall target 80%")
    ax2.axvline(chosen[0], color="gray", linestyle="--",
                label=f"chosen threshold {chosen[0]:.2f}")
    ax2.set_xlabel("decision threshold")
    ax2.set_ylabel("score")
    ax2.set_title("The tradeoff dial: move the threshold, trade the metrics")
    ax2.legend(loc="center right")

    fig.tight_layout()
    out = "model-evaluation/evaluation_plot.png"
    fig.savefig(out, dpi=120)
    print(f"    plot saved → {out}")


if __name__ == "__main__":
    main()
