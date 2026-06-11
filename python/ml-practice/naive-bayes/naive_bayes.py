"""Learn more Python by building Naive Bayes.

Part 8 — prediction by multiplication. No gradients, no distances, no trees:
count how often each word appears in each class, multiply, pick the bigger.
This is also the series' first TEXT model — strings finally enter the picture.
  STEP 1: Bayes by counting          (the doc's 10,000 people → 8.3%, asserted)
  STEP 2: the doc's 6-email trace    (87% spam / 13% ham, reproduced exactly)
  STEP 3: underflow + Laplace        (0.01**1000 == 0.0, and the unseen-word veto)
  STEP 4: ScratchMultinomialNB       (learn: str methods, dict comprehensions,
                                      Counter.update, set().union(*tokenized))
  STEP 5: sklearn agreement          (closed-form counting → must match EXACTLY)
  STEP 6: what the model learned     (top spam/ham words by log-ratio)

Theory companion: ../../ml/naive-bayes.md

Run from python/ml-practice/:
    uv run naive-bayes/naive_bayes.py
"""

import math
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ──────────────────────────────────────────────────────────────────────────────
# The corpus: 16 tiny emails, hand-written so every token is a clean word
# (this keeps our .lower().split() identical to sklearn's tokenizer in STEP 5).
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_DOCS = [
    "free money click now",                  # spam
    "win free prize now",                    # spam
    "free lottery ticket claim now",         # spam
    "cheap meds online buy now",             # spam
    "win cash prize today",                  # spam
    "claim your free bonus today",           # spam
    "urgent offer expires click now",        # spam
    "buy cheap watches online",              # spam
    "meeting at 3pm tomorrow",               # ham
    "project deadline is friday",            # ham
    "lunch plans for tomorrow",              # ham
    "please review the attached report",     # ham
    "schedule the team standup",             # ham
    "notes from the client call",            # ham
    "can we move our meeting",               # ham
    "the deploy is done",                    # ham
]
TRAIN_LABELS = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]   # 1=spam, 0=ham

TEST_DOCS = [
    "free cash offer click now",
    "team meeting tomorrow",
    "claim the report today",
    "win money at lunch",
]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Multinomial Naive Bayes from scratch: training IS counting.
# ──────────────────────────────────────────────────────────────────────────────

class ScratchMultinomialNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha                       # Laplace smoothing (never 0!)

    def fit(self, docs: list[str], labels: list[int]):
        tokenized = [doc.lower().split() for doc in docs]
        self.classes_ = sorted(set(labels))
        self.vocab_ = sorted(set().union(*tokenized))    # one set from all docs

        # priors: just how common each class is
        self.log_prior_ = {c: math.log(labels.count(c) / len(labels))
                           for c in self.classes_}

        # one big word-tally per class — Counter.update adds counts in place
        counts = {c: Counter() for c in self.classes_}
        for tokens, label in zip(tokenized, labels):
            counts[label].update(tokens)
        totals = {c: sum(counts[c].values()) for c in self.classes_}

        # smoothed log P(word | class), for every word in the vocabulary
        v = len(self.vocab_)
        self.log_likelihood_ = {
            c: {word: math.log((counts[c][word] + self.alpha)
                               / (totals[c] + self.alpha * v))
                for word in self.vocab_}
            for c in self.classes_
        }
        return self

    def _log_score(self, doc: str, c: int) -> float:
        return self.log_prior_[c] + sum(
            self.log_likelihood_[c][word]
            for word in doc.lower().split()
            if word in self.log_likelihood_[c])      # unseen words: skipped,
                                                     # same as sklearn's transform
    def predict_proba_one(self, doc: str) -> dict[int, float]:
        scores = {c: self._log_score(doc, c) for c in self.classes_}
        shift = max(scores.values())                 # subtract the max BEFORE exp:
        exp_scores = {c: math.exp(s - shift) for c, s in scores.items()}
        total = sum(exp_scores.values())             # ...so exp can't underflow
        return {c: e / total for c, e in exp_scores.items()}

    def predict_one(self, doc: str) -> int:
        proba = self.predict_proba_one(doc)
        return max(proba, key=proba.get)


def main() -> None:
    # STEP 1 — Bayes' theorem is "count the right group"
    print("STEP 1 — the doc's counting tree (10,000 people, 1% sick, 90% test):")
    true_pos, false_pos = 90, 990                # sick & caught / healthy & flagged
    by_counting = true_pos / (true_pos + false_pos)
    by_formula = (0.90 * 0.01) / 0.108           # likelihood × prior ÷ evidence
    assert math.isclose(by_counting, by_formula, rel_tol=1e-9)
    print(f"    tested positive: {true_pos + false_pos:,} people, "
          f"actually sick: {true_pos}")
    print(f"    by counting {by_counting:.1%}  =  by formula {by_formula:.1%} "
          "— math.isclose agrees")
    print("    → a positive test on a rare disease ≈ 8%, not 90%: "
          "the false alarms outnumber the catches\n")

    # STEP 2 — the doc's 6-email hand-trace, in plain arithmetic
    print("STEP 2 — the doc's free/money table (6 emails), classify 'free money':")
    p_free_spam, p_money_spam = 2 / 3, 2 / 3
    p_free_ham, p_money_ham = 1 / 3, (0 + 1) / (3 + 2)   # smoothing rescues the 0
    spam_score = p_free_spam * p_money_spam * 0.5
    ham_score = p_free_ham * p_money_ham * 0.5
    p_spam = spam_score / (spam_score + ham_score)
    print(f"    spam: 0.67 × 0.67 × 0.5 = {spam_score:.3f}   "
          f"ham: 0.33 × 0.20 × 0.5 = {ham_score:.3f}")
    print(f"    normalized → spam {p_spam:.0%}, ham {1 - p_spam:.0%} "
          "(the doc's 87/13)")
    assert round(p_spam, 2) == 0.87
    print()

    # STEP 3 — why logs, and why smoothing is not optional
    print("STEP 3 — two ways multiplication breaks, two one-line fixes:")
    print(f"    underflow: 0.01 ** 1000 = {0.01 ** 1000}  "
          "(a real number, but floats give up)")
    print(f"    with logs: 1000 * log10(0.01) = {1000 * math.log10(0.01):.0f}  "
          "(perfectly representable)")
    p_unseen = (0 + 1) / (500 + 1 * 10_000)
    print(f"    zero-veto: P(unseen word | spam) = 0 → the WHOLE product is 0,"
          " one word vetoes everything")
    print(f"    smoothed:  (0+1)/(500+10,000) = {p_unseen:.4f} "
          "— tiny, but the veto is gone (the doc's 1/10,500)\n")

    # STEP 4 — the scratch model on the 16-email corpus
    print("STEP 4 — ScratchMultinomialNB (training is COUNTING — no loop, "
          "no epochs):")
    model = ScratchMultinomialNB(alpha=1.0).fit(TRAIN_DOCS, TRAIN_LABELS)
    print(f"    vocabulary: {len(model.vocab_)} unique words from "
          f"{len(TRAIN_DOCS)} emails")
    for doc in TEST_DOCS:
        proba = model.predict_proba_one(doc)
        label = "SPAM" if model.predict_one(doc) == 1 else "ham "
        print(f"    {label} {proba[1]:>5.1%} spam | \"{doc}\"")
    print()

    # STEP 5 — sklearn must agree EXACTLY (both are closed-form counting)
    print("STEP 5 — sklearn (CountVectorizer + MultinomialNB), same alpha:")
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(TRAIN_DOCS)
    sk = MultinomialNB(alpha=1.0).fit(X_train, TRAIN_LABELS)
    X_test = vectorizer.transform(TEST_DOCS)
    sk_proba = sk.predict_proba(X_test)

    scratch_proba = np.array([[model.predict_proba_one(d)[0],
                               model.predict_proba_one(d)[1]] for d in TEST_DOCS])
    assert np.allclose(scratch_proba, sk_proba)
    print(f"    scratch spam probabilities: {np.round(scratch_proba[:, 1], 4)}")
    print(f"    sklearn spam probabilities: {np.round(sk_proba[:, 1], 4)}")
    print("    → np.allclose passed: no randomness, no optimizer — counting has "
          "exactly one answer\n")

    # STEP 6 — interpretability: which words tip the scales?
    print("STEP 6 — what it learned: log P(word|spam) − log P(word|ham):")
    log_ratio = {word: model.log_likelihood_[1][word] - model.log_likelihood_[0][word]
                 for word in model.vocab_}
    ranked = sorted(log_ratio.items(), key=lambda pair: pair[1], reverse=True)
    spammiest, hammiest = ranked[:5], ranked[-5:][::-1]
    for (s_word, s_val), (h_word, h_val) in zip(spammiest, hammiest):
        print(f"    spam {s_val:+.2f} {s_word:<8}   ham {h_val:+.2f} {h_word}")
    print("    → readable, like the tree in Part 3: the 'model' is a table "
          "of word counts\n")

    # Plot: test-email posteriors (left), most decisive words (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    spam_probs = scratch_proba[:, 1]
    short = [f'"{d}"' for d in TEST_DOCS]
    colors = ["crimson" if p >= 0.5 else "steelblue" for p in spam_probs]
    ax1.barh(short[::-1], spam_probs[::-1], color=colors[::-1])
    ax1.axvline(0.5, color="gray", linestyle="--")
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("P(spam | words)")
    ax1.set_title("Posteriors for the four test emails")

    top = spammiest + hammiest[::-1]
    words = [w for w, _ in top]
    values = [v for _, v in top]
    bar_colors = ["crimson" if v > 0 else "steelblue" for v in values]
    ax2.barh(words[::-1], values[::-1], color=bar_colors[::-1])
    ax2.axvline(0.0, color="gray")
    ax2.set_xlabel("log P(word|spam) − log P(word|ham)")
    ax2.set_title("The most decisive words (the entire model, visualized)")

    fig.tight_layout()
    out = "naive-bayes/bayes_plot.png"
    fig.savefig(out, dpi=120)
    print(f"    plot saved → {out}")


if __name__ == "__main__":
    main()
