# Naive Bayes

## TL;DR

Naive Bayes uses probability theory (Bayes' Theorem) to classify data. It calculates: "given these features, what's the probability of each class?" The "naive" part: it assumes all features are independent of each other — which is almost never true, but works surprisingly well in practice. It's extremely fast, works great for text classification (spam detection, sentiment analysis), and needs very little data to learn. Always the first algorithm to try for NLP tasks.

> 💡 **Key Insight:** Even though the "naive" independence assumption is wrong (words clearly influence each other), the ranking of class probabilities is usually still correct. You don't need accurate probabilities — just the right ranking to pick the most likely class.

---

## The Mental Model

Think of a **customs officer checking bags at an airport**.

The officer checks each item in your bag independently: "Is this suspicious?" for scissors, "Is this suspicious?" for liquids, "Is this suspicious?" for electronics. They don't think "scissors AND liquids together is MORE suspicious" — they just tally up individual suspicion scores. Despite this simplification, they correctly flag most suspicious bags.

Mapping:
- Items in the bag → features (words in an email)
- Officer's experience with each item → P(feature | class) from training data
- Overall suspicion score → P(class | features)
- "Flag it" threshold → classification decision
- Each item checked separately → the naive independence assumption
- Surprisingly catches most actual threats → "naive" assumption works in practice

---

## Why It Exists

### The Problem

To compute P(class | all features together), you'd need to see every possible combination of feature values — a combinatorial explosion. With 1,000 words (features) in an email, you'd need 2¹⁰⁰⁰ possible combinations in your training data. Impossible.

```
P(spam | word₁="free", word₂="money", word₃="click", ..., word₁₀₀₀="now")
→ We'd need to have seen this EXACT combination before. Never works.
```

### The Solution

Assume features are independent: `P(all features | class) = P(feature₁ | class) × P(feature₂ | class) × ...`

Now you only need: P(each word | spam) — estimable from thousands of spam emails. Each probability is estimated separately and multiplied together.

### What Changed

Naive Bayes made text classification computationally feasible in the 1990s. Before modern deep learning, it was THE spam filter algorithm — still used today in many production systems for its speed and simplicity.

---

## Core Concepts

### 1. Bayes' Theorem

**One-line definition:** A formula that updates your belief in a hypothesis based on evidence.

**Analogy:** You're a doctor. 1% of people have disease X. A test comes back positive. The test is 90% accurate. Bayes' Theorem tells you the probability that the patient actually has the disease (it's not 90% — it's much lower, because the disease is rare).

```
Bayes' Theorem:
  P(class | features) = P(features | class) × P(class) / P(features)
  
In plain English:
  Probability              How often this         How common
  it's spam         =      exact pattern     ×    spam is      / (normalizing constant)
  given these words        appears in spam

The terms:
  P(class | features)  → POSTERIOR: what we want (probability of class given evidence)
  P(features | class)  → LIKELIHOOD: how likely are these features given the class?
  P(class)             → PRIOR: how common is this class overall?
  P(features)          → EVIDENCE: how common are these features? (just a normalizer)
```

**Common misconception:** You need calculus to understand Bayes' Theorem. The formula is just a recipe for updating probability estimates with evidence. The concept is intuitive: rare diseases are unlikely even with positive tests.

---

### 2. The Naive Independence Assumption

**One-line definition:** Assume each feature is independent of all others when given the class — so we multiply individual probabilities.

**Analogy:** Rating a movie by independently rating the acting, the plot, and the visuals — without considering how they interact. An amazing cast might offset a weak plot, but Naive Bayes ignores that interaction.

```
Real world (wrong but used):
  P("free" AND "money" | spam) ≠ P("free" | spam) × P("money" | spam)
  
  In reality: "free" and "money" appear together in spam more than either alone.
  The words are correlated — NOT independent.

Naive Bayes (incorrect but works):
  P("free" AND "money" | spam) ≈ P("free" | spam) × P("money" | spam)
  = 0.7 × 0.6 = 0.42  ← slightly wrong probability

  But for RANKING:
  P(spam | "free money") >> P(not spam | "free money")
  → Still correctly identifies the email as spam!
  
  Ranking right → classification right. Who cares about exact probabilities?
```

**Common misconception:** The naive assumption makes the model useless because real features aren't independent. The classifier still works because classification only requires the correct ranking of class probabilities — not perfectly calibrated probabilities.

---

### 3. Laplace Smoothing

**One-line definition:** Adding a small count to every feature to prevent zero-probability problems when a word appears in the test set but never appeared in training.

**Analogy:** You're making a recipe based on past experience. If you've never seen someone use paprika in a dessert, your model assigns 0 probability — so any recipe with paprika = 0% chance of being a dessert. Even one paprika dessert would fix this. Laplace smoothing is artificially adding "1 paprika dessert" to your experience so you're never at exactly 0%.

```
Without smoothing:
  P("cryptocurrency" | spam) = 0 / 500 = 0.0
  (500 = total word occurrences in spam docs; "cryptocurrency" never appeared)

  New email with "cryptocurrency":
  P(spam | ..., "cryptocurrency", ...) = ... × 0.0 × ... = 0.0 always!

  → Any email with an unseen word is always classified the same way. Broken.

With Laplace smoothing (scikit-learn default, alpha=1):
  P(word | spam) = (count + alpha) / (total_word_count_in_spam + alpha × V)

  If V = 10,000 unique words in the vocabulary and spam has 500 total word
  occurrences, then:
  P("cryptocurrency" | spam) = (0 + 1) / (500 + 1 × 10,000) = 1 / 10,500
                             ≈ 0.0001  ← tiny but nonzero

  → Unseen words get a small probability. Classification still works.

V = vocabulary size (number of unique words in training data)
```

**Common misconception:** Smoothing artificially inflates probabilities. The effect is tiny — 1 added to counts of thousands. It just prevents the catastrophic failure of zero-probability multiplication.

---

### 4. Naive Bayes Variants

**One-line definition:** Different variants are designed for different types of features.

```
Variant         | Feature Type        | Example Use Case
─────────────────────────────────────────────────────────────────
Multinomial     | Count/frequency     | Word counts in text (most common for NLP)
Bernoulli       | Binary (0/1)        | Word present or absent (short texts)
Gaussian        | Continuous numbers  | Height, weight, temperature measurements
Complement NB   | Text (imbalanced)   | Better for multi-class text classification
```

```python
# For text classification:
from sklearn.naive_bayes import MultinomialNB    # use this for email/reviews/articles
from sklearn.naive_bayes import BernoulliNB      # use for short texts, binary features
from sklearn.naive_bayes import GaussianNB       # use for continuous numeric features
```

**Common misconception:** Gaussian Naive Bayes is the default for everything. Only use it for numeric features that roughly follow a normal distribution. For text, always use Multinomial or Bernoulli.

---

### 5. Log Probabilities — Why We Use Them

**One-line definition:** Instead of multiplying tiny probabilities (which causes numerical underflow), we add their logarithms.

**Analogy:** Multiplying 1000 numbers that are each 0.01 = 10⁻²⁰⁰⁰ — a number so small no computer can represent it. Adding their logarithms instead: sum of log(0.01) for 1000 terms = -2000 × 1000 = -2000 (a perfectly normal number).

```python
# Naive: multiply probabilities (breaks for long documents)
P_spam = P_free * P_money * P_click * P_now * ... × 500 words = underflow!

# Smart: add log probabilities (always works)
log_P_spam = log(P_free) + log(P_money) + log(P_click) + log(P_now) + ...
# scikit-learn handles this automatically
```

**Common misconception:** You need to implement this yourself. scikit-learn's Naive Bayes implementations automatically use log probabilities internally. You don't need to worry about this — just know it's why the model works for long documents.

---

## How It Actually Works (Step-by-Step)

```
Training data (6 emails):

Email | contains "free" | contains "money" | Spam?
────────────────────────────────────────────────────
1     | Yes             | Yes              | Spam
2     | Yes             | No               | Spam
3     | No              | Yes              | Spam
4     | No              | No               | Not Spam
5     | No              | No               | Not Spam
6     | Yes             | No               | Not Spam

Step 1: Compute priors
  P(spam) = 3/6 = 0.5
  P(not spam) = 3/6 = 0.5

Step 2: Compute likelihoods
  P("free" | spam)     = 2/3 = 0.67
  P("money" | spam)    = 2/3 = 0.67
  P("free" | not spam) = 1/3 = 0.33
  P("money" | not spam)= 0/3 → with smoothing: 1/5 = 0.20

Step 3: New email with "free" AND "money" — classify it

  P(spam | free, money) ∝ P(free|spam) × P(money|spam) × P(spam)
                        ∝ 0.67 × 0.67 × 0.5 = 0.224
  
  P(not spam | free, money) ∝ P(free|not spam) × P(money|not spam) × P(not spam)
                            ∝ 0.33 × 0.20 × 0.5 = 0.033

Step 4: Normalize (so they sum to 1)
  Total = 0.224 + 0.033 = 0.257
  P(spam | free, money) = 0.224 / 0.257 = 87%
  P(not spam | ...) = 0.033 / 0.257 = 13%

Step 5: Predict SPAM (87% confidence)
```

---

## Code in Practice

### 1. Hello World — Spam Detector

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = [
    "free money click now",  "win prize free lottery",
    "free consultation",     "meeting at 3pm tomorrow",
    "project deadline today","lunch plans for Friday",
]
labels = [1, 1, 1, 0, 0, 0]   # 1=spam, 0=not spam

# Convert text to word count matrix
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

# Predict new emails
test = vectorizer.transform(["free money win", "office meeting schedule"])
probs = model.predict_proba(test)
preds = model.predict(test)

for text, pred, prob in zip(["free money win", "office meeting"], preds, probs):
    print(f'"{text}" → {"SPAM" if pred else "NOT SPAM"} ({max(prob):.0%} confident)')
```

### 2. Practical — TF-IDF + Evaluation

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_20newsgroups

# Real multi-class text classification dataset (20 news categories)
categories = ['sci.space', 'talk.politics.guns', 'rec.sport.baseball']
data = fetch_20newsgroups(subset='all', categories=categories)

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# TF-IDF: better than raw counts — penalizes common words like "the"
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB(alpha=1.0)   # alpha = Laplace smoothing
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=categories))
```

### 3. Real-World Pattern — Inspecting What the Model Learned

```python
import numpy as np

def top_words_per_class(model, vectorizer, n=10):
    """Show what words most indicate each class."""
    feature_names = vectorizer.get_feature_names_out()
    
    for i, class_name in enumerate(categories):
        # log probabilities for each word given this class
        log_probs = model.feature_log_prob_[i]
        top_indices = np.argsort(log_probs)[-n:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        print(f"\n{class_name}: {', '.join(top_words)}")

# sci.space:          nasa, space, moon, orbit, shuttle
# politics.guns:      gun, guns, firearm, weapons, amendment
# sport.baseball:     baseball, game, team, players, season
# This is what the model "learned" — interpretable!
top_words_per_class(model, vectorizer)
```

---

## Gotchas & Pitfalls

```
❌ Using Gaussian NB for text/count data
   Gaussian NB assumes continuous normal distribution — word counts are discrete
✅ Use MultinomialNB for word counts, BernoulliNB for word presence/absence

❌ Forgetting Laplace smoothing (alpha=0)
   Any word in test not seen in training → zero probability → broken model
✅ Default alpha=1.0 is fine; never set alpha=0

❌ Not using TF-IDF when words have very different frequencies
   Word "the" appears everywhere but tells you nothing about class
✅ Use TfidfVectorizer instead of CountVectorizer for better feature weighting

❌ Using raw probability outputs without calibration
   Naive Bayes probabilities are overconfident (99.9% when it should be 80%)
✅ Use CalibratedClassifierCV if you need calibrated probabilities

❌ Applying Naive Bayes where feature interactions are critical
   "not good" (negative sentiment) ≠ "not" + "good" (Naive Bayes sees two positives)
✅ Use BERT or logistic regression when context and word order matter

❌ Scaling features before MultinomialNB
   MultinomialNB requires NON-NEGATIVE counts — StandardScaler creates negative values
✅ Only use MinMaxScaler if you must scale, or better, don't scale at all for Multinomial
```

---

## When to Use / When NOT to Use

### Use Naive Bayes When:
- Text classification: spam detection, sentiment, topic categorization
- Small datasets — it estimates parameters from very little data
- Need an extremely fast model (real-time, high-volume classification)
- Want a strong probabilistic baseline before trying complex models

### Don't Use Naive Bayes When:
- Features are highly correlated and interactions matter (e.g., "not good")
- You need perfectly calibrated probabilities
- Continuous numeric features with complex distributions (use logistic regression)
- Image or audio data (use neural networks)

---

## Related Concepts

| Concept | Connection |
|---------|------------|
| Logistic Regression | Another probabilistic classifier — LR learns weights, NB uses counting. LR handles feature correlation better. |
| TF-IDF | The feature engineering step that makes Naive Bayes work better on text |
| Text Preprocessing | Tokenization, stop words, stemming all feed into the word-count matrix NB uses |
| Bayes' Theorem | The mathematical foundation — understanding priors and likelihoods is key |
| Gradient Boosting | Often replaces NB when you have enough data and need better accuracy |

---

## Cheat Sheet

```python
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Text classification pipeline:
vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train_text)

model = MultinomialNB(alpha=1.0)   # alpha = Laplace smoothing (don't set to 0)
model.fit(X_train_vec, y_train)

model.predict(X_test_vec)          # class predictions
model.predict_proba(X_test_vec)    # class probabilities per sample
model.feature_log_prob_            # log P(word | class) — what it learned

Variants quick reference:
  MultinomialNB  → word counts or TF-IDF (most common for text)
  BernoulliNB    → binary features (word present/absent)
  GaussianNB     → continuous numeric features only

Remember:
  1. Never set alpha=0 — Laplace smoothing prevents zero probabilities
  2. For text: TF-IDF > raw counts (penalizes common words)
  3. Naive Bayes probabilities are overconfident — calibrate if probabilities matter
```

---

## Self-Check Questions

<details>
<summary>Click to reveal answers</summary>

**Q1: Why is the algorithm called "naive"?**
Because it naively assumes all features are conditionally independent given the class. In reality, features are almost always correlated — but the model ignores these correlations. Surprisingly, this often doesn't hurt classification accuracy because ranking classes correctly (not computing exact probabilities) is sufficient.

**Q2: What is Laplace smoothing and why is it necessary?**
Laplace smoothing adds a small count (usually 1) to every feature-class combination, so no probability is exactly zero. Without it, if a word never appeared in spam emails during training, seeing that word in a test email would make the spam probability exactly 0 — no matter how many other spam words appear. One unseen word would veto the entire calculation.

**Q3: What's the difference between MultinomialNB and GaussianNB?**
MultinomialNB is for discrete count data (like word counts in text) — it estimates P(word|class) from counts. GaussianNB is for continuous numeric features — it assumes each feature follows a Gaussian distribution and estimates the mean and variance per class. Using GaussianNB for text gives very poor results.

**Q4: Why do we use log probabilities instead of regular probabilities?**
When classifying a long document, we multiply hundreds of small probabilities together (one per word). The product becomes so tiny it underflows to 0 in floating-point arithmetic — making all classes equally likely (0). Taking log transforms multiplication into addition: log(p₁ × p₂ × ... × pₙ) = log(p₁) + log(p₂) + ... + log(pₙ). Addition never underflows.

**Q5: "Naive Bayes gives 99% confidence that this review is positive." Should you trust that number?**
No. Naive Bayes probabilities are systematically overconfident because the independence assumption amplifies certainty. The ranking (positive > negative) is reliable; the absolute probability value is not. If you need calibrated probabilities (for threshold tuning or business decisions), use CalibratedClassifierCV(MultinomialNB(), method='isotonic').

</details>

---

## Go Deeper

| Resource | Why It's Worth Your Time |
|----------|--------------------------|
| [StatQuest: Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA) | Visual explanation of Bayes' Theorem and how the spam classifier works. Best 15-minute intro. |
| [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) | All three variants explained with mathematical detail and code examples. |
| [A Practical Explanation of Naive Bayes — towardsdatascience](https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c) | Best written walkthrough of the algorithm with the full probability calculation. |
| *Speech and Language Processing* Ch. 4 — Jurafsky & Martin | The academic reference for Naive Bayes in NLP. Free online. Shows how NB became the foundation of modern text classification. |
| [Kaggle: SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | Classic spam detection dataset — build a real Naive Bayes spam filter in 30 minutes. |
