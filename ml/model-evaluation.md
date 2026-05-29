# Model Evaluation

## TL;DR

Model evaluation answers one question: **"Is my model actually good, or does it just look good?"** You split your data so the model is tested on examples it has never seen, then pick a metric that matches your real-world cost of being wrong. Accuracy alone is a trap — for imbalanced data (fraud, disease, spam), precision/recall/F1 tell the real story. Cross-validation gives you a stable estimate instead of one lucky number, and bias-variance tells you whether your model is too simple or too complex.

> 💡 **Key Insight:** Evaluation is not a step you do at the end — it's the compass for every decision you make. If you can't measure it, you can't improve it.

---

## The Mental Model

Think of **evaluating a student**.

You don't grade them on the exact problems they practiced — you give them a **new exam** to see if they actually learned the material or just memorized answers. One exam isn't enough either (maybe they got lucky), so you give them several exams and average the scores. And you don't grade a med student the same way as a history student — the cost of a wrong answer is different, so the grading rubric changes.

Mapping:
- Practice problems → training set
- Final exam → test set
- Multiple exams averaged → cross-validation
- Grading rubric → metric (accuracy, F1, RMSE...)
- Cost of wrong answer → why the rubric matters

The algorithm's only job is to learn patterns. **Your** job is to prove those patterns generalize.

---

## Build the Intuition From Zero

Two evaluation ideas trip up almost everyone: **precision vs. recall (which is which, and which to chase), and the bias–variance tradeoff (why fixing one breaks the other).** Let's lock both down with sticky pictures before the formulas.

### Idea 1: Precision vs. Recall — the fishing-net story

Imagine you're catching fish (the "positives" you want) with a net, in a lake full of fish and old boots (the "negatives").

```
PRECISION = "of everything I pulled up, how much was actually fish?"  → purity of your catch
RECALL    = "of all the fish in the lake, how many did I catch?"       → completeness of your catch
```

Now the key insight — **they trade off**, controlled by how aggressively you scoop:

```
TINY careful net (high threshold):      HUGE greedy net (low threshold):
  pull up 3 things, all fish              pull up everything in the lake
  → PRECISION 100% (no boots!)            → RECALL 100% (caught every fish!)
  → RECALL low (missed most fish)         → PRECISION low (tons of boots too)
```

So you can't max both — pushing one down pushes the other up. **Which to favor depends on what a mistake costs:**

```
Cancer screening → favor RECALL.    Missing a sick patient (a fish you let swim away)
                                     is deadly; a false alarm just means more tests.
Spam filter      → favor PRECISION. Dumping a real email in spam (a boot you called a fish)
                                     loses someone's job offer; a little spam slipping through is fine.
```

> 💡 **The memory hook:** **Recall = "did I recall (catch) them all?"** **Precision = "was I precise (correct) about each catch?"** F1 is just the single number you report when you care about both equally.

### Idea 2: Bias vs. Variance — the seesaw you can't beat

Every model's total error splits into two opposite failures. Picture a student studying for an exam:

```
HIGH BIAS (underfitting)              HIGH VARIANCE (overfitting)
= too simple, didn't learn enough     = too complex, memorized the practice exam
"barely studied, fails everything"    "memorized answers, panics on new questions"
wrong on BOTH train AND test          great on train, bad on test
```

Here's why it's a *tradeoff* and not just two separate bugs — it's a seesaw:

```
error
  │\                              /     ← too complex: memorizes noise (high variance)
  │ \                          /
  │  \         sweet spot    /
  │   \___________ ⌄ ______/         ← lowest TOTAL error lives in the middle
  │   too simple (high bias)
  └──────────────────────────────── model complexity →
     simple ←─────────────────→ complex
```

Make the model **more complex** to cut bias (it learns more) and you *raise* variance (it starts memorizing noise). Make it **simpler** to cut variance and you *raise* bias (it learns less). You can't drive both to zero — you hunt for the dip in the middle where their sum is smallest.

```
How you diagnose which side you're on (just compare two numbers):
  train accuracy LOW,  test accuracy LOW   → HIGH BIAS    → make model more complex / add features
  train accuracy HIGH, test accuracy LOW   → HIGH VARIANCE→ simplify / regularize / get more data
  train accuracy HIGH, test accuracy HIGH  → 🎯 you found the sweet spot
```

That train-vs-test gap is the single most useful diagnostic in all of ML. The bias–variance, cross-validation, and regularization sections below are all tools for finding and holding that sweet spot.

---

## Why It Exists

### The Problem Before

Early ML practitioners trained on all their data and reported training accuracy. Models looked amazing — 99% accuracy! — and then failed catastrophically in production. Why? They had **memorized** the training data (overfitting), not learned the underlying pattern.

### The Solution

Hold out data the model never sees during training. Evaluate there. If it generalizes, it works. Add cross-validation to remove luck. Add proper metrics to reflect the real cost of errors.

### What Changed

Every modern ML pipeline treats evaluation as a first-class citizen. In LLMs, this evolved into **evals** — golden datasets, LLM-as-judge, prompt regression testing. Same principle, different medium.

---

## Core Concepts

### 1. Train / Validation / Test Split

**One-line definition:** Slice your data into three chunks so the model can learn, tune, and be judged fairly.

**Analogy:** Practice problems (train), mock exam (validation), final exam (test). Never peek at the final.

**Technical explanation:**
```
All Data (100%)
├── Training (60-70%)    → model learns weights here
├── Validation (15-20%)  → tune hyperparameters here
└── Test (15-20%)        → evaluate ONCE at the end
```

**Common misconception:** People often think "I'll just use train/test." That works — until you tune hyperparameters on the test set, which leaks information and inflates your score. The validation set exists so the test set stays pristine.

```python
from sklearn.model_selection import train_test_split

# First split: separate test set (kept untouched)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: train vs validation from what's left
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 × 0.8 = 0.2
)
```

---

### 2. Cross-Validation (K-Fold)

**One-line definition:** Split data into K folds, train on K-1, test on 1, rotate, average the scores.

**Analogy:** A 5-judge panel, each judging a different slice of your data. Average their scores — more stable than trusting one judge.

**Technical explanation:**
```
5-Fold CV:
Round 1: [TEST][----TRAIN----]          → score 1
Round 2: [TR][TEST][--TRAIN--]          → score 2
Round 3: [--TR--][TEST][-TR-]           → score 3
Round 4: [----TR----][TEST][T]          → score 4
Round 5: [----TRAIN----][TEST]          → score 5

Final score = mean(score 1..5), std = stability
```

**Code:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Common misconception:** People think CV replaces the test set. It doesn't — CV is for **tuning**; the test set is still for the final verdict. Also: for time series, use `TimeSeriesSplit`, never random K-fold (you'd be training on the future to predict the past).

---

### 3. Confusion Matrix

**One-line definition:** A 2x2 table showing every way your classifier can be right or wrong.

**Analogy:** A medical test for a disease.
- **True Positive (TP):** sick person correctly flagged sick
- **False Positive (FP):** healthy person wrongly flagged sick (false alarm)
- **True Negative (TN):** healthy person correctly cleared
- **False Negative (FN):** sick person missed (the dangerous one)

**Technical explanation:**
```
                 Predicted
                 Pos    Neg
Actual Pos   [  TP  |  FN  ]
       Neg   [  FP  |  TN  ]
```

**Code:**
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot()
```

**Common misconception:** People treat all errors as equal. In fraud detection, an FN (missed fraud) costs $10k; an FP (false alarm on a real customer) costs a support ticket. **Which mistake is cheaper?** is the question every metric tries to answer.

---

### 4. Accuracy, Precision, Recall, F1

**One-line definition:** Four lenses on the confusion matrix, each answering a different question.

**The formulas (memorize these):**
```
Accuracy  = (TP + TN) / ALL            → "Overall, how often right?"
Precision = TP / (TP + FP)             → "When I say positive, am I right?"
Recall    = TP / (TP + FN)             → "Of all real positives, how many did I catch?"
F1        = 2 × (P × R) / (P + R)      → "Harmonic mean — balances both"
```

**Analogy: a spam filter.**
| Metric | Question it answers | Fails when... |
|---|---|---|
| Accuracy | "% of emails labeled correctly" | 99% of emails are ham — always guess "ham" → 99% accuracy, useless |
| Precision | "Of emails I flagged as spam, how many really were?" | You mark 1 spam as spam, ignore 999 others — 100% precision, terrible recall |
| Recall | "Of all spam, how much did I catch?" | You mark everything as spam → 100% recall, precision near 0 |
| F1 | "Balance between the two" | When FP and FN costs are wildly unequal (use weighted variants) |

**The precision vs recall tradeoff:**
```
High precision, low recall:
  "I only flag when VERY sure."
  → Fewer false alarms, but misses cases.
  → Use when FP is expensive (spam, content moderation).

High recall, low precision:
  "I flag anything remotely suspicious."
  → Catches almost everything, but many false alarms.
  → Use when FN is expensive (cancer screening, fraud).
```

**Code:**
```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
#               precision    recall  f1-score   support
#     class 0       0.89      0.95      0.92       100
#     class 1       0.78      0.61      0.68        50
```

---

### 5. ROC Curve & AUC

**One-line definition:** A curve showing how your classifier trades off true-positives against false-positives as you move the decision threshold.

**Analogy:** A spam filter with a knob — crank the knob and it becomes more aggressive. ROC plots every possible knob setting. AUC = area under that curve = probability the model ranks a random positive higher than a random negative.

**Technical explanation:**
```
        1 │        ┌─── perfect (AUC=1.0)
 True   │       ╱
 Pos    │      ╱ ← your model (AUC=0.85)
 Rate   │    ╱
        │  ╱  ← random guess (AUC=0.5)
        0 └────────────
          0            1
          False Positive Rate
```

**Rule of thumb:**
- **AUC > 0.9** — excellent
- **AUC 0.8–0.9** — good
- **AUC 0.7–0.8** — fair
- **AUC 0.5** — no better than a coin flip
- **AUC < 0.5** — worse than random (flip your predictions!)

**Common misconception:** AUC is threshold-independent, which is its strength AND weakness. For **severely imbalanced** data (fraud at 0.1%), AUC can look great while precision is terrible. Use **PR-AUC** (precision-recall AUC) for heavy imbalance.

---

### 6. Regression Metrics (MAE, MSE, RMSE, R²)

**One-line definition:** For numeric predictions, measure how far off you were — in different shapes.

**The four horsemen:**
```
MAE  = mean(|y_true - y_pred|)           → avg absolute error, in same units as y
MSE  = mean((y_true - y_pred)²)          → squared error, punishes big misses
RMSE = √MSE                              → back to original units, still punishes outliers
R²   = 1 - (SS_res / SS_total)           → % of variance explained (0 to 1, higher better)
```

**When to use which:**
| Metric | Best for | Why |
|---|---|---|
| MAE | Stable predictions, outliers don't dominate | Treats all errors equally |
| MSE/RMSE | When big errors are disproportionately bad | Squaring makes outliers hurt more |
| R² | Communicating to non-ML stakeholders | "Explains 85% of the variance" is intuitive |

**Code:**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
```

**Common misconception:** R² can be negative if your model is worse than predicting the mean. That's a real signal, not a bug.

---

### 7. Bias-Variance Tradeoff

**One-line definition:** The fundamental tension between "too simple to learn" (bias) and "too complex, memorized noise" (variance).

**Analogy: shooting arrows at a target.**
```
Low Bias, Low Variance:    Low Bias, High Variance:
 (accurate AND consistent)  (accurate on avg, but scattered)
   🎯 🎯                    🎯
   🎯 🎯                    🎯🎯🎯🎯

High Bias, Low Variance:   High Bias, High Variance:
 (all miss same way)        (all over the place)
  🎯                         🎯   🎯
  🎯                            🎯
  🎯  🎯 🎯                  🎯     🎯
```

**How it shows up:**
```
Underfitting (high bias):    Overfitting (high variance):
Train accuracy: 62%           Train accuracy: 99%
Test accuracy:  60%           Test accuracy:  68%
Gap: small                    Gap: huge
Fix: bigger model, more       Fix: more data, regularization,
     features, less reg           simpler model, dropout
```

**The rule of thumb:** If train and test scores are both low → add complexity. If train is high but test is low → add regularization or data.

---

### 8. Stratified Sampling & Class Imbalance

**One-line definition:** When one class is rare, random splits can accidentally hide it — stratify to keep the ratio.

**Why it matters:** In fraud detection, 0.5% of transactions are fraud. A random test split might end up with 0 fraud cases. Stratified split preserves the ratio in every fold.

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Use this instead of plain KFold for classification
```

**For severe imbalance, also consider:**
- `class_weight='balanced'` in sklearn classifiers
- Oversampling minority (SMOTE) or undersampling majority
- **Use PR-AUC, not ROC-AUC, as your headline metric**

---

## How It Actually Works: A Full Evaluation Walkthrough

Say you're building a loan default classifier.

```
Step 1: LOAD data, check class balance
  → 10,000 loans, 8% defaulted → imbalanced

Step 2: SPLIT stratified
  → 70% train, 15% val, 15% test — preserve 8% default ratio in each

Step 3: TRAIN baseline (logistic regression) on train
  → predicts probability of default

Step 4: EVALUATE on validation with multiple metrics
  Accuracy: 92%   ← looks great, but...
  Precision: 0.71
  Recall: 0.35    ← MISSING 65% OF DEFAULTS
  F1: 0.47

Step 5: CROSS-VALIDATE to check stability
  5-fold CV F1: 0.46 ± 0.03 → consistent, not a fluke

Step 6: DIAGNOSE
  Train F1: 0.49, Val F1: 0.47 → no overfitting, high bias
  → try Random Forest

Step 7: RETRAIN, re-evaluate
  Val F1: 0.68 ✓

Step 8: FINAL check on test set (touched ONCE)
  Test F1: 0.66 → close to val → generalizes

Step 9: SHIP or ITERATE
  Business says "recall must be > 80%" → lower threshold, accept lower precision
```

---

## Code in Practice

### Example 1: Minimal Evaluation

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = LogisticRegression(max_iter=5000).fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Example 2: Cross-Validation with Multiple Metrics

```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
results = cross_validate(model, X, y, cv=5, scoring=scoring)

for metric in scoring:
    scores = results[f'test_{metric}']
    print(f"{metric:10s}: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Example 3: Tuning the Decision Threshold

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

probs = model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, probs)

# Find threshold where recall >= 0.80
idx = np.argmin(np.abs(recall - 0.80))
best_threshold = thresholds[idx]
print(f"Threshold for 80% recall: {best_threshold:.3f}")
print(f"Precision at that threshold: {precision[idx]:.3f}")

# Apply it
y_pred_tuned = (probs >= best_threshold).astype(int)
```

---

## Gotchas & Pitfalls

- ❌ **"Accuracy is the best metric"** → ✅ Only when classes are balanced. For imbalanced data, it's meaningless.
- ❌ **"Test set is for tuning"** → ✅ Test set is touched ONCE. Tune on validation.
- ❌ **"I'll just use a single train/test split"** → ✅ For small datasets (<10k rows), always cross-validate — single splits are noisy.
- ❌ **"High training accuracy means I'm done"** → ✅ High training, low test = overfitting. Always compare the two.
- ❌ **"ROC-AUC is enough for imbalanced data"** → ✅ Use PR-AUC for heavy imbalance.
- ❌ **"Random K-fold on time series"** → ✅ Use `TimeSeriesSplit` — you can't train on the future.
- ❌ **"My model has 0.85 F1, I'll deploy it"** → ✅ What does 0.85 mean in *dollars*? Always map metrics to business impact.

---

## When to Use Which Metric

| Situation | Metric |
|---|---|
| Balanced classification | Accuracy, F1 |
| Imbalanced (fraud, disease, rare events) | Precision, Recall, F1, PR-AUC |
| FP and FN have very different costs | Cost-weighted F-beta (F0.5 favors precision, F2 favors recall) |
| Ranking / recommendation | AUC, NDCG, MAP |
| Regression, outliers matter | RMSE |
| Regression, outliers don't matter | MAE |
| Explain to non-technical stakeholders | Accuracy (balanced) or R² (regression) |
| Time series | MAE or MAPE with `TimeSeriesSplit` |

### When NOT to rely on a metric alone

- Small test sets (<500 examples) → confidence intervals are too wide
- Distribution shift (train and prod data differ) → offline metrics mislead
- LLM outputs → traditional metrics don't capture quality; use LLM-as-judge + human eval

---

## Related Concepts (The Map)

- **If you know unit testing, evaluation is like testing — but probabilistic.** You assert the model meets a metric threshold, not an exact output.
- **LLM Evals** are the LLM-world version of this: golden datasets + LLM-as-judge instead of precision/recall. Same principle.
- **Hyperparameter tuning** (`GridSearchCV`, `RandomizedSearchCV`) sits on top of CV — it's CV applied to every candidate config.
- **A/B testing** is online evaluation — measuring real users instead of held-out data.
- **Feature engineering** and evaluation feed each other: new features → re-evaluate → iterate.

---

## Cheat Sheet

### Key terms
- **Training set** — model learns here
- **Validation set** — tune hyperparameters here
- **Test set** — final judgment, touched once
- **Overfitting** — train high, test low
- **Underfitting** — both low
- **Stratified** — preserves class ratio across splits

### Core formulas
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2PR / (P + R)
Accuracy  = (TP + TN) / ALL
RMSE      = √mean((y - ŷ)²)
R²        = 1 - SS_res / SS_total
```

### The 3 things that matter most
1. **Always split before you train.** Leakage destroys everything.
2. **Pick the metric that matches the cost of being wrong**, not the one that looks best.
3. **Compare train vs test scores** — the gap tells you bias vs variance.

---

## Self-Check Questions

1. Your fraud classifier has 99.5% accuracy. Should you ship it?
2. What's the difference between validation set and test set?
3. When would you prefer recall over precision?
4. Your model has 98% train accuracy and 65% test accuracy. What's happening and what do you do?
5. Why is ROC-AUC misleading for heavily imbalanced data?

<details>
<summary>Answers</summary>

1. No — if only 0.5% of transactions are fraud, predicting "not fraud" always gives 99.5% accuracy and catches zero fraud. Check precision, recall, and PR-AUC.
2. Validation is used during tuning (you look at it repeatedly). Test is touched **once** after all tuning is done, to get an unbiased estimate.
3. When false negatives are expensive: cancer screening, fraud detection, security alerts. Missing a real case costs more than a false alarm.
4. Overfitting — model memorized training data. Fix: more data, simpler model, regularization, dropout, or early stopping.
5. Because TPR/FPR don't penalize low precision. You can have high AUC with terrible precision when positives are rare. Use PR-AUC instead.

</details>

---

## Go Deeper

1. **Google's ML Crash Course — Classification** (developers.google.com/machine-learning/crash-course) — the cleanest visual explanation of precision/recall tradeoffs, including an interactive ROC/PR widget.
2. **"A systematic analysis of performance measures for classification tasks"** (Sokolova & Lapalme, 2009) — the paper behind every metric you'll ever use. Read sections 1-3 for the taxonomy.
3. **scikit-learn model evaluation docs** (scikit-learn.org/stable/modules/model_evaluation.html) — the single most practical reference. Bookmark it.
4. **Andrew Ng — "Error Analysis" lecture** (Coursera ML Yearning, free PDF) — teaches you to *read* a confusion matrix and prioritize fixes. Essential for real work.
5. **"Evaluating Machine Learning Models" by Alice Zheng** (O'Reilly, free short book) — 50 pages, covers everything in this doc with more depth. Perfect weekend read.
