# Logistic Regression

## TL;DR

Despite the name, logistic regression is a **classification** model — it predicts a category (spam/not spam), not a number. It works exactly like linear regression, but wraps the output in a sigmoid function to squash it into a probability between 0 and 1. If probability ≥ 0.5, predict class 1; otherwise class 0. It's the go-to baseline for any binary classification problem.

> 💡 **Key Insight:** The name is confusing — it's really "logistic classification." It predicts the *probability* that something belongs to a class. That probability is then thresholded to make a decision.

---

## The Mental Model

Think of a **doctor making a diagnosis** based on test results.

The doctor looks at blood pressure, age, cholesterol, and calculates a risk score. If the risk score is above a certain threshold, they say "high risk of heart disease." Below it: "low risk." They're not predicting a number — they're predicting a binary outcome.

Mapping:
- Test results (blood pressure, cholesterol) → input features
- Risk score calculation → the linear equation `z = w₁x₁ + w₂x₂ + b`
- Sigmoid transformation → converting risk score to a probability (0-1)
- Threshold (50% risk) → the decision boundary
- "High risk / Low risk" → the class prediction

Logistic regression is a linear decision boundary: "I draw a straight line; everything on this side is class 0, everything on the other side is class 1."

---

## Why It Exists

### The Problem

Linear regression can predict probabilities, but it has a fatal flaw for classification:

```
Linear regression for spam detection:
  Input: "free money win" → predicted value: 1.4   ← what does 1.4 even mean?
  Input: "meeting agenda" → predicted value: -0.2  ← negative probability?

Linear regression has no bounds. It predicts ANY number.
A probability MUST be between 0 and 1.
```

### The Solution

Add a sigmoid function that mathematically squashes any number into [0, 1]. Now the model outputs a valid probability.

### What Changed

Logistic regression made probabilistic classification practical and interpretable. It's still used in production today for credit scoring, medical diagnosis, and A/B test analysis because its weights are directly interpretable as "log-odds" — you can explain every decision.

---

## Core Concepts

### 1. The Sigmoid Function

**One-line definition:** A mathematical S-curve that converts any number to a probability between 0 and 1.

**Analogy:** Think of a volume knob. No matter how hard you crank it (positive infinity) or down (negative infinity), the output is capped between 0 (silent) and 1 (max). The sigmoid is that cap.

```
Formula: σ(z) = 1 / (1 + e^(-z))

Input z → Output probability:
  z = -10  →  0.00005  (nearly 0, very confident: class 0)
  z = -2   →  0.12     (probably class 0)
  z =  0   →  0.50     (exactly 50/50 — the decision boundary)
  z = +2   →  0.88     (probably class 1)
  z = +10  →  0.99995  (nearly 1, very confident: class 1)

    probability
    1 |            _________
      |          /
    .5|─────────/────────── ← threshold
      |       /
    0 |______/
      └────────────────── z
            0
```

**Common misconception:** The output is a prediction, not a probability. The output IS a probability. It's the model's confidence that the input belongs to class 1.

---

### 2. Decision Boundary

**One-line definition:** The line (or surface) that separates class 0 from class 1 in feature space.

**Analogy:** Imagine drawing a line on a map dividing "safe neighborhoods" from "risky neighborhoods." Points on one side are predicted class 0; on the other side, class 1.

```
Feature space with 2 inputs (x₁, x₂):

  x₂ ↑
     |  × × ×          (class 1: spam)
     |    × ×
     |─────────────────  ← decision boundary (where z = 0, prob = 0.5)
     |  o  o
     |o    o o          (class 0: not spam)
     └──────────── x₁

Logistic regression can ONLY draw straight-line boundaries.
For curved boundaries, you need non-linear models.
```

**Common misconception:** Logistic regression can only handle 2 classes. False — multiclass logistic regression uses Softmax, which handles any number of classes.

---

### 3. Log Loss (Binary Cross-Entropy)

**One-line definition:** The loss function that penalizes confident wrong predictions extremely harshly.

**Analogy:** A doctor who says "I'm 99% sure this is cancer" when it's not should be penalized much more than one who says "I'm 55% sure." Log loss does exactly this.

```
Log Loss = -[y × log(p) + (1-y) × log(1-p)]

Where y = actual label (0 or 1), p = predicted probability

Examples:
  y=1, p=0.95 → loss = -log(0.95) = 0.05  (correct, confident → tiny loss)
  y=1, p=0.50 → loss = -log(0.50) = 0.69  (correct, uncertain → moderate loss)
  y=1, p=0.05 → loss = -log(0.05) = 3.00  (wrong AND confident → huge loss!)

The model learns to be calibrated — confident only when it's right.
```

**Common misconception:** Log loss and accuracy measure the same thing. Accuracy counts right/wrong. Log loss measures HOW confident the model was. A model with 90% accuracy but terrible log loss is overconfident on its wrong predictions.

---

### 4. Classification Threshold

**One-line definition:** The probability cutoff above which you predict class 1 (default is 0.5, but you can change it).

**Analogy:** A fire alarm threshold. Set it too sensitive (0.1) and it triggers on toast. Set it too lenient (0.9) and it misses real fires. The right threshold depends on the cost of each type of error.

```
Threshold = 0.5 (default):
  prob ≥ 0.5 → class 1
  prob < 0.5 → class 0

Threshold = 0.3 (for disease detection: catching cases matters more):
  prob ≥ 0.3 → positive (fewer false negatives, more false positives)
  → You catch more sick patients but flag more healthy ones too

Threshold = 0.8 (for spam: false alarms are annoying):
  prob ≥ 0.8 → spam (fewer false positives, more false negatives)
  → Less likely to mark real email as spam
```

**Common misconception:** You should always use 0.5 as the threshold. The optimal threshold depends on the business cost of false positives vs false negatives. In medicine, you'd rather have false positives (unnecessary tests) than false negatives (missed disease).

---

### 5. Evaluation Metrics for Classification

**One-line definition:** Because accuracy alone is misleading on imbalanced datasets, use precision, recall, and F1.

```
Confusion matrix:
              Predicted: No  |  Predicted: Yes
Actual: No    True Negative  |  False Positive (false alarm)
Actual: Yes   False Negative |  True Positive  (correct catch)

Precision = TP / (TP + FP)  → "Of all predicted spam, how many were really spam?"
Recall    = TP / (TP + FN)  → "Of all actual spam, how many did we catch?"
F1        = 2 × (P × R) / (P + R)  → harmonic mean of precision and recall

Example — fraud detection (1% of transactions are fraud):
  Naive model that predicts "not fraud" always:
    Accuracy: 99% ← looks great!
    Recall:    0% ← catches 0 fraudulent transactions ← terrible

  Logistic regression:
    Accuracy: 96%, Precision: 70%, Recall: 80% ← actually useful
```

**Common misconception:** High accuracy = good model. On imbalanced data, accuracy is meaningless. Always check precision, recall, and F1.

---

## How It Actually Works (Step-by-Step)

Let's predict whether a student passes (1) or fails (0) based on study hours:

```
Data:
  Study Hours | Pass?
  ────────────────────
  1           | 0
  2           | 0
  3           | 0
  5           | 1
  7           | 1
  8           | 1

Step 1: Initialize weights
  w = 0,  b = 0

Step 2: Compute z for each student
  z = w × hours + b = 0 × hours + 0 = 0 for all students

Step 3: Apply sigmoid
  probability = σ(0) = 0.5 for all students  ← useless 50/50

Step 4: Compute log loss
  For student who studied 1hr (actual: fail, y=0):
    loss = -[(0 × log(0.5)) + (1 × log(0.5))] = 0.693

Step 5: Gradient descent — adjust w and b
  After gradient update: w = 0.3, b = -1.2

Step 6: Repeat 1000 times
  ...converges to...
  w = 2.1,  b = -9.0

Step 7: Final model
  z = 2.1 × hours - 9.0
  prob = σ(z)

  3 hours: z = -2.7, prob = 0.06 → FAIL ✓
  5 hours: z = 1.5,  prob = 0.82 → PASS ✓
  4 hours: z = -0.6, prob = 0.35 → FAIL (borderline)
```

---

## Code in Practice

### 1. Hello World — Binary Classification

```python
from sklearn.linear_model import LogisticRegression

# [study_hours, sleep_hours] → pass(1) or fail(0)
X = [[2, 4], [3, 5], [5, 7], [7, 6], [8, 8], [1, 3]]
y = [0, 0, 1, 1, 1, 0]

model = LogisticRegression()
model.fit(X, y)

# Predict a new student (4 hours study, 6 hours sleep)
prob = model.predict_proba([[4, 6]])[0]
print(f"Probability of passing: {prob[1]:.1%}")
print(f"Prediction: {'Pass' if model.predict([[4, 6]])[0] == 1 else 'Fail'}")
```

### 2. Practical — With Proper Evaluation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Load real dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Scale features (logistic regression is sensitive to scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=data.target_names))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
```

### 3. Real-World Pattern — Threshold Tuning

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

# Get probabilities (not just predictions)
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# Find precision and recall at every possible threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# Find threshold where recall ≥ 0.95 (critical for medical use)
target_recall = 0.95
idx = np.argmin(np.abs(recalls - target_recall))
best_threshold = thresholds[idx]
print(f"Threshold for {target_recall:.0%} recall: {best_threshold:.3f}")
print(f"Precision at that threshold: {precisions[idx]:.3f}")

# Apply custom threshold
y_custom = (y_probs >= best_threshold).astype(int)
```

---

## Gotchas & Pitfalls

```
❌ Not scaling features
   Logistic regression uses gradient descent — unscaled features (range 0-1000)
   dominate and slow convergence, or worse, never converge
✅ Always StandardScaler() before LogisticRegression

❌ Using accuracy as the only metric for imbalanced data
   99% "not fraud" predictions look great but catch nothing
✅ Use precision, recall, F1 — or AUC-ROC for threshold-independent comparison

❌ Forgetting that logistic regression needs linear decision boundaries
   If your data has circular or complex separations, it will underfit badly
✅ Plot your data first; if classes aren't linearly separable, use tree-based models

❌ Ignoring max_iter warnings
   scikit-learn's default max_iter=100 often doesn't converge
✅ Always set max_iter=1000 or until the convergence warning disappears

❌ Treating the 0.5 threshold as sacred
   Real business problems have asymmetric costs for false positives vs negatives
✅ Use precision_recall_curve() to find the threshold that fits your use case

❌ Not checking feature coefficients
   Logistic regression is interpretable — use it! The weights tell you what matters.
✅ model.coef_[0] gives feature importances; large absolute value = more important
```

---

## When to Use / When NOT to Use

### Use Logistic Regression When:
- Binary classification (yes/no, spam/not spam, pass/fail)
- You need to explain predictions — weights show exactly which features matter
- You need probability outputs, not just class labels
- You want a fast, lightweight baseline before trying complex models

### Don't Use Logistic Regression When:
- The decision boundary is non-linear (classes aren't separable by a line)
- You have many interacting features (tree-based models handle this better)
- Your data is images or audio (use CNNs)
- You need to handle multi-class natively without tricks (use Random Forest or neural networks)

---

## Related Concepts

| Concept | Connection |
|---------|------------|
| Linear Regression | Same underlying equation — logistic adds sigmoid on top to output probabilities |
| Softmax / Multiclass | Generalization of logistic regression to 3+ classes |
| Gradient Descent | The same learning algorithm — minimizes log loss instead of MSE |
| Decision Trees | Also a classifier, but finds non-linear boundaries and needs no scaling |
| ROC/AUC | The standard way to compare logistic regression models at all thresholds |

---

## Cheat Sheet

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train_scaled, y_train)

model.predict(X_test)             # class labels (0 or 1)
model.predict_proba(X_test)[:,1]  # probability of class 1
model.coef_[0]                    # feature weights (interpretable!)
model.intercept_                  # bias term

Key hyperparameters:
  C = 1/λ   → inverse regularization strength (lower C = more regularization)
  max_iter  → training iterations (increase if convergence warning)
  solver    → 'lbfgs' (default, small data), 'saga' (large data)

Metrics to report:
  classification_report(y_test, y_pred)  # precision, recall, F1 for all classes
  confusion_matrix(y_test, y_pred)       # see which classes are confused

Remember:
  1. Scale your features — always
  2. Logistic regression draws straight-line boundaries only
  3. Tune threshold based on the cost of false positives vs false negatives
```

---

## Self-Check Questions

<details>
<summary>Click to reveal answers</summary>

**Q1: Why is logistic regression called "regression" if it's for classification?**
Historically, it was framed as regression of probabilities — you're fitting a regression line to the log-odds. The output is a probability (a continuous number between 0 and 1), which is then thresholded to make a class prediction. The name stuck.

**Q2: What happens if you have heavily imbalanced classes (99% class 0, 1% class 1)?**
A naive model predicting class 0 always achieves 99% accuracy but is useless. You should use class_weight='balanced' in scikit-learn, which weights the minority class more heavily during training, and evaluate with precision/recall rather than accuracy.

**Q3: Why does scaling matter for logistic regression?**
Logistic regression uses gradient descent. If one feature has range [0, 10000] and another [0, 1], the gradient with respect to the large-range feature dominates. The model wastes many iterations correcting for this imbalance. Scaling puts all features on the same playing field.

**Q4: What does a large positive weight coefficient mean?**
When that feature is high, the model is more confident the prediction is class 1. Example: a large positive weight on "contains 'free'" means the presence of that word strongly pushes the spam probability up.

**Q5: How is logistic regression different from a decision tree for classification?**
Logistic regression learns a linear decision boundary — it can only separate classes with a straight line (or hyperplane in higher dimensions). A decision tree learns arbitrary boundaries using nested if-else splits. Decision trees can capture non-linear patterns; logistic regression can't. But logistic regression is more interpretable and often better with small datasets.

</details>

---

## Go Deeper

| Resource | Why It's Worth Your Time |
|----------|--------------------------|
| [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) | Visual, intuitive explanation of sigmoid and log loss. The clearest 15-minute intro. |
| [Scikit-learn Logistic Regression docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | All parameters explained with examples. Especially useful for understanding solver choices. |
| [ROC and AUC Explained](https://www.youtube.com/watch?v=4jRBRDbJemM) | StatQuest again — master the threshold-independent evaluation metric every ML engineer needs. |
| *Hands-On ML* Ch. 3 (Classification) — Aurélien Géron | The best chapter on classification metrics, confusion matrices, and PR curves with code. |
| [Kaggle: Titanic Competition](https://www.kaggle.com/c/titanic) | The classic beginner ML competition — logistic regression is the standard first solution. Great practice. |
