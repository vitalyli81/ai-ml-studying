# Logistic Regression

## What Is It?

Despite the name, logistic regression is for **classification** (yes/no, spam/not spam), not regression. It predicts the **probability** that something belongs to a category.

Think of it as: "What are the chances this email is spam?"

## Real-World Examples

- Is this email **spam or not**?
- Will a customer **churn or stay**?
- Is this transaction **fraudulent or legitimate**?
- Does a patient **have a disease or not**?

## How It Works (Step by Step)

### 1. Start Like Linear Regression

Compute a score using weights:

```
z = w1*x1 + w2*x2 + ... + b
```

Problem: this score can be any number (-infinity to +infinity), but we need a probability (0 to 1).

### 2. The Sigmoid Function (The Key Trick)

The **sigmoid** squashes any number into the range 0 to 1:

```
probability = 1 / (1 + e^(-z))
```

```
Input z:   -10    -2     0     2     10
Output:    0.00   0.12   0.50  0.88  1.00
```

It makes an S-shaped curve:
- Very negative z → probability near 0
- z = 0 → probability = 0.5 (the decision boundary)
- Very positive z → probability near 1

### 3. Making a Decision

Pick a **threshold** (usually 0.5):

```
if probability >= 0.5 → predict YES (class 1)
if probability <  0.5 → predict NO  (class 0)
```

You can adjust the threshold:
- **Lower threshold (0.3)** → catch more positives, but more false alarms (good for disease detection)
- **Higher threshold (0.7)** → fewer false alarms, but miss some positives (good for spam filtering)

### 4. How It Learns

Uses **log loss** (cross-entropy) instead of MSE:

```
Log Loss = -[y * log(p) + (1-y) * log(1-p)]
```

In plain English:
- If actual = 1 and you predicted 0.99 → small loss (good)
- If actual = 1 and you predicted 0.01 → huge loss (bad)

The model adjusts weights using gradient descent to minimize this loss.

## Linear Regression vs Logistic Regression

| Linear Regression | Logistic Regression |
|-------------------|---------------------|
| Predicts a **number** | Predicts a **probability** (0 to 1) |
| Output: any value | Output: 0 to 1 (via sigmoid) |
| Loss: Mean Squared Error | Loss: Log Loss |
| Example: house price | Example: spam or not |

## When to Use It

| Good For | Bad For |
|----------|---------|
| Binary classification (2 classes) | Complex non-linear boundaries |
| When you need probability scores | Image/audio classification |
| Understanding feature importance | When features interact in complex ways |
| Baseline model — try this first | When classes overlap heavily |

## Multiclass Classification

For more than 2 classes (e.g., cat/dog/bird), two strategies:

- **One-vs-Rest (OvR)** — train one model per class. "Is it a cat?" "Is it a dog?" "Is it a bird?" Pick the highest probability.
- **Softmax Regression** — generalize sigmoid to multiple classes. All probabilities sum to 1.

## Evaluation Metrics

Don't just use accuracy! Especially with imbalanced data (99% not fraud, 1% fraud):

| Metric | What It Measures | When It Matters |
|--------|-----------------|-----------------|
| **Accuracy** | % correct overall | Balanced classes |
| **Precision** | Of predicted positives, how many were right? | When false alarms are costly (spam) |
| **Recall** | Of actual positives, how many did you find? | When missing positives is costly (disease) |
| **F1 Score** | Balance of precision and recall | When you need both |
| **AUC-ROC** | Overall ranking quality | Comparing models |

## Python Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Data: [study_hours, sleep_hours] → pass(1) or fail(0)
X = [[2, 4], [3, 5], [5, 7], [7, 6], [8, 8], [1, 3]]
y = [0, 0, 1, 1, 1, 0]

# Train
model = LogisticRegression()
model.fit(X, y)

# Predict
new_student = [[4, 6]]
probability = model.predict_proba(new_student)
prediction = model.predict(new_student)

print(f"Probability of passing: {probability[0][1]:.1%}")
print(f"Prediction: {'Pass' if prediction[0] == 1 else 'Fail'}")
print(f"\nFeature importance:")
print(f"  Study hours weight: {model.coef_[0][0]:.2f}")
print(f"  Sleep hours weight: {model.coef_[0][1]:.2f}")
```

## Key Takeaway

Logistic regression is the **go-to baseline for classification**. It's fast, interpretable, and surprisingly effective. Always try it first before jumping to complex models — you'll be surprised how often it's good enough.
