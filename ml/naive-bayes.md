# Naive Bayes

## What Is It?

Naive Bayes is a **probability-based classifier** built on Bayes' Theorem. It calculates the probability of each class given the input features and picks the most likely one. The "naive" part: it assumes all features are **independent** of each other.

Think of it like a spam filter that checks each word separately: "viagra" → probably spam, "meeting" → probably not spam, "free viagra meeting" → combines all the individual probabilities.

## Real-World Examples

- **Spam detection** — the classic Naive Bayes application
- **Sentiment analysis** — positive or negative review?
- **Document classification** — categorize news articles by topic
- **Medical diagnosis** — given symptoms, what's the likely disease?

## How It Works (Step by Step)

### 1. Bayes' Theorem (The Core Idea)

```
P(class | features) = P(features | class) × P(class) / P(features)
```

In plain English:

```
Probability it's spam    How often spam emails     How common     
given these words     =  contain these words    ×  spam is       
                         ─────────────────────────────────────
                         How common these words are overall
```

### 2. The "Naive" Assumption

Assume each feature contributes **independently**:

```
P("free" AND "money" | spam) = P("free" | spam) × P("money" | spam)
```

This is obviously wrong in reality ("free" and "money" often appear together), but it works surprisingly well in practice. The ranking of classes is usually correct even if the exact probabilities are off.

### 3. Worked Example: Spam Filter

Training data:

| Email | Contains "free" | Contains "money" | Spam? |
|-------|:-:|:-:|:-:|
| 1 | Yes | Yes | Spam |
| 2 | Yes | No  | Spam |
| 3 | No  | Yes | Spam |
| 4 | No  | No  | Not Spam |
| 5 | No  | No  | Not Spam |
| 6 | Yes | No  | Not Spam |

From the data:
- P(spam) = 3/6 = 0.5
- P(not spam) = 3/6 = 0.5
- P("free" | spam) = 2/3 = 0.67
- P("free" | not spam) = 1/3 = 0.33
- P("money" | spam) = 2/3 = 0.67
- P("money" | not spam) = 0/3 = 0.0 (we add smoothing → 0.1)

New email contains "free" AND "money":
```
P(spam | free, money)     ∝ 0.67 × 0.67 × 0.5 = 0.224
P(not spam | free, money) ∝ 0.33 × 0.1  × 0.5 = 0.017

Normalize: spam = 0.224 / (0.224 + 0.017) = 93%

Prediction: SPAM (93% confident)
```

## Types of Naive Bayes

| Type | Use When | Features |
|------|----------|----------|
| **Multinomial** | Text classification | Word counts / TF-IDF |
| **Bernoulli** | Binary features | Word present (yes/no) |
| **Gaussian** | Continuous numbers | Heights, temperatures, etc. |

**For text, use Multinomial.** It's the standard for NLP tasks.

## When to Use It

| Good For | Bad For |
|----------|---------|
| Text classification (spam, sentiment, topics) | When features are highly dependent |
| Very fast training and prediction | When you need precise probabilities |
| Works well with small datasets | Complex relationships between features |
| High-dimensional data (many features) | Numeric data with complex patterns |
| Baseline model for NLP tasks | Image or audio classification |

## Why It Works Despite Being "Wrong"

The independence assumption is almost never true. "New" and "York" are clearly not independent. But Naive Bayes only needs to **rank** classes correctly, not get exact probabilities right. And for ranking, the simplified math usually gives the right order.

It's like estimating commute time by adding up each segment independently — you ignore that traffic on one road affects another, but your total estimate is still useful.

## Python Example

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Sample email data
emails = [
    "free money now", "win free lottery", "free prize click",
    "meeting tomorrow", "project update attached", "lunch at noon",
    "free consultation meeting", "schedule review call",
    "claim your free gift", "quarterly report ready",
]
labels = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]  # 1=spam, 0=not spam

# Convert text to word counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train
model = MultinomialNB()
model.fit(X, labels)

# Predict new emails
new_emails = ["free money win", "meeting schedule update"]
X_new = vectorizer.transform(new_emails)
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

for email, pred, prob in zip(new_emails, predictions, probabilities):
    label = "SPAM" if pred == 1 else "NOT SPAM"
    confidence = max(prob) * 100
    print(f'"{email}" → {label} ({confidence:.0f}% confident)')
```

## Key Takeaway

Naive Bayes is **fast, simple, and surprisingly effective for text**. It's the first thing to try for any text classification task. The independence assumption is technically wrong but practically useful. If you're building a spam filter, sentiment analyzer, or document classifier, start here.
