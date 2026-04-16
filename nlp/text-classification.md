# Text Classification

## What Is It?

Text classification assigns a **label (category)** to a piece of text. It's the most common NLP task — you've interacted with it thousands of times without realizing.

Give text in → get a category out.

## Frontend Analogy

```javascript
// You've built classification UIs before:
// A form with a <select> dropdown — user picks a category manually.
// Text classification does this AUTOMATICALLY from the text content.

// Frontend "classification" you already know:
function classifyRoute(url) {
  if (url.startsWith('/api'))    return 'api';
  if (url.startsWith('/admin'))  return 'admin';
  return 'public';
}

// NLP classification: same idea, but from FREE TEXT, not structured URLs
classify("This product broke after one day!")  → "negative"
classify("Best purchase I've ever made!")      → "positive"
```

## Real-World Examples

| Task | Input | Output Labels |
|------|-------|---------------|
| **Spam detection** | Email text | spam / not spam |
| **Sentiment analysis** | Product review | positive / negative / neutral |
| **Topic classification** | News article | sports / politics / tech / science |
| **Intent detection** | Chatbot message | greeting / question / complaint / order |
| **Language detection** | Any text | English / French / Spanish / ... |
| **Toxicity detection** | Comment | toxic / not toxic |
| **Support ticket routing** | Customer email | billing / technical / returns / other |

## The Three Approaches (Old → Modern)

### Approach 1: Bag of Words + Classical ML (Old but Educational)

Turn text into word counts, then use a classifier:

```
"I love this movie"  → {I:1, love:1, this:1, movie:1}  → [1,1,1,1,0,0,0...]
"I hate this movie"  → {I:1, hate:1, this:1, movie:1}  → [1,0,1,1,1,0,0...]

Feed these vectors into Naive Bayes / Logistic Regression / SVM
```

**TF-IDF** (Term Frequency - Inverse Document Frequency) is the improved version:
- Words that appear everywhere ("the", "is") get **low weight**
- Words that appear in few documents ("excellent", "terrible") get **high weight**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Simple but effective pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', MultinomialNB()),
])

texts = ["Great product!", "Terrible quality", "Love it", "Waste of money"]
labels = ["positive", "negative", "positive", "negative"]

model.fit(texts, labels)
print(model.predict(["This is amazing!"]))  # → ["positive"]
```

**When to use:** Quick baseline, small datasets, when you need speed and simplicity.

### Approach 2: Pretrained Transformer (The Standard)

Use BERT or similar model. Fine-tune on your labeled data:

```python
from transformers import pipeline

# Zero-shot: no training needed! Just describe your categories.
classifier = pipeline("zero-shot-classification")

result = classifier(
    "The new MacBook Pro has incredible battery life",
    candidate_labels=["technology", "sports", "politics", "food"]
)
print(result['labels'][0])    # → "technology"
print(result['scores'][0])    # → 0.97
```

**When to use:** The default choice for most projects. Fine-tune for best accuracy.

### Approach 3: LLM Prompting (Newest)

Just ask Claude or GPT:

```python
# Using an LLM API (no training at all)
prompt = """Classify this review as positive, negative, or neutral.
Review: "The food was okay but the service was incredibly slow."
Classification:"""

# LLM responds: "negative"
```

**When to use:** When you need flexibility, few examples, or the task changes frequently. More expensive per prediction.

## Comparison: Which Approach?

| | Bag of Words + ML | Fine-tuned Transformer | LLM Prompting |
|---|---|---|---|
| **Accuracy** | Good (80-85%) | Best (90-95%) | Great (85-93%) |
| **Speed** | Very fast | Fast | Slow (API call) |
| **Cost** | Free | GPU for training | $ per prediction |
| **Training data needed** | 100s-1000s | 100s-1000s | 0 (zero-shot) |
| **Setup effort** | Low | Medium | Very low |
| **Best for** | Baselines, fast prototypes | Production systems | Prototypes, flexible tasks |

## Sentiment Analysis — Deep Dive (Most Common Task)

### Binary Sentiment (Positive / Negative)

```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

results = sentiment([
    "I absolutely love this product!",
    "Worst experience ever. Never again.",
    "It's okay, nothing special.",
])

for r in results:
    print(f"{r['label']}: {r['score']:.2%}")
# POSITIVE: 99.87%
# NEGATIVE: 99.72%
# NEGATIVE: 56.12%   ← model is uncertain (near 50%) — makes sense!
```

### Multi-class Sentiment

```python
from transformers import pipeline

# Model trained on 6 emotions
emotion = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion"
)

result = emotion("I can't believe they cancelled the concert!")
print(result)  # [{'label': 'anger', 'score': 0.94}]
```

## Multi-Label vs Multi-Class

```
Multi-class:  Pick ONE label from many
  "What sport is this about?" → football (not baseball, not tennis)

Multi-label:  Pick MULTIPLE labels
  "Tag this article" → [technology, business, AI]  (can have several)
```

```python
# Multi-label: use sigmoid (independent probabilities per label)
# Multi-class: use softmax (probabilities sum to 1)

# Multi-label in zero-shot:
classifier = pipeline("zero-shot-classification")
result = classifier(
    "Apple released a new AI-powered MacBook for developers",
    candidate_labels=["technology", "business", "AI", "sports"],
    multi_label=True,  # ← allow multiple labels
)
# technology: 0.95, AI: 0.91, business: 0.72, sports: 0.01
```

## Evaluation Metrics for Classification

Don't just use accuracy — especially with imbalanced data:

```
Dataset: 950 "not spam" + 50 "spam"
Model predicts EVERYTHING as "not spam"
Accuracy: 95%!!! ← but it never catches spam. Useless.
```

| Metric | Formula | Use When |
|--------|---------|----------|
| **Accuracy** | correct / total | Balanced classes |
| **Precision** | true positives / predicted positives | False alarms are costly (spam filter) |
| **Recall** | true positives / actual positives | Missing positives is costly (disease, fraud) |
| **F1** | harmonic mean of precision & recall | You need both |

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
#               precision  recall  f1-score  support
#   negative       0.89    0.92    0.90      250
#   positive       0.91    0.88    0.89      250
#   accuracy                       0.90      500
```

## Fine-Tuning for Classification (Full Example)

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import load_dataset

# 1. Load dataset
dataset = load_dataset("imdb")  # 50K movie reviews (pos/neg)

# 2. Load pretrained model + tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2   # positive / negative
)

# 3. Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

dataset = dataset.map(tokenize, batched=True)

# 4. Train
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,          # small LR for fine-tuning!
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()

# 5. Use the fine-tuned model
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
classifier("This movie was absolutely brilliant!")
# → [{'label': 'POSITIVE', 'score': 0.9994}]
```

## Key Takeaway

Text classification is the **bread and butter of NLP**. For quick prototypes, use `pipeline("zero-shot-classification")` — no training needed. For production, fine-tune a DistilBERT or BERT model on your labeled data. For maximum flexibility, prompt an LLM. Start with the simplest approach that works, then upgrade only if accuracy isn't good enough.
