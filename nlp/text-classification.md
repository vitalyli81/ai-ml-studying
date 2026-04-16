# Text Classification

## 1. TL;DR

Text classification assigns a **label to a piece of text** — spam/not-spam, positive/negative, topic category, intent. It's the most common NLP task. Three approaches in order of complexity: (1) `pipeline("zero-shot-classification")` — no training, works instantly; (2) fine-tuned DistilBERT — best accuracy for fixed labels; (3) LLM prompting — most flexible, most expensive. Start with zero-shot and only upgrade when accuracy demands it.

---

## 2. The Mental Model

> 💡 **Think of it like this:** Text classification is a **sorting machine for language**.

A postal worker sorts letters by reading the address and dropping them in the right bin. Text classification reads text and drops it in the right category bin — automatically, at scale, in milliseconds.

| Real world | Technical concept |
|---|---|
| Postal worker reading the address | Model reading the text |
| Address → correct mailbox | Text → predicted label |
| Sorting rules (city names, ZIP codes) | Model weights (learned from training data) |
| Wrong address → returned letter | Wrong prediction → misclassification |
| Separate bins for each city | One output neuron per class |

---

## 3. Why It Exists

**The problem:** Humans generate text at a scale no human team can label fast enough. A social media platform gets millions of posts per day. An e-commerce site gets thousands of reviews per hour. Manual labeling is impossible.

**What came before:** Keyword rules — if the email contains "CONGRATULATIONS" and "CLAIM YOUR PRIZE," mark as spam. Worked for obvious cases but failed for anything subtle ("The battery life is not that bad" → positive or negative?).

**What changed:** Statistical ML (Naive Bayes, SVMs on TF-IDF features) brought probabilistic decisions. Then Transformers (2018+) brought context-awareness — "not bad" = positive, even though "bad" is in there. Now you can classify with near-human accuracy on most tasks.

---

## 4. Core Concepts

### Label Space

**One-line definition:** The fixed set of possible output categories the model chooses from.

**Analogy:** Like a multiple-choice test — the model picks from A, B, C, D. It can't invent option E.

```
Binary:      {spam, not_spam} — exactly 2 choices
Multi-class: {sports, politics, tech, science, culture} — one choice from N
Multi-label: {technology, business, AI} — zero to N choices simultaneously
```

**Common misconception:** ❌ "The model understands category names" → ✅ During training, categories are just integers (0, 1, 2). A model trained on IMDB doesn't inherently understand "positive" and "negative" — it learned that certain patterns map to label 0 or 1.

---

### TF-IDF (The Classical Baseline)

**One-line definition:** A score that measures how important a word is to a document relative to a corpus.

**Analogy:** If every book in a library mentions "the," that word tells you nothing specific. But if only cookbooks mention "sauté," that word is very diagnostic.

```
TF (Term Frequency):  how often does "excellent" appear in this review?
IDF (Inverse DF):     how rare is "excellent" across all reviews?
TF-IDF = TF × IDF

"excellent" in a review: common word for praise → high TF-IDF → strong signal
"the" in a review:       appears everywhere → low IDF → near-zero TF-IDF
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression()),
])

model.fit(train_texts, train_labels)
preds = model.predict(["This is absolutely terrible!"])  # → ["negative"]
```

**Common misconception:** ❌ "TF-IDF understands meaning" → ✅ TF-IDF is purely statistical — it counts words. "Not bad" and "bad" both score high on "bad." Transformers understand the "not" changes the meaning.

---

### Zero-Shot Classification

**One-line definition:** Classify text into categories you define at inference time — no training data needed.

**Analogy:** Like asking a knowledgeable friend to sort things by your custom criteria. You describe the categories; they apply their world knowledge to sort.

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

result = classifier(
    "The new MacBook Pro has incredible battery life",
    candidate_labels=["technology", "sports", "politics", "food"],
)
# technology: 0.97, sports: 0.01, politics: 0.01, food: 0.01
```

**Common misconception:** ❌ "Zero-shot means the model has never seen any classification" → ✅ The model was trained on NLI (natural language inference) tasks that teach it to reason about entailment. "The MacBook has incredible battery life" *entails* "technology."

---

### Fine-Tuning for Classification

**One-line definition:** Take a pretrained BERT model and add a classification head, then train on your labeled data.

**Analogy:** Like training an expert intern on your company's specific style guide. They already know the language; you just teach them your categories.

```
Pretrained BERT
  [CLS] token → 768-dim vector → Linear(768, 2) → softmax → [positive: 0.98, negative: 0.02]
```

Only the last linear layer (and sometimes a few BERT layers) are updated during fine-tuning.

**Common misconception:** ❌ "Fine-tuning means retraining from scratch" → ✅ Fine-tuning starts from pretrained weights and makes small adjustments. Use learning rate 2e-5 (100x smaller than training from scratch).

---

### Evaluation Metrics

**One-line definition:** Accuracy alone is misleading — use precision, recall, and F1 to understand model behavior.

**Analogy:** A spam filter that marks nothing as spam has 100% precision (zero false alarms) but 0% recall (misses all spam). You need both.

```
Accuracy  = (TP + TN) / all    → misleading on imbalanced data
Precision = TP / (TP + FP)     → "of what I flagged, how much was right?"
Recall    = TP / (TP + FN)     → "of all actual positives, how many did I catch?"
F1        = 2 × (P × R)/(P+R)  → harmonic mean — balances both

Example: 950 "not spam" + 50 "spam"
  Predict ALL as "not spam":
    Accuracy: 95% ← looks great!
    Recall:    0% ← catches zero spam
    F1:       ~0% ← terrible
```

**Common misconception:** ❌ "High accuracy = good model" → ✅ Always check per-class metrics. Imbalanced datasets make accuracy meaningless — a model can be 95% accurate while being completely useless.

---

## 5. How It Actually Works (Step-by-Step)

Let's trace "I absolutely love this product!" through a fine-tuned DistilBERT classifier:

```
INPUT: "I absolutely love this product!"

Step 1: Tokenize
  ["[CLS]", "i", "absolutely", "love", "this", "product", "!", "[SEP]"]
  → [101, 1045, 7078, 2293, 2023, 4031, 999, 102]

Step 2: Embed each token
  Each token ID → 768-dimensional vector

Step 3: Pass through 6 DistilBERT layers (bidirectional)
  All tokens attend to all other tokens
  "[CLS]" learns: "absolutely love" + exclamation → very positive

Step 4: Extract [CLS] output vector
  [0.92, -0.3, 0.65, ..., 0.1]  (768 numbers)

Step 5: Classification head (linear layer)
  768 → 2 (positive, negative)
  [4.2, -3.8]  (raw logits)

Step 6: Softmax
  positive: e^4.2 / (e^4.2 + e^-3.8) = 0.9998
  negative: 0.0002

Step 7: Output
  POSITIVE (99.98% confidence)
```

> 💡 **Key Insight:** The model never explicitly learned the word "love" means positive. It learned that the *pattern* of embeddings produced by sentences containing "love," "great," "excellent," etc. maps to label 1, through thousands of training examples.

---

## 6. Code in Practice

### Minimal: Zero-shot (no training needed)

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

text = "My order arrived 3 days late and the packaging was damaged"
labels = ["shipping", "product quality", "billing", "general complaint"]

result = classifier(text, candidate_labels=labels)
print(result['labels'][0])   # → "shipping"
print(result['scores'][0])   # → 0.72
```

### Practical: Sentiment analysis with pretrained model

```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

reviews = [
    "Absolutely love it! Best purchase this year.",
    "Arrived broken. Waste of money.",
    "It's okay, does what it says.",
]

for review in reviews:
    result = sentiment(review)[0]
    print(f"{result['label']:8} ({result['score']:.0%}): {review[:50]}")

# POSITIVE (100%): Absolutely love it! Best purchase this year.
# NEGATIVE (100%): Arrived broken. Waste of money.
# NEGATIVE ( 57%): It's okay, does what it says.   ← uncertain!
```

### Real-world pattern: Fine-tune for custom classification

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import load_dataset
import numpy as np
import evaluate

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

dataset = load_dataset("imdb")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=512)

dataset = dataset.map(tokenize, batched=True)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,        # small LR for fine-tuning
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

# Use fine-tuned model
from transformers import pipeline
my_classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
my_classifier("This product is outstanding!")
# → [{'label': 'POSITIVE', 'score': 0.9994}]
```

---

## 7. Gotchas & Pitfalls

❌ **Evaluating only on accuracy with imbalanced data** → ✅ Always check per-class precision, recall, and F1 with `classification_report`. A model predicting only the majority class can score 90%+ accuracy.

❌ **Skipping the baseline** → ✅ Always compare against: (1) majority class prediction, (2) TF-IDF + Logistic Regression. If your fancy model doesn't beat these, something is wrong.

❌ **Using zero-shot for high-stakes production** → ✅ Zero-shot is great for prototyping and flexible tasks. For production classification with fixed labels, fine-tune — you'll gain 10-15% accuracy.

❌ **Confusing multi-class and multi-label** → ✅ Multi-class: exactly one label (use `softmax`). Multi-label: zero to N labels (use `sigmoid` per class). Wrong setup → wrong loss function → bad training.

❌ **Not shuffling training data** → ✅ If your data is ordered (all positives then all negatives), the model sees a biased gradient. Always shuffle before splitting into train/val/test.

❌ **Using max_length=512 for all inputs without thinking** → ✅ Most social media text fits in 64-128 tokens. Padding everything to 512 wastes compute. Profile your token lengths and set max_length accordingly.

❌ **Trusting model confidence scores** → ✅ A classifier can output 99% confidence and still be wrong. Calibrate your model or add a human review threshold for low-confidence predictions in production.

---

## 8. When to Use / When NOT to Use

### Use text classification when:
- **Fixed, known label set** — spam/not spam, topic buckets, intent categories
- **High volume** — thousands of texts per day where manual labeling is impossible
- **Consistent categories over time** — the categories don't change week to week
- **Speed matters** — BERT classifiers respond in milliseconds

### Don't use text classification when:
- **Open-ended extraction** — use NER or LLM prompting instead
- **Categories change frequently** — fine-tuned models require retraining; use LLM zero-shot
- **Very small datasets** (< 50 examples per class) — zero-shot or few-shot LLM will outperform fine-tuning
- **Nuanced judgment** — ethical decisions, content moderation edge cases need human review

---

## 9. Related Concepts (The Map)

- **NER (Named Entity Recognition)** — token-level classification instead of sequence-level. Both use BERT; they differ in what level the classification head operates at.
- **Embeddings** — classification works by mapping text to a vector, then separating vector clusters by label. Visualization: t-SNE plots show whether positive/negative clusters are separable.
- **Fine-tuning** — the main technique for customizing classification for your domain. Same pipeline as NER and QA fine-tuning.
- **LLM prompting** — the alternative to fine-tuning. Use when: labels change, you have no training data, or you need multi-hop reasoning.
- **Precision/Recall tradeoff** — by adjusting the classification threshold (default 0.5), you can trade precision for recall. Critical for fraud detection (catch more fraud at cost of more false alarms).

---

## 10. Cheat Sheet

| Approach | Accuracy | Speed | Cost | Training Data Needed |
|---|---|---|---|---|
| **TF-IDF + ML** | 80-85% | Very fast | Free | 100s-1000s |
| **Zero-shot pipeline** | 70-85% | Fast | Free | 0 |
| **Fine-tuned DistilBERT** | 90-95% | Fast | GPU time | 100s-1000s |
| **LLM API (few-shot)** | 85-93% | Slow | $ per call | 0-10 |

**Metric formulas:**
```
Precision = TP / (TP + FP)   "of what I flagged, how many were right"
Recall    = TP / (TP + FN)   "of all actual positives, how many did I find"
F1        = 2 * P * R / (P + R)
```

**Core fine-tuning pattern:**
```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=N
)
# lr=2e-5, epochs=3, batch_size=16
```

**Remember this:**
1. Start with `pipeline("zero-shot-classification")` — it works without training
2. Accuracy alone lies on imbalanced data — always check F1
3. Fine-tuned DistilBERT is the production sweet spot (fast + accurate + cheap)

---

## 11. Self-Check Questions

1. Why is accuracy a misleading metric for spam detection where 95% of emails are legitimate?
2. What's the difference between multi-class and multi-label classification?
3. When would you choose LLM prompting over fine-tuning a BERT classifier?
4. A model outputs "NEGATIVE (62% confidence)" on a review. Should you trust this prediction?
5. Why does zero-shot classification work without any task-specific training data?

<details>
<summary>Answers</summary>

1. If 95% of emails are legitimate, a model that labels everything as "not spam" achieves 95% accuracy — but catches zero spam. Precision/recall reveal this: recall for "spam" = 0%. Accuracy is only meaningful when classes are roughly balanced.

2. **Multi-class**: each input gets exactly one label from the set (e.g., news topic = sports OR politics OR tech). Use softmax so probabilities sum to 1. **Multi-label**: an input can get zero to N labels simultaneously (e.g., an article tagged as [technology, business, AI]). Use sigmoid with a threshold per class.

3. Choose LLM prompting when: (1) you have no or very little labeled data, (2) your label categories change frequently, (3) the task requires nuanced judgment that's hard to capture in examples, or (4) you need rapid prototyping. Choose fine-tuning when you have labeled data and need high accuracy on a fixed label set at scale.

4. 62% confidence means the model is quite uncertain — it's barely above the 50% threshold for a binary classifier. This prediction should be flagged for human review or treated as "uncertain." In production, set a minimum confidence threshold (e.g., 80%) and route lower-confidence predictions to a fallback path.

5. Zero-shot classification uses a model trained on Natural Language Inference (NLI) — the task of determining if a hypothesis follows from a premise. It rephrases classification as: "Does this text entail the label 'technology'?" The model's NLI training gives it the ability to reason about label compatibility without ever seeing your specific task.

</details>

---

## 12. Go Deeper

- **[Hugging Face NLP Course Chapter 3](https://huggingface.co/learn/nlp-course/chapter3/1)** — hands-on fine-tuning walkthrough. You'll train a real BERT classifier on GLUE benchmarks in Colab. Best practical start.
- **["A Survey of Text Classification Algorithms" (Kowsari 2019)](https://arxiv.org/abs/1904.08067)** — academic survey covering everything from TF-IDF to Transformers. Skim Section 1-3 for historical context, then jump to Section 5 for deep learning methods.
- **[scikit-learn Text Classification Tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)** — the definitive TF-IDF + classical ML guide. Build a spam filter in 50 lines. Understand the baseline before moving to Transformers.
- **["Zero-Shot Text Classification via Natural Language Inference" (Yin 2019)](https://arxiv.org/abs/1909.00161)** — the paper explaining why zero-shot classification works. 8 pages. Understand the NLI connection.
- **[Weights & Biases Text Classification Tutorial](https://wandb.ai/authors/text-classification/reports/Text-Classification-with-Hugging-Face--VmlldzoxMDYwNTQ)** — fine-tuning with experiment tracking. Learn how to monitor training, compare runs, and pick the best checkpoint.
