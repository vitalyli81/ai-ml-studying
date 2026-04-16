# BERT & Encoder Models

## What Is It?

BERT (Bidirectional Encoder Representations from Transformers) is a **Transformer encoder model** that reads text **in both directions simultaneously** to build a deep understanding of language. It's the foundation of most NLP **understanding** tasks — classification, search, NER, question answering.

```
GPT reads:    The cat sat → on → the → mat      (left to right only)
BERT reads:   The cat sat on the mat             (all at once, both directions)
              ←───────────────────────→
```

## Frontend Analogy

```javascript
// GPT is like streaming a response — you process tokens one by one, left to right
// You can only see what came BEFORE the current token

// BERT is like having the full DOM loaded — you can querySelector() anything
// Every token can "see" every other token, regardless of position

// GPT:  like reading a book page by page, can't peek ahead
// BERT: like reading the whole page at once, seeing how words relate
```

## How BERT Was Trained

BERT learned language through two clever self-supervised tasks (no human labels needed):

### Task 1: Masked Language Model (MLM)

Randomly hide 15% of words and make BERT guess them:

```
Input:  "The [MASK] sat on the [MASK]"
BERT:   "The  cat   sat on the  mat"

It learns: "cat" fits because of "sat on" — animals sit.
           "mat" fits because of "sat on the" — you sit on surfaces.
```

Like a fill-in-the-blank test for the entire internet's text.

### Task 2: Next Sentence Prediction (NSP)

Given two sentences, is the second one the actual next sentence?

```
Input:  "The cat sat on the mat." + "It was a warm afternoon."  → YES (follows)
Input:  "The cat sat on the mat." + "Football is popular."      → NO (random)
```

This teaches BERT about **relationships between sentences**.

### The Training Data

BERT was trained on:
- **BookCorpus** — 800M words from 11,000 books
- **English Wikipedia** — 2,500M words

Result: BERT understands English grammar, facts, relationships, and context — without anyone explicitly teaching it any rules.

## BERT Architecture

```
Input:  [CLS] The cat sat on the mat [SEP]

         ↓     ↓   ↓   ↓   ↓   ↓   ↓    ↓
      ┌─────────────────────────────────────────┐
      │          Transformer Encoder             │
      │          (12 layers, 12 attention heads) │
      │          (each token attends to ALL      │
      │           other tokens simultaneously)   │
      └─────────────────────────────────────────┘
         ↓     ↓   ↓   ↓   ↓   ↓   ↓    ↓

      [CLS]  The  cat  sat  on  the  mat  [SEP]
        ↓
   Use [CLS] vector for sentence-level tasks (classification)
   Use individual token vectors for token-level tasks (NER)
```

**Special tokens:**
- `[CLS]` — "Classification" token. Its output vector represents the whole sentence.
- `[SEP]` — "Separator" token. Marks boundaries between segments.
- `[MASK]` — "Masked" token. Used during training for fill-in-the-blank.

## BERT Sizes

| Model | Layers | Hidden | Heads | Params | Speed |
|-------|--------|--------|-------|--------|-------|
| **BERT-base** | 12 | 768 | 12 | 110M | Moderate |
| **BERT-large** | 24 | 1024 | 16 | 340M | Slow |
| **DistilBERT** | 6 | 768 | 12 | 66M | Fast (97% of BERT's accuracy) |

**Start with DistilBERT** for prototyping. Move to BERT-base if you need more accuracy.

## What BERT Is Good At (and Not)

| Good At (Understanding) | Bad At (Generation) |
|------------------------|-------------------|
| Text classification | Writing text |
| Sentiment analysis | Chatbots |
| Named entity recognition | Summarization |
| Question answering (extractive) | Translation |
| Semantic search / similarity | Creative writing |
| Fill-in-the-blank | Open-ended generation |

**Rule of thumb:** BERT = understanding, GPT = generation.

## Using BERT for Common Tasks

### Text Classification

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")  # uses DistilBERT by default

results = classifier([
    "This movie was absolutely wonderful!",
    "I wasted two hours of my life.",
])
# [{'label': 'POSITIVE', 'score': 0.9998},
#  {'label': 'NEGATIVE', 'score': 0.9995}]
```

### Extractive Question Answering

BERT reads a passage and **highlights the answer span** (doesn't generate new text):

```python
from transformers import pipeline

qa = pipeline("question-answering")

result = qa(
    question="What is the capital of France?",
    context="France is a country in Western Europe. Its capital is Paris, "
            "which is known for the Eiffel Tower and Louvre Museum."
)
print(result)
# {'answer': 'Paris', 'score': 0.98, 'start': 57, 'end': 62}
```

### Semantic Similarity / Search

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')  # BERT-based

# Encode sentences into vectors
sentences = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "What's the weather like?",
]
embeddings = model.encode(sentences)

# Compare similarity
sims = cosine_similarity(embeddings)
print(f"Password vs Credentials: {sims[0][1]:.2f}")  # ~0.82 (similar!)
print(f"Password vs Weather:     {sims[0][2]:.2f}")  # ~0.11 (different)
```

### Fill-in-the-Blank (Masked Language Model)

```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="bert-base-uncased")

results = fill_mask("The capital of France is [MASK].")
for r in results[:3]:
    print(f"  {r['token_str']:10} ({r['score']:.0%})")
# paris      (98%)
# lyon       (1%)
# marseille  (0%)
```

## The BERT Family

| Model | Improvement Over BERT | When to Use |
|-------|----------------------|-------------|
| **DistilBERT** | 60% faster, 97% accuracy | Default choice for speed |
| **RoBERTa** | Better training (no NSP, more data) | When you need best accuracy |
| **ALBERT** | Smaller model, parameter sharing | Memory-constrained environments |
| **DeBERTa** | Improved attention mechanism | State-of-the-art on benchmarks |
| **XLM-RoBERTa** | Multilingual (100 languages) | Non-English or multilingual tasks |

## BERT vs GPT — The Fundamental Difference

```
BERT (Encoder — Bidirectional):
  Sees ALL tokens at once
  "The [MASK] sat on the mat" → fills in "cat"
  Good at: understanding what text MEANS

GPT (Decoder — Left-to-right):
  Only sees tokens that came BEFORE
  "The cat sat on the" → predicts "mat"
  Good at: generating NEW text

BERT: "What does this text mean?"   (classification, search, NER)
GPT:  "What should come next?"       (chatbots, code gen, writing)
```

Think of it as:
- **BERT** = reading comprehension test
- **GPT** = essay writing test

## Key Takeaway

BERT is the **go-to model for understanding text** — classification, NER, search, and question answering. It reads text bidirectionally, so it understands context better than left-to-right models. Start with **DistilBERT** for speed, upgrade to **RoBERTa** or **DeBERTa** for accuracy. Use BERT for understanding tasks and GPT for generation tasks. In production, BERT-family models power most search engines, content moderation systems, and text analytics pipelines.
