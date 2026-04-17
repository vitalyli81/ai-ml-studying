# NLP & Transformers

## What Is NLP?

NLP (Natural Language Processing) is teaching computers to **understand, generate, and work with human language**. Every time you use autocomplete, translate text, ask ChatGPT a question, or filter spam — that's NLP.

As an AI Engineer, NLP is your **most important domain**. LLMs (GPT, Claude) are NLP models. RAG systems process text. Chatbots understand language. This is where your career lives.

## The Mental Model

> 💡 **Think of it like this:** NLP is **teaching a computer to speak your language by converting everything it sees into numbers.**

Every NLP system, no matter how fancy, boils down to three moves:
1. **Text → numbers** (tokenization + embeddings)
2. **Numbers → numbers** (some neural network does math on them)
3. **Numbers → text or label** (decode back to something a human reads)

Everything in this folder is either a step in that pipeline, a model that does step 2, or a way to adapt a model to your specific task.

## Frontend Analogy

You already do text processing:

```javascript
// Frontend "NLP" you already know:
const search = (query, items) => items.filter(i =>
  i.toLowerCase().includes(query.toLowerCase())  // basic text matching
);

const truncate = (text, max) =>
  text.length > max ? text.slice(0, max) + '...' : text;  // summarization

const validate = (email) =>
  /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);  // pattern recognition

// Real NLP does the same things, but understands MEANING:
// "The food was not bad" → positive sentiment (not just keyword matching)
// "Bank of the river" vs "Bank account" → different meanings of "bank"
```

## The NLP Landscape — What Matters in 2024+

```
Before Transformers (2017):                After Transformers:
  Rule-based systems                        Pretrained models for everything
  Bag of words, TF-IDF                      BERT, GPT, T5
  Word2Vec, GloVe embeddings                Contextual embeddings
  RNNs/LSTMs for sequences                 Attention-based models
  Task-specific models                      One model, many tasks
  Months of feature engineering             Fine-tune in hours
```

**The modern NLP workflow:**
1. Pick a pretrained model from Hugging Face
2. Fine-tune it on your data (or just use it zero-shot)
3. Done

You almost never build from scratch anymore. But understanding the foundations helps you choose the right model and debug problems.

## Core NLP Concepts

### How Computers See Text

Computers don't understand words — they understand numbers. NLP converts text to numbers at every stage:

```
"I love coding"
     ↓ Tokenization
["I", "love", "coding"]
     ↓ Vocabulary lookup
[42, 891, 3847]
     ↓ Embedding
[[0.2, -0.1, 0.8, ...], [0.9, 0.3, -0.2, ...], [0.7, 0.5, 0.1, ...]]
     ↓ Model processing
Understanding / prediction
```

### The NLP Task Zoo

| Task | What It Does | Example |
|------|-------------|---------|
| **Text Classification** | Assign a label to text | Spam detection, sentiment analysis |
| **Named Entity Recognition (NER)** | Find names, places, dates in text | "**Apple** released **iPhone 16** in **September**" |
| **Summarization** | Condense long text | Article → 3-sentence summary |
| **Translation** | Convert between languages | English → French |
| **Question Answering** | Answer questions from context | "What year was Python created?" → "1991" |
| **Text Generation** | Create new text | ChatGPT, Claude, code completion |
| **Semantic Search** | Find by meaning, not keywords | "how to fix slow website" finds "performance optimization guide" |

## The Evolution — From Rules to LLMs

```
Era 1: Rule-Based (1950s-2000s)
  if "not" in sentence: flip_sentiment()
  → Brittle. Thousands of hand-written rules.

Era 2: Statistical ML (2000s-2013)
  Naive Bayes, SVM on TF-IDF features
  → Better, but you still engineer features manually.

Era 3: Word Embeddings (2013-2017)
  Word2Vec, GloVe: words → dense vectors
  "king" - "man" + "woman" ≈ "queen"
  → Words have meaning! But context-independent.

Era 4: Transformers / Pretrained Models (2017-2022)
  BERT, GPT: context-dependent understanding
  "bank" near "river" ≠ "bank" near "money"
  → One model handles many tasks. Fine-tune, don't build.

Era 5: Large Language Models (2022+)
  GPT-4, Claude: emergent abilities, reasoning
  → Prompt it, don't train it. Zero-shot capabilities.
```

**Where you should focus:** Era 4 and 5. Understand Era 3 conceptually, but spend your time on Transformers and LLMs.

## The Three Transformer Paradigms

Why are there three? Because NLP tasks come in three shapes:
- **Input → label** (classify, extract) → needs deep *understanding* → **encoder-only** (BERT)
- **Input → more text** (chat, autocomplete) → needs *generation* → **decoder-only** (GPT)
- **Input → different text** (translate, summarize) → needs *understand THEN generate* → **encoder-decoder** (T5, BART)

The architectures mirror the task shapes. Pick the model family that matches your task shape first; pick the specific model second.

### Encoder Models (BERT family) — Understanding

Read the **entire text at once** and produce a rich representation. Great for tasks that need to understand text.

```
Input:  "The movie was surprisingly good despite bad reviews"
BERT:   [understands full context bidirectionally]
Output: Classification → "Positive sentiment"
```

**Use for:** classification, NER, search, similarity

### Decoder Models (GPT family) — Generation

Read text **left to right** and predict what comes next. Great for generating new text.

```
Input:  "The recipe for chocolate cake starts with"
GPT:    [predicts next token, then next, then next...]
Output: "preheating the oven to 350°F. Mix 2 cups of flour..."
```

**Use for:** chatbots, content generation, code completion, summarization

### Encoder-Decoder Models (T5, BART)

Encode the full input, then decode an output. Best for transforming text.

```
Input:  "Translate to French: The cat sat on the mat"
T5:     [encode full input] → [decode output]
Output: "Le chat s'est assis sur le tapis"
```

**Use for:** translation, summarization, question answering

## Docs in This Folder

Read in this order:

### Foundations

| # | File | Topic | Why It Matters |
|---|------|-------|---------------|
| 1 | [tokenization.md](tokenization.md) | How text becomes numbers | First step in every NLP pipeline |
| 2 | [embeddings.md](embeddings.md) | Word/sentence vectors | How meaning is represented numerically |

### Core Tasks

| # | File | Topic | Why It Matters |
|---|------|-------|---------------|
| 3 | [text-classification.md](text-classification.md) | Sentiment, spam, topics | Most common NLP task |
| 4 | [ner.md](ner.md) | Named Entity Recognition | Extracting structured info from text |
| 5 | [seq2seq-summarization.md](seq2seq-summarization.md) | Summarization & translation | Text-to-text generation |

### Models & Tools

| # | File | Topic | Why It Matters |
|---|------|-------|---------------|
| 6 | [bert.md](bert.md) | BERT & encoder models | The understanding side of NLP |
| 7 | [gpt-decoder-models.md](gpt-decoder-models.md) | GPT & decoder models | The generation side (leads to LLMs) |
| 8 | [huggingface.md](huggingface.md) | Hugging Face ecosystem | The npm of AI — where you get models |
| 9 | [fine-tuning-nlp.md](fine-tuning-nlp.md) | Fine-tuning for NLP tasks | Adapting models to your specific needs |

## What Comes After This?

After NLP & Transformers, you move to **Phase 5: LLMs & AI Engineering** — prompt engineering, RAG, agents, and LLM APIs. That phase builds directly on everything in this folder. NLP is the foundation, LLM engineering is the application.
