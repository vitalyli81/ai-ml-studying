# BERT & Encoder Models

## 1. TL;DR

BERT reads your entire sentence **in both directions at once** to deeply understand what each word means in context. It doesn't generate text — it understands it. Use BERT for classification, search, NER, and question answering. Start with `DistilBERT` (60% faster, almost same accuracy), upgrade to `RoBERTa` when you need peak performance. The mental model: BERT = reading comprehension, GPT = essay writing.

---

## 2. The Mental Model

> 💡 **Think of it like this:** BERT reads a sentence like a **detective examining a crime scene** — not left-to-right, but all at once, looking at every clue in relation to every other clue.

GPT reads a mystery novel page by page without being able to flip back. BERT reads the whole page at once and can see how every sentence relates to every other.

| Real world | Technical concept |
|---|---|
| Detective reading all clues simultaneously | BERT processing all tokens bidirectionally |
| Every clue has meaning relative to others | Every token's embedding depends on all others |
| Scene understanding (not narrative) | Classification, NER, search (not generation) |
| Final verdict (one answer) | [CLS] token → classification output |
| Annotating specific items (highlight the knife) | Token vectors → NER output |

---

## Build the Intuition From Zero

The core idea: **what "bidirectional" really buys you, and how do you even train a model to read both directions at once?** That training trick (masking) is BERT's whole secret.

### Idea 1: Why reading both directions matters

GPT reads left-to-right, so when it's at a word it has only seen what came *before*. BERT sees the *whole* sentence, so each word is understood using context on **both sides**:

```
"I deposited cash at the bank."     left-to-right at "bank": knows "deposited cash" → leans money ✓
"I sat on the bank of the river."   left context "I sat on the" alone is ambiguous;
                                    but the words AFTER ("of the river") nail it → riverbank
   → only a model that sees the RIGHT side too gets this reliably. That's bidirectionality.
```

For *understanding* tasks (classification, search, NER) you have the full text available, so why handicap yourself to one direction? BERT doesn't.

### Idea 2: The training trick — fill in the blanks (masked language modeling)

But there's a chicken-and-egg problem: if you train a model to predict the next word *and* let it see both directions, it can just peek at the answer. BERT's fix is brilliant and simple — **hide some words and make the model guess them from both sides**:

```
Input:  "The cat [MASK] on the warm [MASK]."
Task:   guess the masked words using ALL surrounding context
Model:  [MASK]₁ → "sat" (0.7), "slept" (0.2)   [MASK]₂ → "mat" (0.5), "rug" (0.3)
```

Predicting a blank from *both* directions can't be cheated, and to do it well the model must genuinely understand grammar, meaning, and world knowledge. Do this on billions of sentences (~15% of words masked) and BERT learns deep bidirectional understanding. It's literal fill-in-the-blank practice at internet scale.

### Idea 3: The [CLS] token — a summary slot

BERT prepends a special `[CLS]` token to every input. Because attention lets it absorb information from every other word, its final vector becomes a **summary of the whole sentence** — the handle you attach a classifier to:

```
[CLS] this movie was fantastic  →  [CLS] vector ──► tiny classifier ──► "positive" ✓
  └── soaks up the whole sentence's meaning via attention ──┘
```

> 💡 **One line:** BERT reads both directions (better for *understanding*), is trained by guessing masked-out words so it can't cheat, and exposes a `[CLS]` summary slot you bolt task heads onto. Contrast with [gpt-decoder-models.md](gpt-decoder-models.md): BERT = reading comprehension (encoder), GPT = writing (decoder). Both are Transformers ([../deep-learning/transformers.md](../deep-learning/transformers.md)); they just differ in direction and training objective.

---

## 3. Why It Exists

**The problem:** Before BERT (2018), models processed text left-to-right, which meant they couldn't see the full context when processing a word. "I made her duck" — does "duck" mean the bird or the action? Left-to-right models only see "I made her" when they hit "duck."

**What came before:** LSTMs and early Transformer decoders processed sequences one direction at a time. ELMo (2018) tried concatenating left-to-right and right-to-left LSTMs, but they were separate passes, not true bidirectional attention.

**What changed:** BERT introduced **masked language modeling** — train by hiding random words and predicting them from context. To predict a masked word, the model must understand the full surrounding context. This forced true bidirectional understanding. BERT dominated every NLP benchmark when it launched.

---

## 4. Core Concepts

### Bidirectional Attention

**One-line definition:** Every token in the input can attend to every other token — past and future positions simultaneously.

**Analogy:** Reading every word with Post-it notes connecting it to every other word. "The bank near the river was flooded" — "bank" has connections to both "river" and "flooded," which together resolve its meaning as riverbank, not financial institution.

**Technical explanation:**

```
Input:  [CLS] The bank near the river was flooded [SEP]

When computing "bank":
  BERT attends to: The(0.1) bank(1.0) near(0.3) the(0.1) river(0.8) was(0.2) flooded(0.5)
  High attention on "river" and "flooded" → understands this is a geographical bank

GPT (left-to-right) when processing "bank":
  Can only see: "The", "bank" — "river" and "flooded" haven't appeared yet!
```

**Common misconception:** ❌ "Bidirectional means BERT processes text twice (forward + backward)" → ✅ BERT processes the entire sequence simultaneously with attention. Every position attends to every other position in a single pass.

---

### Masked Language Modeling (MLM)

**One-line definition:** BERT's training task — randomly hide 15% of tokens and train the model to predict them from context.

**Analogy:** A fill-in-the-blank test where the blanks are scattered randomly throughout the text. To answer, you must understand every other word. "The ___ sat on the mat" — you need both sides to guess "cat."

```
Training example:
  Original: "The cat sat on the mat"
  Masked:   "The [MASK] sat on the [MASK]"
  Task:     predict "cat" and "mat"

The model MUST read both directions to solve this:
  "[MASK] sat on the mat" → uses "sat on", "mat" → "cat" (animals sit on mats)
  "The cat sat on the [MASK]" → uses "sat on" → "mat" (you sit on surfaces)
```

**Common misconception:** ❌ "MLM means BERT can fill in blanks at inference time" → ✅ MLM is the training objective, not the primary use. You wouldn't use BERT to fill blanks in production — that's a side effect. The real benefit is that MLM forces BERT to learn deep contextual representations.

**How MLM → classification works (the step most docs skip):** During pretraining, BERT's output layer is `hidden_dim (768) → vocab_size (30,522)` — it predicts which word fills each masked position. When you fine-tune for classification, you **throw away that output layer** and bolt on a new one: `hidden_dim (768) → num_classes (e.g., 2)`. The body of BERT (the 12 encoder layers) stays the same — you're reusing the language understanding it built during MLM. The classification "head" is a new, randomly initialized layer that learns your task on top.

---

### [CLS] Token

**One-line definition:** A special token added at the start of every BERT input whose final vector summarizes the whole sequence.

**Analogy:** The executive summary at the top of a report. The actual content is in the body, but the first sentence captures the gist. The [CLS] output vector is the "gist" of the sentence.

```
Input:  [CLS] This movie was amazing [SEP]
Output:  ↓      ↓     ↓    ↓    ↓      ↓

        [CLS]  This  movie  was  amazing  [SEP]
          ↓
        768-dimensional vector
          ↓
        Linear layer → "POSITIVE" (0.99)
```

**Common misconception:** ❌ "Any token's output vector can be used for classification" → ✅ [CLS] is specifically trained to aggregate sequence-level meaning. Individual token vectors are better for token-level tasks (NER). Using a random token's vector for classification gives worse results.

---

### Next Sentence Prediction (NSP)

**One-line definition:** A secondary training task where BERT predicts whether two segments actually follow each other in the original text.

**Analogy:** A quiz game — "Do these two sentences come from the same paragraph, or were they picked randomly?"

```
True pair:   "The cat sat on the mat." + "It was a warm afternoon."  → IsNext
Random pair: "The cat sat on the mat." + "Football is popular."      → NotNext
```

This teaches BERT **cross-sentence relationships** — useful for question answering and multi-sentence classification. Note: later research (RoBERTa) found NSP wasn't critical and removing it improved performance.

---

### The 12-Layer Architecture

**One-line definition:** BERT processes text through 12 stacked Transformer encoder layers, each layer refining the representation.

**Analogy:** Like an art critique process — the first layer notices basic shapes, the next notices composition, then color relationships, then emotional tone, building up from simple to complex understanding.

```
Layer 1-4:   Surface patterns (syntax, part-of-speech)
Layer 5-8:   Semantics (word relationships, coreference)
Layer 9-12:  Task-specific understanding (sentiment, entities)
```

```
Input tokens
    ↓
[Embedding Layer]       ← token + position embeddings
    ↓
[Transformer Encoder 1] ← 12 attention heads, 768 hidden units
    ↓
[Transformer Encoder 2]
    ↓
    ...12 layers total...
    ↓
[Transformer Encoder 12]
    ↓
Output vectors (one per input token)
```

---

## 5. How It Actually Works (Step-by-Step)

Let's classify the sentiment of "The movie was surprisingly good despite bad reviews":

```
Step 1: Tokenize + add special tokens
  ["[CLS]", "the", "movie", "was", "surprisingly", "good", "despite", "bad", "reviews", "[SEP]"]

Step 2: Convert to embeddings (3 parts summed)
  Token embedding:    learned vector for each token
  Position embedding: learned vector for position 0, 1, 2, ...
  Segment embedding:  0 or 1 (for two-sentence tasks)
  Sum all three → input to first Transformer layer

Step 3: Pass through 12 Transformer encoder layers
  Each layer:
    - Multi-head self-attention (each token attends to all others)
    - Feed-forward network
    - Layer normalization
  After 12 layers: each token has a rich 768-dimensional vector

Step 4: Extract [CLS] vector for classification
  The [CLS] token's output: [0.2, -0.5, 0.8, ..., 0.1]  (768 numbers)

Step 5: Classification head (one linear layer)
  768 → 2 (positive/negative)
  Apply softmax → [0.02, 0.98]

Step 6: Output
  "POSITIVE" with 98% confidence
```

> 💡 **Key Insight:** BERT itself doesn't know about "positive" or "negative" — it just produces vectors. The classification head (trained on labeled data) maps those vectors to labels. This is what fine-tuning does.

---

## 6. Code in Practice

### Minimal: Sentiment with pipeline

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")  # uses DistilBERT by default

results = classifier([
    "This movie was absolutely wonderful!",
    "Worst experience ever. Never again.",
    "It's fine, I guess.",
])

for r in results:
    print(f"{r['label']}: {r['score']:.2%}")
# POSITIVE: 99.98%
# NEGATIVE: 99.95%
# NEGATIVE: 57.12%  ← uncertain, near 50%
```

### Practical: Extractive question answering

```python
from transformers import pipeline

qa = pipeline("question-answering")

result = qa(
    question="Who founded Apple?",
    context=(
        "Apple Inc. was founded on April 1, 1976, by Steve Jobs, "
        "Steve Wozniak, and Ronald Wayne. The company is headquartered "
        "in Cupertino, California."
    )
)

print(result)
# {'answer': 'Steve Jobs, Steve Wozniak, and Ronald Wayne',
#  'score': 0.95, 'start': 52, 'end': 94}
```

### Real-world pattern: Fine-tune BERT for custom classification

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
        learning_rate=2e-5,          # small LR — preserve pretrained knowledge
        eval_strategy="epoch",       # renamed from evaluation_strategy in transformers 4.41+
        save_strategy="epoch",       # must match eval_strategy for load_best_model_at_end
        load_best_model_at_end=True,
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)
trainer.train()
```

---

## 7. Gotchas & Pitfalls

❌ **Using BERT for text generation** → ✅ BERT is an encoder — it has no decoder. It can't generate new text. Use GPT-family models for generation.

❌ **Forgetting the 512-token limit** → ✅ BERT-base has a hard 512-token limit. Long documents must be chunked. Common strategy: use the first 512 tokens, or split and aggregate predictions.

❌ **Using bert-base-uncased for NER** → ✅ Uncased models lowercase everything — "Apple" (company) and "apple" (fruit) look identical. For NER and casing-sensitive tasks, use `bert-base-cased`.

❌ **Setting a high learning rate during fine-tuning** → ✅ Use 1e-5 to 5e-5. Higher rates cause "catastrophic forgetting" — the model overwrites its pretrained knowledge. BERT is fragile to large gradient updates.

❌ **Ignoring DistilBERT** → ✅ DistilBERT is 60% faster and 40% smaller while retaining 97% of BERT's performance. Always start with DistilBERT and only upgrade if you hit an accuracy wall.

❌ **Using [CLS] for sentence similarity without fine-tuning** → ✅ Vanilla BERT's [CLS] vector is poor for sentence similarity. Use `sentence-transformers` (BERT fine-tuned specifically for semantic similarity) instead.

❌ **Not using grouped_entities in NER pipeline** → ✅ By default, `pipeline("ner")` returns individual subword tokens. Use `grouped_entities=True` to merge "New" + "York" into "New York."

---

## 8. When to Use / When NOT to Use

### Use BERT when:
- **Text classification** — sentiment, topic, intent detection
- **Named Entity Recognition** — extract people, places, dates
- **Semantic search** — find documents by meaning (use sentence-transformers)
- **Extractive QA** — highlight the answer span in a passage
- **Text similarity** — find duplicate/near-duplicate content

### Don't use BERT when:
- **Generating text** — use GPT/Llama instead
- **Long documents** (>512 tokens) — use Longformer, BigBird, or chunk+aggregate
- **Chatbots or assistants** — BERT can't hold a conversation
- **Translation or summarization** — use encoder-decoder models (T5, BART)
- **Zero-shot classification of many classes** — LLM prompting is more flexible

---

## 9. Related Concepts (The Map)

- **GPT (Decoder models)** — the generation counterpart. If BERT = reading comprehension, GPT = writing. They're the two poles of the Transformer world.
- **Transformer architecture** — BERT uses only the encoder half of the original "Attention is All You Need" Transformer. GPT uses only the decoder half.
- **Embeddings** — BERT's final layer output vectors ARE contextual embeddings. Sentence-Transformers is just BERT fine-tuned to produce better sentence-level embeddings.
- **Fine-tuning** — BERT is almost never used raw. You always fine-tune a classification/NER/QA head on top of it for your specific task.
- **RoBERTa, DeBERTa, ALBERT** — BERT variants that improved specific weaknesses. Think of them as BERT 1.1, 1.2, 1.3.

---

## 10. Cheat Sheet

| Model | Params | Speed | Accuracy | Best for |
|---|---|---|---|---|
| **DistilBERT** | 66M | Fast | 97% of BERT | Default starting point |
| **BERT-base** | 110M | Moderate | Baseline | General tasks |
| **BERT-large** | 340M | Slow | +1-2% | When accuracy matters most |
| **RoBERTa** | 125M | Moderate | Best | High-accuracy production |
| **DeBERTa** | 183M | Moderate | SOTA on benchmarks | Competitive benchmarks |
| **XLM-RoBERTa** | 270M | Moderate | Strong | Multilingual tasks |

**Core fine-tuning pattern:**
```python
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=N)
# Train with lr=2e-5, epochs=3, batch_size=16
```

**Task → Model head mapping:**
```
Classification → AutoModelForSequenceClassification
NER            → AutoModelForTokenClassification
QA             → AutoModelForQuestionAnswering
Similarity     → SentenceTransformer('all-MiniLM-L6-v2')
```

**Remember this:**
1. BERT = understanding, not generation
2. Always start with DistilBERT
3. [CLS] → sentence tasks; token vectors → token tasks (NER)

---

## 11. Self-Check Questions

1. Why can't BERT generate text like GPT can?
2. What is masked language modeling, and why does it force bidirectional understanding?
3. When should you use `bert-base-cased` instead of `bert-base-uncased`?
4. A colleague says "I'll use BERT to build a chatbot." What would you tell them?
5. What's the difference between using BERT's [CLS] output for similarity vs. using `sentence-transformers`?

<details>
<summary>Answers</summary>

1. BERT is an encoder — it produces representations for an existing sequence but has no decoder to generate new tokens autoregressively. Generating text requires a model that can predict the next token, which BERT was never trained to do. Use GPT/Llama for generation.

2. During training, 15% of tokens are replaced with [MASK]. To predict them, the model must use both the words before AND after the masked position. A left-to-right model can't see future tokens, so it can't be trained this way. MLM forces the model to build a complete bidirectional understanding of context.

3. Use `bert-base-cased` when capitalization carries semantic meaning for your task. Named Entity Recognition is the classic case — "Apple" (company) vs. "apple" (fruit) is only distinguishable by the capital letter. Uncased models lowercase everything, losing this signal.

4. BERT cannot hold a conversation — it classifies or extracts from existing text, it doesn't generate replies. You'd want a GPT-family model for a chatbot. BERT could be used as a component inside a chatbot (e.g., intent classification), but it cannot be the chatbot itself.

5. Vanilla BERT's [CLS] token was trained with Next Sentence Prediction, which is a different task than semantic similarity. The resulting vectors are not well-suited for comparing sentence meaning. `sentence-transformers` fine-tunes BERT specifically on semantic similarity tasks (using contrastive learning), so its output vectors have a much better geometry for cosine similarity comparison.

</details>

---

## 12. Go Deeper

- **["BERT: Pre-training of Deep Bidirectional Transformers" (Devlin 2018)](https://arxiv.org/abs/1810.04805)** — the original paper. Clear, well-written, 13 pages. Read the abstract, introduction, and Section 3 to get the full picture.
- **[Jay Alammar's "The Illustrated BERT"](https://jalammar.github.io/illustrated-bert/)** — the best visual explanation of BERT that exists. Animated diagrams showing attention, MLM, and fine-tuning. Essential.
- **[Hugging Face NLP Course, Chapter 3](https://huggingface.co/learn/nlp-course/chapter3/1)** — hands-on fine-tuning walkthrough with BERT. You'll train a real classifier in Colab.
- **["RoBERTa: A Robustly Optimized BERT" (Liu 2019)](https://arxiv.org/abs/1907.11692)** — shows that BERT was undertrained. Key lessons: remove NSP, train longer, use larger batches. 8 pages, worth reading after the original.
- **[BertViz](https://github.com/jessevig/bertviz)** — interactive visualization of BERT's attention heads in Jupyter. See exactly what patterns each of the 12×12 attention heads has learned.
