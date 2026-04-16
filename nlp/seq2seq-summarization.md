# Sequence-to-Sequence & Summarization

## What Is It?

Seq2Seq (sequence-to-sequence) models take **text in and produce text out**. The input and output can be completely different lengths and even different languages. It's the architecture behind translation, summarization, and any text transformation task.

```
Input sequence  → [Encoder] → [Decoder] → Output sequence

"How are you?"  → [Encoder] → [Decoder] → "Comment allez-vous?"
Long article    → [Encoder] → [Decoder] → 3-sentence summary
```

## Frontend Analogy

```javascript
// Seq2Seq is like a compiler/transpiler:
// Input language → understanding → Output language

// TypeScript → Compiler → JavaScript
// JSX → Babel → Plain JS
// Sass → Preprocessor → CSS
// English → Seq2Seq → French

// Or like a build pipeline:
// Source (full article) → Process (understand meaning) → Output (summary)
```

## How Seq2Seq Works

### The Encoder-Decoder Architecture

```
INPUT: "The cat sat on the mat"

ENCODER (reads the full input):
  "The" → "cat" → "sat" → "on" → "the" → "mat"
                                              ↓
                                   [context vector]
                                   (compressed understanding)
                                              ↓
DECODER (generates output one token at a time):
  [start] → "Le" → "chat" → "s'est" → "assis" → "sur" → "le" → "tapis" → [end]
```

**Encoder** = reads and understands the input (produces a representation)
**Decoder** = generates the output token by token, using the encoder's understanding

### With Attention (The Modern Version)

Instead of compressing everything into one vector, the decoder can **look back at any part of the input** at each step:

```
Generating "chat" (cat in French):
  Decoder looks at encoder outputs →
  High attention on "cat" (0.85)
  Low attention on "the" (0.03)
  Low attention on "sat" (0.05)
  → Uses mostly "cat" information to generate "chat"
```

This is the attention mechanism from the Transformers doc — and it's what makes modern seq2seq work so well.

## Summarization — The Most Practical Seq2Seq Task

### Two Types of Summarization

**Extractive** — picks the most important sentences directly from the text (copy-paste):

```
Original: "The company reported record earnings. Revenue grew 25%.
           The CEO attributed growth to new product launches.
           Office renovations were completed last month."

Extractive: "The company reported record earnings. Revenue grew 25%."
(just picked the best sentences — no new words generated)
```

**Abstractive** — generates a NEW summary in its own words (like a human would):

```
Abstractive: "The company achieved record earnings with 25% revenue growth,
              driven by new product launches."
(new sentence that wasn't in the original — paraphrased and condensed)
```

Modern models (T5, BART, GPT) do **abstractive** summarization. It's harder but produces better summaries.

### Summarization in Practice

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
The Amazon rainforest, often referred to as the "lungs of the Earth,"
produces about 20% of the world's oxygen. Spanning across nine countries
in South America, it covers approximately 5.5 million square kilometers.
The rainforest is home to an estimated 10% of all species on Earth,
including over 40,000 plant species, 1,300 bird species, and 3,000
types of fish. However, deforestation has been accelerating in recent
years, with an area roughly the size of a football field being cleared
every minute. Scientists warn that continued deforestation could push
the Amazon past a tipping point, transforming it from a carbon sink
into a carbon source, which would dramatically accelerate climate change.
"""

summary = summarizer(article, max_length=60, min_length=20)
print(summary[0]['summary_text'])
# "The Amazon rainforest produces about 20% of the world's oxygen
#  and is home to 10% of all species. Deforestation could push it
#  past a tipping point, accelerating climate change."
```

## Translation

```python
from transformers import pipeline

translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

result = translator("Machine learning is transforming every industry.")
print(result[0]['translation_text'])
# "L'apprentissage automatique transforme chaque industrie."
```

## Key Seq2Seq Models

| Model | Creator | Architecture | Best For |
|-------|---------|-------------|----------|
| **T5** | Google | Encoder-Decoder | Everything (treats all tasks as text-to-text) |
| **BART** | Meta | Encoder-Decoder | Summarization, text generation |
| **mBART** | Meta | Encoder-Decoder | Multilingual translation |
| **MarianMT** | Helsinki NLP | Encoder-Decoder | Translation (many language pairs) |
| **Pegasus** | Google | Encoder-Decoder | Summarization (trained specifically for it) |

### T5 — The "Everything is Text-to-Text" Model

T5's big idea: **every NLP task can be framed as text in → text out**:

```
Classification:  "sentiment: This movie is great"      → "positive"
Summarization:   "summarize: [long article]"            → "short summary"
Translation:     "translate English to French: Hello"   → "Bonjour"
QA:              "question: Who is CEO? context: ..."   → "Tim Cook"
```

One model, one format, any task. Just change the prefix.

```python
from transformers import pipeline

t5 = pipeline("text2text-generation", model="t5-base")

# Summarization
t5("summarize: The Amazon rainforest produces 20% of...")

# Translation
t5("translate English to German: Hello, how are you?")

# Same model, different tasks — just change the prefix!
```

## Controlling Output

### Key Parameters

```python
summary = summarizer(
    article,
    max_length=130,      # maximum tokens in output
    min_length=30,       # minimum tokens in output
    do_sample=False,     # False = deterministic (beam search)
)
```

### Decoding Strategies (How the Model Picks Words)

```
"The cat sat on the ___"

Greedy:      Pick the highest probability word every time
             → "mat" (p=0.4) — always the same output, can be repetitive

Beam Search: Track top-K candidates and pick the best overall sequence
             → Considers "mat" (0.4), "floor" (0.3), "rug" (0.2)
             → Picks best SEQUENCE, not just best next word

Sampling:    Randomly sample from the probability distribution
             → Sometimes "mat", sometimes "floor" — more creative/diverse

Top-k:       Sample from only the top K most likely words
             → Filter to top 50 words, then sample from those

Top-p:       Sample from the smallest set of words that sum to probability p
(nucleus)    → If p=0.9, sample from words that cover 90% of probability mass
```

```python
# Deterministic (same output every time) — good for summarization
output = model.generate(input_ids, num_beams=4, do_sample=False)

# Creative (varied output) — good for story generation
output = model.generate(input_ids, do_sample=True, top_p=0.9, temperature=0.8)

# temperature: controls randomness
# < 1.0 = more focused/deterministic
# > 1.0 = more random/creative
```

## Evaluation Metrics

### ROUGE (for Summarization)

ROUGE measures **overlap** between generated summary and reference summary:

```
Reference: "The cat sat on the mat"
Generated: "The cat is on the mat"

ROUGE-1: unigram overlap = 5/6 = 0.83  (matched: the, cat, on, the, mat)
ROUGE-2: bigram overlap  = 2/5 = 0.40  (matched: "the cat", "the mat")
ROUGE-L: longest common subsequence     (longest matching sequence in order)
```

```python
from evaluate import load

rouge = load("rouge")
results = rouge.compute(
    predictions=["The cat is on the mat"],
    references=["The cat sat on the mat"]
)
print(results)
# {'rouge1': 0.83, 'rouge2': 0.4, 'rougeL': 0.83}
```

### BLEU (for Translation)

Similar to ROUGE but designed for translation — measures n-gram precision.

## Common Gotchas

| Problem | Cause | Fix |
|---------|-------|-----|
| Summary just copies the first sentences | Model is lazy / extractive | Use abstractive model (BART, T5) |
| Summary cuts off mid-sentence | `max_length` too short | Increase `max_length` |
| Repetitive output | Greedy decoding | Use beam search or `no_repeat_ngram_size=3` |
| Output in wrong language | Wrong model for language pair | Check model card for supported languages |
| Hallucinated facts | Model generates plausible but wrong info | Known limitation — verify critical facts |

## Key Takeaway

Seq2Seq models transform text into other text — summaries, translations, reformulations. The encoder understands the input, the decoder generates the output. In practice, use **BART for summarization**, **MarianMT/mBART for translation**, and **T5 for any text-to-text task**. The key parameters to control are `max_length`, `num_beams`, and `temperature`. This architecture is the direct ancestor of modern LLMs — GPT is essentially a decoder-only seq2seq model.
