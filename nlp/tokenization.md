# Tokenization

## What Is It?

Tokenization is the **first step in every NLP pipeline** — breaking text into smaller pieces (tokens) that a model can process. Tokens can be words, subwords, or even individual characters.

Think of it as splitting a string, but smarter than `.split(' ')`.

## Frontend Analogy

```javascript
// You've done tokenization before:
"Hello, world!".split(' ')           // ["Hello,", "world!"]  — naive word split
"Hello, world!".match(/\w+/g)        // ["Hello", "world"]    — regex tokenizer
JSON.parse('{"key": "value"}')       // Parse structured text into usable parts

// NLP tokenization is similar but handles edge cases:
// "don't"    → ["do", "n't"] or ["don", "'", "t"]?
// "New York" → one token or two?
// "unhappily" → ["un", "happi", "ly"]?   (subword tokenization)
```

## Why Not Just Split on Spaces?

```
Problem 1: Punctuation
  "Hello, world!" → ["Hello,", "world!"]   ← comma stuck to Hello

Problem 2: Compound words
  "New York" → ["New", "York"]             ← loses the meaning of the pair

Problem 3: Unknown words
  "transformerify" → ???                    ← not in any dictionary

Problem 4: Different languages
  "我喜欢编程" → ???                         ← no spaces in Chinese

Problem 5: Vocabulary size
  Every unique word = one entry. English has 170,000+ words.
  Add misspellings, slang, names... → millions of entries → too many.
```

## The Three Approaches

### 1. Word-Level Tokenization (Old School)

Split on spaces/punctuation. Each word is a token.

```
"I love machine learning"
→ ["I", "love", "machine", "learning"]
→ [42, 891, 2547, 3102]   (vocabulary IDs)
```

**Problem:** Unknown words. If "transformerify" isn't in the vocabulary → `[UNK]` (unknown token). You lose all information.

### 2. Character-Level Tokenization

Each character is a token.

```
"cat"
→ ["c", "a", "t"]
→ [67, 65, 84]

Vocabulary size: ~256 (all characters)
```

**Problem:** Sequences become very long. "machine learning" = 16 characters = 16 tokens. The model has to figure out that "m-a-c-h-i-n-e" means "machine" — that's hard and slow.

### 3. Subword Tokenization (The Modern Standard)

Split words into **meaningful subpieces**. Common words stay whole, rare words get broken into known parts.

```
"unhappily"
→ ["un", "happi", "ly"]

"transformerify"
→ ["transform", "er", "ify"]     ← never seen this word, but knows the parts!

"playing"
→ ["play", "ing"]

"cat"
→ ["cat"]                         ← common word stays whole
```

**Best of both worlds:**
- Small vocabulary (30K-50K tokens instead of millions)
- No unknown words (anything can be built from subwords)
- Reasonable sequence length

## BPE — The Algorithm Behind Modern Tokenizers

**Byte Pair Encoding (BPE)** is how GPT, Claude, and Llama build their vocabularies.

### How BPE Learns (Training Phase)

```
Start with all individual characters:
  Vocabulary: [a, b, c, d, e, ..., z, space]

1. Count all pairs of adjacent tokens in the training data
   Most common pair: "t" + "h" → merge into "th"
   Vocabulary: [a, b, ..., z, space, th]

2. Repeat: find next most common pair
   Most common: "th" + "e" → merge into "the"
   Vocabulary: [a, b, ..., z, space, th, the]

3. Keep merging until you reach the desired vocabulary size (e.g., 50,000)
   ...after thousands of merges...
   Vocabulary includes: [a, b, ..., the, ing, tion, "trans", "form", ...]
```

### How BPE Tokenizes (Inference Phase)

```
Input: "unhappiness"

Apply merges in order:
  u-n-h-a-p-p-i-n-e-s-s
  → un-h-a-p-p-i-n-e-s-s     (merge u+n)
  → un-h-a-pp-i-n-e-ss       (merge p+p, s+s)
  → un-happ-i-n-e-ss          (merge common subwords)
  → un-happi-ness             (merge to known subwords)

Final: ["un", "happi", "ness"]
```

## Tokenizers in Practice

### GPT / Claude Tokenization

```
"Hello, how are you?"
→ ["Hello", ",", " how", " are", " you", "?"]
   Token IDs: [15339, 11, 703, 527, 499, 30]

Note: spaces are included IN the tokens (" how" not "how")
```

**Token counts matter** because LLMs charge per token and have context limits:

```
1 token ≈ 4 characters ≈ 0.75 words (in English)

"Machine learning is fascinating" = 4 tokens
"The quick brown fox jumps" = 5 tokens

100K token context window ≈ 75K words ≈ 300 pages of text
```

### Special Tokens

Every tokenizer adds special tokens for structure:

```
[CLS]   → "Start of sequence" (BERT uses this for classification)
[SEP]   → "Separator between segments"
[PAD]   → "Padding" (fill shorter sequences to equal length)
[MASK]  → "Hidden word" (BERT's training trick)
[BOS]   → "Beginning of sequence" (GPT)
[EOS]   → "End of sequence" (GPT)
```

```python
# BERT tokenizer adds special tokens:
"I love NLP"
→ ["[CLS]", "I", "love", "NLP", "[SEP]"]

# GPT tokenizer:
"I love NLP"
→ ["<|endoftext|>", "I", " love", " NLP"]
```

## Hugging Face Tokenizers (What You'll Use)

```python
from transformers import AutoTokenizer

# Load a tokenizer (matches the model you'll use)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize
text = "I love Natural Language Processing!"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['i', 'love', 'natural', 'language', 'processing', '!']

# Full pipeline: text → token IDs (ready for model)
encoded = tokenizer(text, return_tensors="pt")
print(encoded['input_ids'])
# tensor([[ 101, 1045, 2293, 3019, 2653, 6364, 999, 102]])
#          [CLS]  i    love  natural language processing !  [SEP]

# Decode back: token IDs → text
decoded = tokenizer.decode(encoded['input_ids'][0])
print(decoded)
# "[CLS] i love natural language processing ! [SEP]"

# Batch tokenization with padding
texts = ["Short text", "This is a much longer text that needs more tokens"]
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print(batch['input_ids'].shape)   # [2, max_length] — padded to same length
print(batch['attention_mask'])     # 1s for real tokens, 0s for padding
```

## The Attention Mask — Why Padding Needs a Mask

When you batch sentences of different lengths, shorter ones get padded:

```
Sentence 1: "I love NLP"        → [101, 1045, 2293, 17953, 102, 0, 0]
Sentence 2: "Deep learning is great" → [101, 2784, 4083, 2003, 2307, 102, 0]

Attention mask tells the model to IGNORE padding:
Sentence 1: [1, 1, 1, 1, 1, 0, 0]    ← ignore last two
Sentence 2: [1, 1, 1, 1, 1, 1, 0]    ← ignore last one
```

Without the mask, the model would think padding tokens are real words.

## Tokenizer Gotchas

| Gotcha | Why It Matters |
|--------|---------------|
| **Different models = different tokenizers** | BERT and GPT tokenize differently. Always use the tokenizer that matches your model. |
| **Tokens ≠ words** | "unhappiness" = 3 tokens. Token count ≠ word count. |
| **Casing matters** | "bert-base-**uncased**" lowercases everything. "bert-base-**cased**" preserves case. |
| **Max length** | Most models have a limit (512 for BERT, 8K-200K for GPT). Longer text gets truncated. |
| **Multilingual** | Some tokenizers handle multiple languages. Others are English-only. |

## Key Takeaway

Tokenization converts text into numbers that models can process. Modern NLP uses **subword tokenization (BPE)** — common words stay whole, rare words split into known pieces. Always use the tokenizer that comes with your model (`AutoTokenizer.from_pretrained`). Remember: 1 token ≈ 4 characters, and token count determines cost and context limits for LLMs.
