# Tokenization

## 1. TL;DR

Tokenization breaks text into small pieces (tokens) so a model can process it as numbers. Modern models use **subword tokenization** — common words stay whole, rare words split into known pieces. You never need to build a tokenizer; always use `AutoTokenizer.from_pretrained()` matching your model. Token count drives cost and context limits in LLMs (~1 token ≈ 4 characters).

---

## 2. The Mental Model

> 💡 **Think of it like this:** Tokenization is a **postal system for words**.

A postal system doesn't ship entire cities — it breaks destinations into country → state → city → street → number. Similarly, tokenization doesn't ship whole essays — it breaks text into the smallest units a model can handle.

| Real world | Technical concept |
|---|---|
| Postal system splitting an address | Tokenizer splitting text |
| Country/state/city hierarchy | Word → subword → character hierarchy |
| ZIP code (a number for a place) | Token ID (a number for a text piece) |
| Address book on the server | Vocabulary (list of all known tokens) |

---

## Build the Intuition From Zero

The puzzle: **why "subword" tokenization, and how does the tokenizer decide where to split?** Why isn't it just "split on spaces"? Let's build the answer.

### Idea 1: The two obvious approaches both fail

```
Split into WORDS:       "running" "cats" "antidisestablishmentarianism"
   → vocabulary explodes to millions; every typo/rare word is "unknown" → model is blind to it.

Split into CHARACTERS:  "r" "u" "n" "n" "i" "n" "g"
   → tiny vocabulary, but sequences get huge and each piece carries almost no meaning.
```

Words give meaning but can't handle rare/new words. Characters handle anything but lose meaning and bloat length. **Subword** tokenization is the compromise that gets both: keep common words whole, break rare words into reusable pieces.

```
"running"  → ["running"]              (common → one token)
"antidisestablishmentarianism" → ["anti","dis","establish","ment","arian","ism"]
                                  (rare → known pieces, no "unknown" token needed)
```

### Idea 2: How it learns where to split (merge the frequent pairs)

The tokenizer isn't hand-coded — it *learns* its vocabulary from a giant corpus by a dead-simple rule (this is **BPE**, byte-pair encoding):

```
1. Start with pure characters:  l o w   l o w e r   n e w e s t   w i d e s t
2. Count adjacent pairs; merge the MOST FREQUENT one into a new token.
       "e"+"s" appears a lot → merge into "es"
3. Repeat thousands of times, each merge adding one token to the vocabulary:
       "es"+"t" → "est"  ;  "l"+"o" → "lo"  ;  "lo"+"w" → "low"  ...
4. Stop at a target vocab size (e.g. 50,000 tokens).
```

So frequent letter-sequences "graduate" into their own tokens. Common words end up as single tokens because their pieces kept getting merged; rare words stay as a handful of subword pieces. **The split points are wherever the learned merges run out** — that's why "running" is whole but a rare word fractures.

> 💡 **One line:** subword tokenization keeps frequent strings whole and breaks rare ones into known pieces, learned by repeatedly merging the most common adjacent pair — giving a fixed vocabulary that can still encode *any* word, including ones it's never seen. This is also why token count ≠ word count, which drives [LLM cost and context limits](../llms/llm-fundamentals.md).

---

## 3. Why It Exists

**The problem:** Computers only understand numbers. "cat" is meaningless to a neural network.

**What came before:** Early systems used word-level splitting (`"I love cats"` → `["I", "love", "cats"]`). This broke immediately on unknown words — "transformerify" would produce `[UNK]` (unknown), losing all information.

**Character-level** was the fix for unknown words, but created impossibly long sequences — "machine learning" = 16 tokens. Models struggled to learn that `m-a-c-h-i-n-e` means `machine`.

**What changed:** Subword tokenization (BPE, WordPiece) hit the sweet spot — small vocabulary, no unknowns, reasonable length. It powers every modern LLM.

---

## 4. Core Concepts

### Token

**One-line definition:** The atomic unit of text a model processes — could be a word, word-piece, punctuation mark, or whitespace.

**Analogy:** A token is like a LEGO brick. You can build anything from the right set of bricks — you don't need a custom brick for every possible structure.

**Technical explanation:** The model never sees raw text. Every input is first mapped to token IDs (integers) from the vocabulary, then to embeddings (vectors).

```
"unhappily" → ["un", "happi", "ly"] → [4403, 14799, 314] → [vector, vector, vector]
```

**Common misconception:** ❌ "1 token = 1 word" → ✅ 1 word = 1-3 tokens on average. "ChatGPT" = 1 token, "antidisestablishmentarianism" = 5+ tokens.

---

### Vocabulary

**One-line definition:** The fixed list of all tokens a model knows (typically 30,000–100,000 entries).

**Analogy:** Like a dictionary, but instead of definitions, each word gets a number. BERT's dictionary has 30,522 entries; GPT-4's has 100,277.

**Technical explanation:** When you load a tokenizer, you're loading its vocabulary. The model's embedding table has exactly `vocab_size` rows — one vector per token.

**Common misconception:** ❌ "Bigger vocabulary = smarter model" → ✅ Larger vocab reduces tokens per word but costs more memory. There are diminishing returns past ~50K.

---

### Subword Tokenization (BPE)

**One-line definition:** Split words into frequent meaningful pieces; common words stay whole, rare words break apart.

**Analogy:** Like abbreviations in texting — "lol", "brb", "omg" are stored as single units because they're common. A new abbreviation "prblm" gets broken into known parts: "pr" + "b" + "lm".

**Technical explanation:** **Byte Pair Encoding (BPE)** learns merges from training data:

```
Start: individual characters [u, n, h, a, p, p, i, l, y]
Most common pair "p"+"p" → merge to "pp"
Most common pair "i"+"l" → merge to "il"
...after 50,000 merges:
"unhappily" → ["un", "happi", "ly"]
```

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize("unhappily")
print(tokens)  # ['un', 'happi', 'ly']
```

**Common misconception:** ❌ "BPE is a universal algorithm" → ✅ BERT uses WordPiece, GPT uses BPE, T5 uses SentencePiece — all subword methods but trained differently. Always use the tokenizer that comes with your model.

---

### Special Tokens

**One-line definition:** Reserved tokens that signal structure to the model — start, end, padding, separation.

**Analogy:** Like HTML tags — `<html>`, `<body>`, `<p>` don't hold content but tell the browser how to interpret the content around them.

| Token | Meaning | Used by |
|---|---|---|
| `[CLS]` | Start of sequence (used for classification) | BERT |
| `[SEP]` | Separator between segments | BERT |
| `[PAD]` | Padding (fill shorter sequences) | BERT, most |
| `[MASK]` | Hidden word (training trick) | BERT |
| `<\|endoftext\|>` | End of sequence | GPT |
| `<s>` / `</s>` | Start / end of sequence | Llama, T5 |

**Common misconception:** ❌ "Special tokens are optional" → ✅ The model was trained expecting them. Skipping `[CLS]` on BERT breaks the classification head. Always use `tokenizer()` — it adds them automatically.

---

### Attention Mask

**One-line definition:** A binary array (1s and 0s) telling the model which tokens are real vs. padding.

**Analogy:** Like a highlighter — 1 means "pay attention here," 0 means "ignore this, it's blank space."

**Technical explanation:** When batching sentences of different lengths, shorter ones get padded. Without a mask, the model would treat padding as real input and produce wrong results.

```
Sentence 1: "I love NLP"        → tokens: [101, 1045, 2293, 17953, 102,   0,   0]
Sentence 2: "Deep learning rocks"→ tokens: [101, 2784, 4083, 6152,  102,   0,   0]

Attention mask:
Sentence 1: [1,    1,    1,     1,     1,    0,   0]  ← ignore last two
Sentence 2: [1,    1,    1,     1,     1,    0,   0]  ← ignore last two
```

**Common misconception:** ❌ "Padding zeros mean the model ignores them automatically" → ✅ You must explicitly pass `attention_mask` — the model doesn't detect zeros on its own.

---

### Context Window

**One-line definition:** The maximum number of tokens a model can process in one pass.

**Analogy:** Your working memory — you can only hold so many things in mind at once. A model with a 4K context window "forgets" anything beyond 4,000 tokens.

**Common misconception:** ❌ "Context window = number of words" → ✅ Context window is in tokens. A 100K token window ≈ 75K words ≈ 300 pages. Code is more expensive per character than prose.

---

## 5. How It Actually Works (Step-by-Step)

Let's trace `"I love NLP!"` through a BERT tokenizer end-to-end:

```
INPUT TEXT
"I love NLP!"

Step 1: Normalization
  Lowercase (for uncased models): "i love nlp!"

Step 2: Pre-tokenization (split on whitespace/punctuation)
  ["i", "love", "nlp", "!"]

Step 3: Apply learned subword merges (WordPiece)
  "i"    → ["i"]          (common word, stays whole)
  "love" → ["love"]       (common word, stays whole)
  "nlp"  → ["nl", "##p"]  (less common — split, ## = continuation)
  "!"    → ["!"]          (punctuation = own token)

  Result: ["i", "love", "nl", "##p", "!"]

Step 4: Add special tokens
  ["[CLS]", "i", "love", "nl", "##p", "!", "[SEP]"]

Step 5: Map to token IDs (vocabulary lookup)
  [101, 1045, 2293, 17953, 1043, 999, 102]

Step 6: Create attention mask
  [1,    1,    1,    1,     1,    1,   1]   ← all real tokens

OUTPUT (what the model receives)
  input_ids:      [101, 1045, 2293, 17953, 1043, 999, 102]
  attention_mask: [1,   1,    1,    1,     1,    1,   1  ]
```

> 💡 **Key Insight:** The model never sees the letters "I love NLP!" — it only sees the integer sequence `[101, 1045, 2293, 17953, 1043, 999, 102]`. Everything else is the tokenizer's job.

---

## 6. Code in Practice

### Minimal: Tokenize a string

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love Natural Language Processing!"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['i', 'love', 'natural', 'language', 'processing', '!']

encoded = tokenizer(text, return_tensors="pt")
print(encoded['input_ids'])
# tensor([[ 101, 1045, 2293, 3019, 2653, 6364, 999, 102]])
```

### Practical: Batch tokenization with padding

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = [
    "Short text",
    "This is a much longer text that needs more tokens to represent",
]

batch = tokenizer(
    texts,
    padding=True,       # pad shorter sequences to match longest
    truncation=True,    # cut anything beyond max_length
    max_length=512,
    return_tensors="pt"
)

print(batch['input_ids'].shape)   # [2, N] — both padded to same length
print(batch['attention_mask'])    # 1s for real tokens, 0s for padding
```

### Real-world pattern: Count tokens before sending to LLM

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def fits_in_context(text: str, max_tokens: int = 4096) -> bool:
    return count_tokens(text) <= max_tokens

article = "..." # long article
print(f"Tokens: {count_tokens(article)}")
print(f"Fits: {fits_in_context(article)}")
```

---

## 7. Gotchas & Pitfalls

❌ **Using the wrong tokenizer for your model** → ✅ BERT and GPT tokenize differently. Always load the tokenizer with `AutoTokenizer.from_pretrained("your-model-name")`.

❌ **Assuming 1 token = 1 word** → ✅ "unhappiness" = 3 tokens. Always measure token count before submitting to APIs.

❌ **Forgetting truncation** → ✅ BERT has a 512-token limit. Texts longer than that silently get cut unless you set `truncation=True`.

❌ **Ignoring padding when batching** → ✅ Sequences in a batch must be the same length. Always use `padding=True` with batches.

❌ **Skipping the attention mask** → ✅ If you pad manually without passing `attention_mask`, the model attends to padding tokens and produces garbage.

❌ **Cased vs. uncased confusion** → ✅ `bert-base-uncased` lowercases everything. Using it for cased tasks (NER with proper nouns) hurts accuracy. Use `bert-base-cased` for tasks where capitalization matters.

❌ **Decoding without skipping special tokens** → ✅ `tokenizer.decode(ids)` includes `[CLS]`, `[SEP]`, `[PAD]`. Use `tokenizer.decode(ids, skip_special_tokens=True)` for clean output.

---

## 8. When to Use / When NOT to Use Each Tokenizer

### Use subword tokenization (BPE/WordPiece) when:
- Working with any Transformer model (BERT, GPT, Llama, T5)
- Handling multilingual text
- Dealing with technical jargon, code, or rare words
- You need a fixed vocabulary size

### Don't use character-level tokenization when:
- Sequence lengths matter (character-level makes sequences ~4x longer)
- You're working with languages with short words (most languages)

### Don't use word-level tokenization when:
- Your vocabulary has rare/technical words
- You're dealing with multiple languages
- You want no unknown tokens

---

## 9. Related Concepts (The Map)

- **Embeddings** — after tokenization, each token ID is looked up in an embedding table to become a vector. Tokenization → Embedding is the first two steps of every NLP pipeline.
- **Attention Mechanism** — operates on the token sequences produced by the tokenizer. Context window limits come from attention's O(n²) cost.
- **BPE vs. WordPiece vs. SentencePiece** — three subword algorithms. BPE = GPT/Llama, WordPiece = BERT, SentencePiece = T5/Llama3. Same idea, different training.
- **Prompt Engineering** — writing good prompts means being aware of token count. Few-shot examples cost tokens. Think in tokens, not words.
- **RAG chunking** — when splitting documents for retrieval, you chunk by token count (not character count) to respect context limits.

---

## 10. Cheat Sheet

| Term | Definition |
|---|---|
| **Token** | Smallest unit of text a model processes |
| **Vocabulary** | Fixed list of all known tokens + their IDs |
| **BPE** | Byte Pair Encoding — merge most frequent pairs |
| **WordPiece** | BERT's variant of BPE |
| **[CLS]** | Start token used for classification output |
| **[SEP]** | Separator between two text segments |
| **[PAD]** | Padding token for equal-length batches |
| **Attention mask** | 1=real token, 0=padding |
| **Context window** | Max tokens the model can process at once |
| **`##`** | WordPiece prefix meaning "continuation of previous word" |

**Core pattern:**
```python
tokenizer = AutoTokenizer.from_pretrained("model-name")
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
# inputs has: input_ids, attention_mask, (token_type_ids for BERT)
```

**Remember this:**
1. Always use the tokenizer that comes with your model
2. 1 token ≈ 4 characters ≈ 0.75 words (English)
3. Token count = cost + context limit — measure before sending

---

## 11. Self-Check Questions

1. What problem does subword tokenization solve that word-level tokenization can't?
2. Why does BERT need an attention mask when you batch sequences?
3. What's the difference between `tokenizer.tokenize()` and `tokenizer()`?
4. A user sends a 10,000-word document to a BERT model with a 512-token limit. What happens?
5. Why does `"bert-base-uncased"` lower-case your text, and when does this matter?

<details>
<summary>Answers</summary>

1. **Unknown words.** Word-level tokenization maps any unseen word to `[UNK]`, losing all information. Subword tokenization can decompose any word into known pieces — "transformerify" → ["transform", "er", "ify"] — zero unknowns.

2. **Padding.** When batching, shorter sequences get padded to match the longest one. The attention mask tells the model which positions are real (1) vs. filler (0), so it doesn't attend to padding.

3. `tokenizer.tokenize(text)` returns a list of string tokens (human-readable). `tokenizer(text)` returns a dict with `input_ids` (integers), `attention_mask`, and optionally `token_type_ids` — ready to feed into a model.

4. With `truncation=True`, it silently cuts to 512 tokens. Without it, it throws an error. The end of the document is lost — for long docs, use chunking strategies or a model with longer context.

5. Uncased models lowercase all input during tokenization so "Apple" and "apple" map to the same token. This helps for sentiment/classification where case doesn't matter. For NER, where "Apple" (company) ≠ "apple" (fruit), use `bert-base-cased`.

</details>

---

## 12. Go Deeper

- **[Hugging Face Tokenizers Course](https://huggingface.co/learn/nlp-course/chapter2/4)** — hands-on notebook showing exactly what happens inside `AutoTokenizer`. Best 30-minute deep dive.
- **["Neural Machine Translation of Rare Words with Subword Units" (Sennrich 2016)](https://arxiv.org/abs/1508.07909)** — the original BPE paper for NLP. Short and readable. Understand *why* BPE was chosen.
- **[Tiktokenizer (OpenAI)](https://tiktokenizer.vercel.app/)** — interactive visual tokenizer. Paste any text and see exactly how GPT-4 tokenizes it. Builds intuition fast.
- **[minBPE by Andrej Karpathy](https://github.com/karpathy/minbpe)** — build a BPE tokenizer from scratch in ~200 lines. If you want to truly understand it, implement it once.
- **[Hugging Face Tokenizers library docs](https://huggingface.co/docs/tokenizers)** — the fast Rust-backed tokenizer library. Essential reading before building production NLP pipelines.
