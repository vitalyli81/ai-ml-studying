# Transformers

## What Is It?

The Transformer is the architecture behind **everything modern in AI** — GPT, Claude, BERT, Stable Diffusion, and more. It processes entire sequences **in parallel** (not step-by-step like RNNs) using a mechanism called **attention** that lets every word look at every other word directly.

The key paper: "Attention Is All You Need" (2017). It changed everything.

## Frontend Analogy — Event Delegation vs Event Bubbling

```javascript
// RNN approach: Event bubbling — info passes through every element in order
// Parent → Child → Grandchild → Great-grandchild
// Problem: by the time info reaches the bottom, early context is lost

// Transformer approach: Event delegation — every element can listen directly
// Any element can directly access any other element
// No information loss, no matter the distance
document.addEventListener('click', (e) => {
  // Direct access to any element — that's attention!
});
```

## Why Transformers Replaced RNNs

| RNN Problem | Transformer Solution |
|-------------|---------------------|
| Sequential processing (slow) | Parallel processing (fast on GPUs) |
| Forgets distant words | Direct access to ALL words via attention |
| Hard to train long sequences | Handles thousands of tokens |
| Vanishing gradients | No gradient chain through time steps |

## The Core Idea: Self-Attention

### What Is Attention?

When you read "The **cat** sat on the **mat** because **it** was tired" — what does "it" refer to? Your brain **attends** to "cat" more than "mat" or "sat."

Self-attention does the same thing computationally: for each word, it calculates **how much attention to pay to every other word**.

### Attention Step by Step

For the sentence: "The cat sat"

```
Step 1: Each word creates three vectors:
  "The" → Query₁, Key₁, Value₁
  "cat" → Query₂, Key₂, Value₂
  "sat" → Query₃, Key₃, Value₃

Step 2: Each word's Query asks "what should I pay attention to?"
        Each word's Key says "here's what I contain"
        Score = Query · Key (dot product — how relevant?)

Step 3: Attention scores for "sat":
  "sat" query × "The" key = 0.1  (low — not very relevant)
  "sat" query × "cat" key = 0.7  (high — who sat? the cat!)
  "sat" query × "sat" key = 0.2  (medium — self-reference)

Step 4: Use scores to weight the Values:
  Output for "sat" = 0.1 × Value("The") + 0.7 × Value("cat") + 0.2 × Value("sat")
  → "sat" now contains mostly "cat" information (it knows WHO sat)
```

### The QKV Analogy — Database Query

Think of it like a database lookup:

```javascript
// Query: "What information do I need?"
// Key:   "What information does each record have?"
// Value: "The actual information in each record"

// Like a fuzzy SQL query:
// SELECT Value FROM words WHERE Key MATCHES Query
// But instead of exact match, it returns a weighted combination
```

Or like a **search engine**:
- **Query** = your search terms
- **Keys** = page titles / metadata
- **Values** = page content
- **Attention score** = relevance ranking
- **Output** = blended content from top results

## Multi-Head Attention

One attention head might learn grammar. Another learns meaning. Another learns position. **Multi-head = multiple perspectives simultaneously:**

```
Head 1: "sat" attends to "cat"     (who did the action?)
Head 2: "sat" attends to "on"      (what preposition follows?)
Head 3: "sat" attends to "mat"     (where?)
Head 4: "sat" attends to "The"     (sentence structure)

Combine all heads → rich understanding of "sat" in context
```

```python
# 8 attention heads, each looking at a different aspect
attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
```

## The Full Transformer Architecture

### The Original: Encoder-Decoder

```
┌─────────────────────────────────────────────┐
│                  ENCODER                     │
│                                              │
│  Input: "The cat sat"                        │
│           ↓                                  │
│  [Positional Encoding]  ← where each word is │
│           ↓                                  │
│  ┌─────────────────────┐                     │
│  │ Multi-Head Attention │ ← words look at    │
│  │                     │    each other       │
│  ├─────────────────────┤                     │
│  │ Feed-Forward Network │ ← process each     │
│  │                     │    word individually│
│  └─────────────────────┘                     │
│  × N layers (6-96 layers)                    │
│           ↓                                  │
│  Rich representation of input                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│                  DECODER                     │
│                                              │
│  Output so far: "Le chat"                    │
│           ↓                                  │
│  [Positional Encoding]                       │
│           ↓                                  │
│  ┌─────────────────────┐                     │
│  │ Masked Self-Attention│ ← can only look    │
│  │                     │    at previous words│
│  ├─────────────────────┤                     │
│  │ Cross-Attention      │ ← look at encoder  │
│  │                     │    output (source)  │
│  ├─────────────────────┤                     │
│  │ Feed-Forward Network │                     │
│  └─────────────────────┘                     │
│  × N layers                                 │
│           ↓                                  │
│  Next word prediction: "s'est"               │
└─────────────────────────────────────────────┘
```

### Three Variants in Practice

| Architecture | Models | Use Case |
|-------------|--------|----------|
| **Encoder only** | BERT, RoBERTa | Understanding text (classification, NER) |
| **Decoder only** | GPT, Claude, Llama | Generating text (chatbots, code gen) |
| **Encoder-Decoder** | T5, BART, original Transformer | Translation, summarization |

**GPT/Claude are decoder-only** — they predict the next token, one at a time, using only what came before (masked attention).

## Positional Encoding — How Transformers Know Word Order

Unlike RNNs (which process in order), Transformers see all words simultaneously. So they need a way to know **position**:

```
"Dog bites man" ≠ "Man bites dog"

Without position info, the Transformer sees the same bag of words!
```

**Solution:** Add a unique position signal to each word:

```
"Dog"  + position_0 = "Dog at position 0"
"bites" + position_1 = "bites at position 1"
"man"  + position_2 = "man at position 2"
```

The original paper uses sine/cosine functions. Modern models learn the positions during training.

## Why Transformers Scale So Well

```
RNN:  Process token 1, then token 2, then token 3...  (sequential)
      Time: O(n) — linear, can't parallelize

Transformer: Process ALL tokens simultaneously        (parallel)
             Time: O(1) per layer — massively parallel on GPUs
             Memory: O(n²) — every token attends to every token
```

This is why Transformers **love GPUs** — matrix multiplication across all tokens runs in parallel. But the O(n²) memory cost is why there's a **context window limit** (can't process infinite text).

## The Scaling Laws

The surprise discovery: **make it bigger = make it smarter**

```
More parameters  → better performance
More data        → better performance
More compute     → better performance

GPT-2:   1.5 billion parameters
GPT-3:   175 billion parameters
GPT-4:   ~1.8 trillion parameters (estimated)
Claude:  undisclosed, but very large
```

This is why companies are racing to build bigger Transformers.

## Python Example — Using a Transformer Layer

```python
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256, nhead=8, num_layers=4):
        super().__init__()

        # Convert word indices to vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (learned)
        self.pos_encoding = nn.Embedding(512, d_model)  # max 512 positions

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,           # 8 attention heads
            dim_feedforward=1024,  # feed-forward hidden size
            dropout=0.1,
            activation='gelu',     # modern activation
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: classify the sequence
        self.classifier = nn.Linear(d_model, 3)  # 3 classes

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Embed tokens + add position info
        x = self.embedding(x) + self.pos_encoding(positions)

        # Pass through transformer
        x = self.transformer(x)

        # Use the [first token] representation for classification
        # (like BERT's [CLS] token)
        x = x[:, 0, :]

        return self.classifier(x)

# Test
model = SimpleTransformer()
input_ids = torch.randint(0, 10000, (4, 50))  # batch=4, seq_len=50
output = model(input_ids)
print(f"Output shape: {output.shape}")  # [4, 3] — 3 class scores
```

## The Evolution

```
2017: Transformer paper         → "Attention Is All You Need"
2018: BERT (encoder only)       → Revolutionized NLP understanding
2018: GPT-1 (decoder only)      → 117M params, showed generation promise
2019: GPT-2                     → 1.5B params, "too dangerous to release"
2020: GPT-3                     → 175B params, few-shot learning
2022: ChatGPT                   → Transformers go mainstream
2023: GPT-4, Claude             → Multimodal, reasoning capabilities
2024+: Bigger, smarter, agents  → Tool use, long context, real-world actions
```

## Key Takeaway

Transformers process sequences **in parallel** using **self-attention** (every token can directly look at every other token). This solves RNN's forgetting problem and scales beautifully on GPUs. The QKV mechanism (Query, Key, Value) is the core — think of it as a fuzzy database lookup where each word queries all other words for relevant information. Understanding Transformers is **essential** because they power every major AI system today: GPT, Claude, BERT, Stable Diffusion, and more.
