# Transformers

## 1. TL;DR

The Transformer is the architecture behind every major AI system today — GPT, Claude, BERT, Stable Diffusion, Whisper. It processes entire sequences in parallel (not step-by-step like RNNs) using **self-attention**: every token can directly look at every other token. This solves RNNs' forgetting problem and scales beautifully with more compute and data. The core mechanism is QKV attention — a learned, fuzzy database lookup. Understanding Transformers is the single most important architectural concept in modern AI.

---

## 2. The Mental Model

> 💡 **Think of it as a meeting room where everyone can talk to everyone simultaneously — vs. RNNs where information passes through a telephone chain.**

In a telephone chain (RNN), person 1 whispers to person 2, who whispers to person 3... by person 20, the original message is distorted or lost. In a meeting room (Transformer), every person can speak directly to every other person at once. Nothing gets lost in transit.

- **Meeting room** → Transformer (all tokens communicate in parallel)
- **Telephone chain** → RNN (information passes sequentially, step-by-step)
- **Person speaking** → a token (word, image patch, audio frame)
- **Deciding who to listen to** → attention scores (how much does token A attend to token B?)
- **Everyone talking at once** → parallel processing (why Transformers love GPUs)
- **A person's full understanding from the conversation** → contextual embedding (enriched by attended information)
- **Multiple conversation threads simultaneously** → multi-head attention

---

## 3. Why It Exists

**The problem:** RNNs process sequences step-by-step and forget distant context. Their sequential nature also means you can't parallelize training across time steps — making them slow on GPUs.

**What came before:** RNNs with attention (Bahdanau, 2014) added a mechanism for the decoder to look at encoder states directly — this was the first form of attention. It helped but was still built on sequential RNN processing.

**What changed:** "Attention Is All You Need" (Vaswani et al., 2017) removed the RNN entirely. Pure attention + feedforward layers, fully parallelizable, no sequential bottleneck. Training speed increased dramatically. And the key empirical discovery: **scaling works** — more parameters + more data + more compute = reliably better models. This scaling law is what produced GPT-3, GPT-4, Claude, and everything since.

---

## 4. Core Concepts

### Self-Attention

**One-line definition:** For each token, compute a weighted sum of all other tokens' information — where weights reflect how relevant each other token is.

**Analogy:** When you read "The bank of the river," your brain instantly flags that "bank" should attend strongly to "river" (disambiguating "bank" = riverbank, not financial). Self-attention does this computationally for every word, simultaneously.

```
Sentence: "The cat sat on the mat because it was tired"

Attention weights for "it":
  "The" → 0.01    (low relevance)
  "cat" → 0.72    (high — "it" refers to "cat"!)
  "sat" → 0.08
  "on"  → 0.02
  "mat" → 0.09    (low — "it" is the cat, not the mat)
  "it"  → 0.05    (self-reference)
  "was" → 0.03
```

Result: the embedding for "it" now contains ~72% cat information — it "knows" it refers to the cat.

**Common misconception:** ❌ "Attention is like a lookup table" → ✅ It's a *soft* lookup — instead of finding one exact match, it returns a weighted blend of all values proportional to relevance scores.

---

### Query, Key, Value (QKV)

**One-line definition:** Each token projects into three roles: Query (what am I looking for?), Key (what do I contain?), Value (what information do I share if selected?).

**Analogy:** A search engine:
```javascript
// Query: your search terms           → "cat videos"
// Keys:  page titles/metadata        → "funny cats", "stock market", "recipes"
// Values: page content               → actual video/article content
// Attention score: relevance ranking → [0.85, 0.02, 0.01, ...]
// Output: weighted blend of content  → mostly cat video content
```

**Technical explanation:**
```
For token x:
  Q = x × W_Q   ← "what am I looking for?"
  K = x × W_K   ← "what do I offer?"
  V = x × W_V   ← "my actual information"

Attention score between token i and j:
  score(i,j) = Q_i · K_j / sqrt(d_k)     ← dot product, scaled

Attention weights:
  weights = softmax(scores)               ← convert to probabilities

Output for token i:
  output_i = Σ weights(i,j) × V_j        ← weighted sum of all values
```

**Common misconception:** ❌ "Q, K, V are the same vector" → ✅ They're three separate linear projections of the same input — each learned independently for different purposes.

---

### Multi-Head Attention

**One-line definition:** Run N independent attention operations in parallel, each learning to attend to different types of relationships.

**Analogy:** A team of specialists analyzing the same document simultaneously — one checks grammar, one tracks entities, one monitors sentiment, one maps logical flow. Then combine all their reports.

```
Head 1: "sat" attends to subject    → "cat" (who did the action?)
Head 2: "sat" attends to location   → "mat" (where?)
Head 3: "sat" attends to verb type  → past tense pattern
Head 4: "sat" attends to structure  → verb in SVO pattern
...
Head 8: "sat" attends to sentiment  → neutral/calm

All 8 heads concatenated → rich, multi-faceted understanding of "sat"
```

```python
attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
# 8 heads, each with dim_per_head = 512/8 = 64
```

**Common misconception:** ❌ "More heads always = better" → ✅ Each head has `embed_dim / num_heads` dimensions — more heads = smaller per-head capacity. Standard is 8 or 16 heads.

---

### Positional Encoding

**One-line definition:** Information added to each token's embedding to tell the model where in the sequence that token appears.

**Analogy:** Numbering every person in the meeting room. Without numbers, the model sees the same "bag of people" regardless of seating order. Position numbers let it know "person at seat 1 spoke before person at seat 5."

```
"Dog bites man" ≠ "Man bites dog"

Without positional encoding: model sees {Dog, bites, man} — unordered set
With positional encoding:    model sees [(Dog,pos=0), (bites,pos=1), (man,pos=2)]
```

Two types:
- **Sinusoidal** (original paper): fixed mathematical formula, generalizes to any length
- **Learned** (GPT, BERT): learned during training, simpler to implement

**Common misconception:** ❌ "Positional encoding is trivial — just add numbers" → ✅ It must be addable to token embeddings (same dimension), periodic to generalize across lengths, and not dominate the semantic signal. The sinusoidal design carefully balances these requirements.

---

### Feed-Forward Network (FFN)

**One-line definition:** A small MLP applied independently to each token's representation after attention — processes each position separately.

**Analogy:** After the group discussion (attention), each person goes to their desk and individually processes what they heard to form their own conclusions (FFN).

```python
# Standard FFN in a Transformer layer
nn.Sequential(
    nn.Linear(d_model, 4 * d_model),  # expand (typical: 4×)
    nn.GELU(),
    nn.Linear(4 * d_model, d_model),  # compress back
)
```

This is where most of the Transformer's "knowledge storage" happens — attention routes information; FFN transforms it.

**Common misconception:** ❌ "Attention does all the work" → ✅ Research shows FFN layers store factual knowledge (facts about the world), while attention layers handle routing/context. Both are essential.

---

### Encoder vs. Decoder vs. Encoder-Decoder

**One-line definition:** Three variants of Transformer architecture for three different task types.

**Analogy:**
- **Encoder-only (BERT)**: A reader who understands the full text — great at comprehension, bad at generation
- **Decoder-only (GPT, Claude)**: A writer who generates text left-to-right, never looking ahead
- **Encoder-Decoder (T5, original Transformer)**: A translator — reads the full source, writes the target

```
BERT (encoder-only):       reads "I love [MASK]" → fills in "coding"
                           uses bidirectional attention — sees full context
                           great for: classification, NER, QA

GPT/Claude (decoder-only): reads "I love" → predicts "coding"
                           uses MASKED attention — can only see past tokens
                           great for: generation, completion, chatbots

T5 (encoder-decoder):     reads "translate French to English: Je t'aime"
                           → writes "I love you"
                           great for: translation, summarization, seq2seq
```

**Common misconception:** ❌ "BERT and GPT are the same architecture" → ✅ BERT uses bidirectional attention (sees all tokens); GPT uses causal/masked attention (sees only previous tokens). This fundamental difference determines what each is good for.

---

## 5. How It Actually Works — Step by Step

Processing "I love coding" through one Transformer encoder layer:

```
Step 1: TOKENIZE + EMBED
  "I love coding" → token IDs [42, 831, 156]
  Each ID → embedding vector (512-dim learned)
  Embeddings: [e₁, e₂, e₃]  shape: [3, 512]

Step 2: ADD POSITIONAL ENCODING
  e₁ += pos_enc(0)   ← "I is at position 0"
  e₂ += pos_enc(1)   ← "love is at position 1"
  e₃ += pos_enc(2)   ← "coding is at position 2"
  Shape still: [3, 512]

Step 3: SELF-ATTENTION (1 head for simplicity)
  Compute Q, K, V for each token:
    Q = embeddings × W_Q   shape: [3, 64]
    K = embeddings × W_K   shape: [3, 64]
    V = embeddings × W_V   shape: [3, 64]

  Attention scores = Q × Kᵀ / sqrt(64)
    shape: [3, 3]  ← each token scores against all tokens

  Attention weights = softmax(scores)
    e.g., for "coding": [0.1, 0.2, 0.7]  ← attends mostly to itself

  Output = weights × V
    shape: [3, 64]  ← each token is now a blend of all values

Step 4: MULTI-HEAD ATTENTION (8 heads)
  Run 8 independent attention operations (each with 64 dims)
  Concatenate: 8 × 64 = 512 dims
  Project: Linear(512, 512)  ← final projection

Step 5: ADD & NORM (residual connection)
  output = LayerNorm(attention_output + embeddings)
  ← skip connection: original embedding is ADDED BACK
  ← gradient flows through the skip connection freely (no vanishing!)

Step 6: FEED-FORWARD NETWORK
  For each of the 3 positions independently:
    x = GELU(Linear(512, 2048)(x))
    x = Linear(2048, 512)(x)

Step 7: ADD & NORM again
  output = LayerNorm(ffn_output + step5_output)

Result: 3 enriched embeddings, each containing context from the whole sentence
```

Repeat this for 12 (BERT-base) or 96 (GPT-3) layers.

---

## 6. Code in Practice

### Minimal — Single attention operation
```python
import torch
import torch.nn as nn

attention = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    batch_first=True,
)

x = torch.randn(4, 20, 512)   # batch=4, seq_len=20, embed=512
output, attn_weights = attention(x, x, x)   # Q=K=V=x → self-attention
print(output.shape)       # [4, 20, 512] — same shape in, same shape out
print(attn_weights.shape) # [4, 20, 20]  — attention matrix
```

### Practical — Full Transformer encoder
```python
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256, nhead=8, num_layers=4, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = nn.Embedding(512, d_model)   # learned positions

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, padding_mask=None):
        B, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoding(positions)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.classifier(x[:, 0, :])   # use [CLS] token (first position)

model = TransformerClassifier()
x = torch.randint(1, 10000, (4, 50))   # batch=4, seq_len=50
print(model(x).shape)                   # [4, 3]
```

### Real-world — Use pretrained Transformer (Hugging Face)
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load pretrained BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

texts = ["I love coding", "Deep learning is fascinating"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)

# CLS token embedding = sentence representation
cls_embeddings = outputs.last_hidden_state[:, 0, :]
print(cls_embeddings.shape)   # [2, 768] — one 768-dim vector per sentence
```

---

## 7. Gotchas & Pitfalls

| ❌ Wrong Assumption | ✅ Reality |
|---|---|
| Transformers have no position information | Without positional encoding, the model is completely position-blind; it's essential |
| BERT can generate text like GPT | BERT uses bidirectional attention — it sees future tokens during training; it can't generate autoregressively |
| Context window = max understanding | The model sees all tokens in the window, but very long contexts dilute attention — quality degrades near limits |
| Attention = the whole Transformer | FFN layers are equally important — they store factual knowledge while attention routes context |
| Bigger = always better | Larger models can be worse with small datasets or wrong fine-tuning approach; match model size to your data |
| `src_key_padding_mask` is optional | Without it, the model wastes attention on padding tokens, degrading performance on variable-length inputs |
| Transformers don't need regularization | Dropout is applied to attention weights and FFN outputs; modern Transformers also use LayerNorm heavily |

---

## 8. When to Use / When NOT to Use

**Use Transformers (encoder-only / BERT-style) when:**
- Text classification, named entity recognition, question answering
- You have the full sequence upfront (no streaming)
- You need bidirectional context (each token should see before AND after)

**Use Transformers (decoder-only / GPT-style) when:**
- Text generation, code completion, chatbots
- You need to generate sequences token by token
- This is what GPT, Claude, LLaMA, and Mistral are

**Use Transformers (encoder-decoder / T5-style) when:**
- Translation, summarization, or any input→output sequence task
- You need to read one sequence and write another

**Use RNNs/LSTMs instead when:**
- Real-time streaming with hard latency requirements (Transformers are memory-hungry)
- Time series with very short sequences and limited compute

**Use CNNs instead when:**
- Pure image tasks with limited data (CNNs still win on small datasets)
- You don't need cross-patch attention

---

## 9. Related Concepts (The Map)

- **RNNs/LSTMs** — what Transformers replaced for NLP; understanding why RNNs failed (sequential, forgetting) makes Transformers click (see `rnn-lstm.md`)
- **Transfer learning** — BERT, GPT, LLaMA are pretrained Transformers; fine-tuning them is the most common AI engineering task (see `transfer-learning.md`)
- **Attention is a generalization of CNNs** — CNNs use local, fixed filters; attention uses global, dynamic, input-dependent filters. Vision Transformers (ViT) apply this to images
- **Scaling laws** — Transformers follow power-law scaling: 10× the compute → predictable performance improvement. This is why companies race to build larger models
- **Residual connections** — every Transformer layer has skip connections (`output = LayerNorm(sublayer(x) + x)`); these solve vanishing gradients in very deep networks (see `backpropagation.md`)

---

## 10. Cheat Sheet

| Architecture | Examples | Attention Type | Best For |
|---|---|---|---|
| **Encoder-only** | BERT, RoBERTa | Bidirectional | Classification, NER, embeddings |
| **Decoder-only** | GPT, Claude, LLaMA | Causal (masked) | Generation, completion |
| **Encoder-Decoder** | T5, BART | Both | Translation, summarization |

**Key hyperparameters:**
| Parameter | Typical Values |
|---|---|
| `d_model` (embedding dim) | 256 (small) → 12,288 (GPT-4 estimated) |
| `nhead` (attention heads) | 8 or 16 |
| `num_layers` | 6 (small) → 96 (large) |
| `dim_feedforward` | 4 × d_model |
| `dropout` | 0.1 |
| Context window | 512 (BERT) → 200K+ (Claude) |

**Attention formula:**
```
Attention(Q, K, V) = softmax(Q × Kᵀ / sqrt(d_k)) × V
```

**Transformer layer structure:**
```
x → [Multi-Head Self-Attention] → Add & LayerNorm
  → [Feed-Forward Network]     → Add & LayerNorm
  → output (same shape as input)
```

**Remember these 3 things:**
1. Self-attention = every token looks at every other token simultaneously — the key insight that made everything possible
2. QKV: Query asks a question, Keys answer "what do I have?", Values are the actual content shared — like a fuzzy database lookup
3. Decoder-only (GPT/Claude) can only see past tokens; encoder-only (BERT) sees all tokens — this determines what each can do

---

## 11. Self-Check Questions

1. Why did Transformers replace RNNs for most NLP tasks? Name two specific advantages.
2. Explain what Q, K, and V are in self-attention using an analogy of your choice.
3. Why does a decoder-only model (GPT) use masked attention, while BERT uses full bidirectional attention?
4. What is the context window and why is there a limit?
5. You want to classify customer support tickets into 5 categories. Should you use an encoder-only or decoder-only Transformer? Why?

<details>
<summary>Brief Answers</summary>

1. **Parallel processing**: RNNs process tokens sequentially (token 1 → token 2 → ...) so you can't parallelize training across time steps. Transformers process all tokens simultaneously via attention — 10× to 100× faster on GPUs. **No forgetting**: RNNs pass information through a bottleneck hidden state that loses early context; Transformers use direct attention connections between any two positions (no information loss regardless of distance — "The cat... it" can still attend directly to "cat" even 50 tokens later).

2. *(Using a library lookup analogy)* The **Query** is your search request — "I'm looking for books about machine learning." The **Keys** are each book's catalog entry/tags — "ML textbook", "cooking", "history". The **attention score** is how well your query matches each key. The **Value** is the actual book content. The output is a weighted blend of all book content, weighted by how relevant each book is to your query — mostly ML content, a tiny bit of related topics. In a Transformer, each token is simultaneously a searcher (Q) and a potential result (K and V).

3. **GPT generates text left-to-right, one token at a time** — when predicting token N, it must NOT see tokens N+1, N+2... (that would be cheating). Masked attention enforces this: a triangular mask sets all future token attention scores to -∞ (→ softmax → 0). **BERT's task is understanding, not generation** — it's trained by masking random tokens and predicting them from surrounding context (bidirectional). Since BERT sees the complete sentence at once (in inference), it can and should use full bidirectional attention for maximum understanding.

4. The context window is the maximum number of tokens the model can process in a single forward pass. There's a limit because **attention is O(n²)** in memory — every token attends to every other token, so doubling the sequence length quadruples the attention matrix size. A 1000-token context needs 1M attention cells; 100K tokens need 10 billion. Practical limits are set by GPU memory. Modern techniques (Flash Attention, sliding window attention) push this boundary further, but it remains a fundamental constraint.

5. **Encoder-only (BERT-style)**. Classification is a *understanding* task — you have the full ticket text upfront and need to categorize it. An encoder can use bidirectional attention to build a rich representation of the entire ticket, then feed the [CLS] token through a linear classifier. A decoder-only model is designed for *generation* — predicting the next token — and while you can use it for classification (with prompt engineering), it's the wrong tool and typically performs worse than a fine-tuned BERT for this task.

</details>

---

## 12. Go Deeper

- **"Attention Is All You Need" (Vaswani et al., 2017)**: The foundational paper. Sections 3-5 describe the architecture with full equations. Surprisingly readable for a landmark paper. [Why: this is the document that defines the Transformer — reading the original gives you terminology and intuition that secondary sources often misstate.]

- **Andrej Karpathy — "Let's build GPT from scratch"** (YouTube, ~2 hours): Builds a GPT model character-by-character in pure PyTorch, explaining every line. The best video on Transformers that exists. [Why: after watching this, you'll be able to implement a Transformer from memory — the deepest possible understanding.]

- **Jay Alammar — "The Illustrated Transformer"** (jalammar.github.io/illustrated-transformer): Beautiful step-by-step visual walkthrough of attention and the full Transformer architecture. [Why: if the math feels abstract, Alammar's diagrams make it concrete — the animated attention visualizations are unforgettable.]

- **"BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)**: Explains the masked language modeling pretraining objective and how BERT's representations work. [Why: BERT fine-tuning is the most common NLP task you'll do as an AI engineer — knowing the architecture at paper level is genuinely useful for debugging.]

- **Hugging Face — "The Annotated Transformer"** (nlp.seas.harvard.edu/annotated-transformer): A line-by-line Python implementation of the original Transformer with extensive comments. [Why: combines the code and the theory in one place — the gap between "I understand attention" and "I can implement it" closes completely after reading this.]
