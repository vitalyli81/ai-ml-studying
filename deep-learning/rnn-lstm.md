# RNN & LSTM (Recurrent Neural Networks)

## 1. TL;DR

RNNs process sequences by passing a hidden state from one step to the next — like `Array.reduce()` where the accumulator carries memory forward. Basic RNNs forget after ~10 steps due to vanishing gradients. LSTMs fix this with learned gates that control what to remember and what to forget, enabling memory of 100+ steps. For most NLP tasks today, Transformers have replaced LSTMs — but LSTMs still dominate time series and streaming data, and understanding them is essential for understanding why Transformers exist.

---

## 2. The Mental Model

> 💡 **An RNN is exactly `Array.reduce()` — same structure, same purpose.**

```javascript
// Regular processing (like a dense network): items processed independently
items.map(item => process(item));

// RNN processing: items processed in order, state passes forward
items.reduce((state, item) => {
    const newState = process(item, state);  // use history + current input
    return newState;
}, initialState);
```

- **Accumulator (state)** → hidden state h (the "memory")
- **Current item** → current input token (a word, time step, frame)
- **Processing function** → the RNN cell (same weights at every step)
- **Final accumulator value** → final hidden state (the full sequence summary)
- **Reduce's initial value** → h₀ (zeros — no memory at start)
- **Different reduce functions** → RNN vs GRU vs LSTM (different "processing functions")

---

## Build the Intuition From Zero

Two things to truly get: **what the "hidden state" actually is, and why a plain RNN forgets — which is the entire reason LSTMs exist.**

### Idea 1: The hidden state is a running summary

Read this sentence one word at a time and, after each word, hold a single mental note that summarizes everything so far. That note is the **hidden state** `h`. The RNN updates it at every step: `new note = combine(old note, current word)`.

```
"The"   → h: "a sentence starts, subject coming"
"cat"   → h: "subject is: cat (animal, singular)"
"sat"   → h: "the cat is doing: sitting"
"on"    → h: "...sitting on something, location coming"
```

The note has a fixed size — it doesn't grow with the sentence. So the RNN is constantly **cramming the whole past into one small summary** and overwriting it each step. That works for short sequences and is exactly the `reduce()` accumulator above.

### Idea 2: Why plain RNNs forget (the squeeze)

Because the same small note gets overwritten at every step, information from long ago gets diluted away — like a rumor passed down a line of 50 people. Worse, training makes it mathematically severe. Recall from [backpropagation.md](backpropagation.md) that gradients **multiply** back through every step:

```
20 steps back: gradient ≈ (0.6)²⁰ ≈ 0.00004  →  VANISHES
  → the word from 20 steps ago receives essentially zero learning signal
  → the RNN literally cannot learn long-range connections
```

So a plain RNN's memory fades after ~10 steps. "The clouds are in the ___" (easy, recent) it handles; "I grew up in France… [50 words]… I speak fluent ___" it can't, because *France* faded.

### Idea 3: LSTM = a notebook with gates instead of one note

The LSTM's fix: add a separate **cell state** — a notebook that information can ride along *unchanged* — plus three little **gates** (each a 0–1 dial, learned) that control it:

```
FORGET gate → "erase this from the notebook?"     (0 = wipe, 1 = keep)
INPUT  gate → "write this new info to the notebook?"
OUTPUT gate → "what part of the notebook do I reveal as the answer right now?"
```

Because relevant facts can sit on the notebook untouched (gate ≈ 1) instead of being overwritten every step, the gradient rides along that highway without vanishing — so LSTMs remember 100+ steps. That's the whole trick: **gates decide what to keep, write, and read**, replacing the RNN's blind overwrite.

> 💡 **One line:** an RNN crams the past into one tiny note it overwrites each step (so it forgets); an LSTM adds a protected notebook with keep/write/read gates (so it remembers). And the *true* fix for long memory — letting every word look at every other word directly — is the Transformer ([transformers.md](transformers.md)), which is why this doc exists mostly to explain why Transformers won.

The cell-state and gate sections below formalize these dials.

---

## 3. Why It Exists

**The problem:** Standard neural networks process each input independently — feed "cat" in position 1 and "cat" in position 50, and the network treats them identically. There's no way to represent that "the bank" means different things in "river bank" vs "bank account" depending on surrounding context.

**What came before:** N-gram models (count word co-occurrences), hidden Markov models (probabilistic state machines). Both failed to capture long-range dependencies.

**What changed:** RNNs (introduced 1980s, popularized 1990s) provided a principled way to process variable-length sequences by maintaining a persistent state. LSTMs (Hochreiter & Schmidhuber, 1997) solved RNNs' forgetting problem with gating mechanisms, enabling everything from machine translation to speech recognition through the 2010s.

---

## 4. Core Concepts

### Hidden State

**One-line definition:** A vector that persists between steps, carrying information about what the RNN has seen so far.

**Analogy:** The accumulator in `reduce()` — it carries the running total (summary of past inputs) forward to influence the processing of each new item.

```
"I"     → RNN Cell → h₁ = f("I",   h₀)
"love"  → RNN Cell → h₂ = f("love", h₁)  ← h₁ carries info about "I"
"coding"→ RNN Cell → h₃ = f("coding", h₂) ← h₂ carries info about "I love"
```

**Common misconception:** ❌ "The hidden state stores raw past words" → ✅ It stores a learned, compressed representation — not the words themselves, but patterns and relationships the network found important.

---

### The Vanishing Gradient Problem in RNNs

**One-line definition:** In long sequences, gradient signals shrink exponentially as they travel back through time steps — early steps stop learning.

**Analogy:** Shouting a message down a long hallway where each room absorbs 50% of the sound. By room 20, no one hears anything.

```
"The cat, which sat on the mat by the fire in the kitchen, was ..."

By the time we need "was" to agree with "cat" (20 words back),
the gradient from "was" has been multiplied through 20 steps:
  0.7^20 ≈ 0.0008   ← nearly zero, "cat" gets no update signal
```

Basic RNN practical memory limit: ~5–10 steps. After that, it forgets.

**Common misconception:** ❌ "More hidden units solve the forgetting problem" → ✅ Larger hidden states store more *at any given step*, but they still forget long-range dependencies. The problem is gradient flow, not storage capacity.

---

### LSTM Gates

**One-line definition:** Three learned mechanisms that control what information to keep, what to add, and what to output from the LSTM cell.

**Analogy:** Think of an LSTM cell as a to-do list app with smart controls:

```
┌────────────────────────────────────────┐
│               LSTM Cell                │
│                                        │
│  🗑 FORGET GATE  "Delete done tasks"   │
│    → What old memory to erase          │
│                                        │
│  📥 INPUT GATE   "Add new tasks"       │
│    → What new information to store     │
│                                        │
│  📤 OUTPUT GATE  "Show relevant tasks" │
│    → What memory to expose as output   │
│                                        │
│  📋 CELL STATE   "The actual list"     │
│    → Long-term memory (conveyor belt)  │
└────────────────────────────────────────┘
```

**The key insight — the cell state as a conveyor belt:**

```
Cell state:  ─────────────────────────────────→
                 ↑ add    ↑ add    ↑ add
                 ↓ forget ↓ forget ↓ forget
             [LSTM]   [LSTM]   [LSTM]
                ↑         ↑         ↑
             word₁     word₂     word₃
```

The cell state flows through unchanged by default (gradient ≈ 1 — no vanishing!). Gates selectively modify it. This is why LSTMs remember 100+ steps.

**The cell state update in one equation:**

```
c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ c̃_t
       └──────────┘     └──────────┘
       keep old memory   add new memory
       (scaled by        (scaled by
        forget gate)      input gate)

where:
  f_t = σ(W_f · [h_{t-1}, x_t])    ← forget gate  ∈ (0, 1)
  i_t = σ(W_i · [h_{t-1}, x_t])    ← input gate   ∈ (0, 1)
  c̃_t = tanh(W_c · [h_{t-1}, x_t]) ← candidate new info
  ⊙ = element-wise multiplication
```

Read this intuitively: **"new memory = (how much to keep) × old memory + (how much to add) × new candidate."** A forget gate of 1.0 means "remember everything"; 0.0 means "erase it all." Because this update is *additive* (not multiplicative through a squashing function like vanilla RNN's `h_t = tanh(W·[h_{t-1}, x_t])`), gradients flowing back through time don't get squashed at each step — they flow through the `+` almost unchanged. That's the mathematical fix for vanishing gradients.

**Common misconception:** ❌ "LSTM gates are fixed rules" → ✅ All gate values (`f_t`, `i_t`, `o_t`) are outputs of learned linear layers + sigmoid. The network learns *what is worth remembering* and *when to forget* for each specific task.

---

### GRU (Gated Recurrent Unit)

**One-line definition:** A simplified LSTM with 2 gates instead of 3 — similar performance, fewer parameters, faster to train.

**Analogy:** A folding knife vs a Swiss Army knife — the GRU does the job with fewer tools.

```
LSTM: forget gate + input gate + output gate + cell state
GRU:  reset gate + update gate (no separate cell state)

Performance: usually within 1-2% of LSTM
Speed: ~30% faster to train (fewer parameters)
Rule of thumb: start with GRU, switch to LSTM only if clearly needed
```

**Common misconception:** ❌ "LSTM is always better than GRU" → ✅ Empirically, they're neck-and-neck on most tasks. GRU is simpler and a fine default.

---

### Bidirectional RNNs

**One-line definition:** Two RNNs processing the sequence in opposite directions — combining both gives each position context from both past and future.

**Analogy:** Reading a sentence forward for grammar and backward for meaning, then combining both readings.

```
"The bank of the river"   ← "bank" meaning depends on "river" (comes AFTER)
  Forward:  → → → → →
  Backward: ← ← ← ← ←
  Combined: each word sees full sentence context
```

```python
lstm = nn.LSTM(input_size=100, hidden_size=256, bidirectional=True)
# Output dim doubles: 256 × 2 = 512 (forward + backward concatenated)
```

**Common misconception:** ❌ "Bidirectional RNNs can generate text" → ✅ They can only be used for tasks where you have the full sequence upfront (classification, named entity recognition). Text generation requires left-to-right (can't look ahead).

---

### Sequence-to-Sequence

**One-line definition:** An encoder RNN compresses the input sequence into a fixed vector; a decoder RNN expands it into an output sequence.

**Analogy:** Translation — you read (encode) the whole French sentence, form an understanding, then speak (decode) the English equivalent.

```
Input: "Je t'aime"
Encoder: "Je"→h₁→"t'aime"→h₂→"."→h₃ = [context vector]
                                              ↓
Decoder: h₃→"I"→"love"→"you"→"<end>"
```

**Common misconception:** ❌ "The context vector can hold unlimited information" → ✅ The fixed-size vector is a bottleneck — it can't faithfully encode long sequences. This is exactly what attention (and later Transformers) was designed to fix.

---

> 🧠 **Quick recall — answer out loud before scrolling on** (all answers are above):
> 1. What is the hidden state, in the `Array.reduce()` framing?
> 2. Why does a plain RNN forget after ~10 steps — what happens to the gradient?
> 3. Name the three LSTM gates and each one's job.
> 4. Why does the additive cell-state update fix vanishing gradients?
> 5. Streaming stock prices vs. translating a document — which gets an LSTM, which a Transformer?

---

## 5. How It Actually Works — Step by Step

Processing "I love coding" for sentiment classification:

```
Step 1: TOKENIZE
  "I love coding" → [token_1=42, token_2=831, token_3=156]

Step 2: EMBED
  Each token index → dense vector (learned)
  [42, 831, 156] → [[0.2, -0.5, ...], [0.8, 0.1, ...], [-0.3, 0.9, ...]]
  Shape: [sequence_len=3, embed_dim=128]

Step 3: LSTM FORWARD PASS

  t=1: input="I" embedding [128-dim]
       h₀ = zeros [256-dim]
       Forget gate: sigmoid(W_f × [h₀, x]) → what to erase from cell
       Input gate:  sigmoid(W_i × [h₀, x]) → what new info to add
       Cell update: tanh(W_c × [h₀, x])   → candidate new memory
       Cell state:  c₁ = forget×c₀ + input×cell_update
       Output gate: sigmoid(W_o × [h₀, x]) → what to expose
       h₁ = output_gate × tanh(c₁)

  t=2: input="love" + h₁ (carries info about "I")
       → h₂ (carries info about "I love")

  t=3: input="coding" + h₂
       → h₃ (carries info about full "I love coding")

Step 4: CLASSIFY
  h₃ → Linear(256, 1) → Sigmoid → 0.92 = "92% positive" ✓

Step 5: LOSS + BACKPROP THROUGH TIME
  Binary CE loss: -log(0.92) = 0.083
  Gradients flow backwards through t=3, t=2, t=1
  (Cell state highway preserves gradient ≈ 1 through each step)
  All weights updated: embedding, LSTM gates, classifier
```

---

## 6. Code in Practice

### Minimal — LSTM layer
```python
import torch
import torch.nn as nn

lstm = nn.LSTM(
    input_size=128,     # embedding dim
    hidden_size=256,    # hidden state size
    batch_first=True,   # input shape: [batch, seq, features]
)

x = torch.randn(4, 20, 128)     # batch=4, seq_len=20, embed=128
output, (h_n, c_n) = lstm(x)

print(output.shape)   # [4, 20, 256] — hidden state at every step
print(h_n.shape)      # [1, 4, 256]  — final hidden state
```

### Practical — Sentiment classifier with LSTM
```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),   # ×2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        embedded = self.embedding(x)              # [B, seq, embed]
        _, (h_n, _) = self.lstm(embedded)          # h_n: [4, B, hidden]
        # Concatenate final forward + backward hidden states
        last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, hidden*2]
        return self.classifier(last)               # [B, 1]

model = SentimentLSTM(vocab_size=10000)
x = torch.randint(1, 10000, (8, 50))   # batch=8, seq_len=50
print(model(x).shape)                   # [8, 1]
```

### Real-world — GRU for time series
```python
class TimeSeriesGRU(nn.Module):
    def __init__(self, input_features=5, hidden=128, forecast_steps=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden, forecast_steps)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])  # use last time step for prediction

# Input: [batch=32, seq_len=60, features=5] — 60 days of 5 stock metrics
model = TimeSeriesGRU(input_features=5, hidden=128, forecast_steps=1)
```

---

## 7. Gotchas & Pitfalls

| ❌ Wrong Assumption | ✅ Reality |
|---|---|
| `batch_first=False` by default | Default input shape is `[seq, batch, features]` — set `batch_first=True` to use `[batch, seq, features]` (more intuitive) |
| LSTM output is just the final hidden state | `output` contains hidden states at *every* step; `h_n` is just the final one. Use `output` for sequence labeling, `h_n` for classification |
| Bidirectional doubles hidden_size automatically | It doubles the **output** dimension — your downstream layer must account for `hidden_size * 2` |
| RNNs handle variable-length sequences natively | You need `pack_padded_sequence` and `pad_packed_sequence` for proper variable-length handling |
| GRU/LSTM are slow to train | They're sequential by nature — can't parallelize across time steps. On long sequences, this is real; Transformers are much faster |
| LSTM is the right choice for NLP | For NLP tasks where you have the full sequence, use Transformers. LSTM shines for streaming/real-time and time series |
| Forget to reset hidden state between batches | By default PyTorch initializes h₀=zeros — this is fine for independent sequences but you must explicitly manage state for streaming |

---

## 8. When to Use / When NOT to Use

**Use RNN/LSTM/GRU when:**
- Real-time/streaming data where you can't see the full sequence upfront
- Time series forecasting (stock prices, sensor readings, weather)
- Low-resource environments where Transformers' memory footprint is prohibitive
- Sequences with strong local structure and short-range dependencies (< 50 steps)
- Online learning where inputs arrive one at a time

**Do NOT use RNN/LSTM when:**
- Full NLP tasks (text classification, translation, summarization) — use Transformers
- You have GPU memory to spare — Transformers are faster and usually better
- Sequences are very long (> 200 tokens) — vanishing gradients will still hurt
- You need bidirectional context AND generation — use Transformer decoders instead

---

## 9. Related Concepts (The Map)

- **Vanishing gradients** — the core problem RNNs suffer from, and why LSTMs were invented; the same issue that ReLU and skip connections solve in feedforward networks (see `backpropagation.md`)
- **Transformers** — replaced LSTMs for most NLP tasks by solving the forgetting problem with attention (direct connections between all positions) rather than gating; understanding RNNs makes Transformers click (see `transformers.md`)
- **Attention mechanism** — originally added ON TOP of RNNs (Bahdanau attention, 2014) to let the decoder look directly at encoder outputs; this later evolved into self-attention and the Transformer architecture
- **Embeddings** — RNNs process token embeddings (`nn.Embedding`), not raw indices; same as Transformers (see `transformers.md`)
- **Regularization** — dropout works differently in RNNs: applied between layers, not within recurrent connections (use `dropout` param in `nn.LSTM`); see `regularization.md`

---

## 10. Cheat Sheet

| Model | Gates | Memory | Speed | Use When |
|---|---|---|---|---|
| **RNN** | 0 | ~5–10 steps | Fast | Almost never — use GRU instead |
| **GRU** | 2 | 50–100 steps | Medium | Time series, streaming (default) |
| **LSTM** | 3 | 100+ steps | Slower | When GRU isn't enough |
| **Transformer** | Attention | 1000s tokens | Fastest (parallel) | NLP, full-sequence tasks |

**LSTM output shapes (`batch_first=True`):**
```python
output, (h_n, c_n) = lstm(x)
# x:      [batch, seq_len, input_size]
# output: [batch, seq_len, hidden_size × num_directions]
# h_n:    [num_layers × num_directions, batch, hidden_size]
# c_n:    [num_layers × num_directions, batch, hidden_size]
```

**Remember these 3 things:**
1. RNN = `Array.reduce()` — same hidden state passed through each step
2. LSTM gates are what enable long-range memory — they control the cell state conveyor belt
3. For NLP: use Transformers. For time series / streaming: use GRU or LSTM.

---

## 11. Self-Check Questions

1. How is an RNN structurally similar to `Array.reduce()`? What maps to what?
2. Why do basic RNNs forget information from early in long sequences?
3. What problem do LSTM gates solve, and how does the cell state help?
4. You're building a real-time stock price predictor that takes in the last 30 days and outputs tomorrow's price. Should you use an LSTM or a Transformer? Why?
5. What does `bidirectional=True` do, and what constraint does it impose on the task?

<details>
<summary>Brief Answers</summary>

1. In `reduce(callback, initialValue)`: the **accumulator** maps to the **hidden state** (carries memory forward), the **current element** maps to the **current input token**, the **callback function** maps to the **RNN cell** (same function/weights reused at every step), and the **final accumulated value** maps to the **final hidden state** (the sequence summary).

2. Because gradients must travel backward through every time step via the chain rule — multiplied at each step. If each multiplication is slightly less than 1 (which happens with sigmoid/tanh activations that saturate), the gradient shrinks exponentially. After 20 steps: `0.7^20 ≈ 0.0008`. Early steps receive essentially no gradient signal and stop learning — the network "forgets" what happened there.

3. Vanilla RNNs have vanishing gradients because the hidden state is fully recomputed at every step (gradient must flow through all transformations). LSTMs add a **cell state** — a separate memory that flows through with minimal modification (gate outputs near 1 → gradient ≈ 1, no vanishing). The three gates (forget, input, output) *learn* what to add/remove/use from the cell state, allowing the network to preserve important information for 100+ steps.

4. **LSTM** (or GRU). A real-time predictor processes prices as they arrive — it's a streaming task where you're building state step by step and predicting the next value. Transformers require the full sequence in memory and process it all in parallel (not streaming). For this causal, time-ordered prediction task, an LSTM is more appropriate and computationally efficient.

5. `bidirectional=True` runs two LSTMs simultaneously — one left-to-right, one right-to-left — then concatenates their outputs at each position (doubling output dimension). This means each position has context from both the past and the future. **Constraint**: you must have the complete sequence available before processing. This means bidirectional RNNs **cannot generate text** or handle real-time streaming — they require the full input upfront.

</details>

---

## 12. Go Deeper

- **"Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)**: The original LSTM paper. Dense but foundational — reading section 1-3 gives you the exact mathematical motivation for each gate. [Why: understanding *why* the cell state was designed the way it was makes it memorable, not just a black box.]

- **Andrej Karpathy — "The Unreasonable Effectiveness of RNNs"** (karpathy.github.io/2015/05/21/rnn-effectiveness): The famous blog post where Karpathy trains character-level LSTMs to generate Shakespeare, Linux kernel code, and Wikipedia markup. [Why: the most convincing demonstration of what RNNs can learn — and a great hands-on project to replicate.]

- **Christopher Olah — "Understanding LSTMs"** (colah.github.io/posts/2015-08-Understanding-LSTMs): The best visual explanation of LSTM internals — gate diagrams, cell state flow, and intuition. [Why: if the LSTM cell still feels opaque after reading this doc, Olah's diagrams will make it click.]

- **PyTorch RNN tutorial** (pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial): Official character-level RNN tutorial for name classification. Short, runnable, and teaches the `pack_padded_sequence` API. [Why: practical PyTorch patterns for variable-length sequences that you'll use in real projects.]

- **"Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (Chung et al., 2014)**: The paper that compared RNN, GRU, and LSTM systematically. GRU often matches LSTM with fewer parameters. [Why: the empirical evidence for why GRU is a good default, not just intuition.]
