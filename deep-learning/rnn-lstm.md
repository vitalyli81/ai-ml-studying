# RNN & LSTM (Recurrent Neural Networks)

## What Is It?

An RNN is a neural network that processes **sequences** — data where **order matters**. Unlike a regular network that sees each input independently, an RNN has **memory**: it passes information from one step to the next.

Think of it like reading a sentence word by word — you remember what came before to understand what comes next.

## Frontend Analogy — The Reducer

An RNN works exactly like `Array.reduce()`:

```javascript
// Regular network: process each item independently
items.map(item => processItem(item));

// RNN: process items in order, passing state from one to the next
items.reduce((state, item) => {
  const newState = processItem(item, state);  // use previous state + current item
  return newState;
}, initialState);
```

Each step takes the **current input** and the **previous state**, and produces a **new state**. That's an RNN.

## Real-World Examples

- **Text generation** — predict the next word in a sentence
- **Machine translation** — English → French (sequence to sequence)
- **Speech recognition** — audio waveform → text
- **Stock price prediction** — past prices → future prices
- **Music generation** — past notes → next note
- **Sentiment analysis** — read a review → positive/negative

## How a Basic RNN Works

### Step by Step Through a Sentence

Processing: "I love coding"

```
Step 1: Input "I"
  state₁ = f(W × "I" + U × state₀)        state₀ = zeros (no memory yet)

Step 2: Input "love"
  state₂ = f(W × "love" + U × state₁)     state₁ carries info about "I"

Step 3: Input "coding"
  state₃ = f(W × "coding" + U × state₂)   state₂ carries info about "I love"

Output = g(state₃)                          Final state has the full sentence context
```

Visually:

```
"I"          "love"        "coding"
 ↓             ↓             ↓
┌────┐       ┌────┐       ┌────┐
│RNN │──h₁──→│RNN │──h₂──→│RNN │──h₃──→ Output
│Cell│       │Cell│       │Cell│
└────┘       └────┘       └────┘

Same cell, same weights, reused at each step
h = hidden state (the "memory")
```

**Key insight:** It's the **same cell** with the **same weights** at every step. The only thing that changes is the input and the hidden state.

## The Problem: Vanishing Gradients (Again)

Basic RNNs forget. In a long sequence, information from early steps **fades away**:

```
"The cat, which was sitting on the mat in the living room next to the
 fireplace where the family gathers every evening during winter, was ..."

By the time we get to "was", the RNN has forgotten "cat".
Gradient has been multiplied through ~20 steps → vanished to near zero.
```

Basic RNN memory: ~5-10 steps back. After that, it forgets.

## LSTM — The Solution (Long Short-Term Memory)

LSTM solves the memory problem with **gates** — learned mechanisms that control what to **remember**, what to **forget**, and what to **output**.

### The LSTM Cell — A Smart Memory Unit

Think of an LSTM cell as a **to-do app** with 3 controls:

```
┌──────────────────────────────────┐
│            LSTM Cell             │
│                                  │
│  🗑 FORGET GATE: "Delete old tasks that are done"
│     → Decides what to remove from memory
│                                  │
│  📥 INPUT GATE: "Add new tasks"
│     → Decides what new info to store
│                                  │
│  📤 OUTPUT GATE: "Show relevant tasks"
│     → Decides what memory to use for current output
│                                  │
│  📋 CELL STATE: "The actual to-do list"
│     → Long-term memory that flows through unchanged
│        (unless gates modify it)
└──────────────────────────────────┘
```

### Why Gates Work

The cell state is like a **conveyor belt** — information flows through unchanged by default:

```
Cell state:  ──────────────────────────────────→
                  ↑ add        ↑ add        ↑ add
                  ↓ forget     ↓ forget     ↓ forget
             [LSTM Cell]  [LSTM Cell]  [LSTM Cell]
                  ↑             ↑             ↑
               input₁       input₂       input₃
```

Without intervention, the cell state passes through unmodified (gradient = 1, no vanishing!). The gates learn **when** to add or remove information. This is why LSTM can remember things 100+ steps back.

### LSTM vs Basic RNN

| Basic RNN | LSTM |
|-----------|------|
| 1 simple state | Cell state + hidden state |
| No gates | 3 gates (forget, input, output) |
| Forgets after ~5-10 steps | Remembers 100+ steps |
| Simple, fast | More complex, slower |
| Rarely used alone anymore | Still widely used |

## GRU — The Simplified LSTM

GRU (Gated Recurrent Unit) is a **lighter version** of LSTM with 2 gates instead of 3:

```
LSTM: 3 gates (forget, input, output) + cell state
GRU:  2 gates (reset, update) — no separate cell state
```

- **Similar performance** to LSTM in most tasks
- **Faster** to train (fewer parameters)
- **Use GRU** when you're not sure — it's simpler and usually works just as well

## Bidirectional RNNs

Standard RNN only reads left → right. But sometimes you need context from **both directions**:

```
"The bank of the river"     ← "bank" means riverbank
"The bank approved the loan" ← "bank" means financial institution
```

A bidirectional RNN reads **both ways** and combines:

```
Forward:   → → → → →
                        } Combine outputs
Backward:  ← ← ← ← ←
```

```python
lstm = nn.LSTM(input_size=100, hidden_size=256, bidirectional=True)
# Output size doubles: 256 × 2 = 512
```

## When to Use RNNs/LSTMs (and When NOT To)

| Good For | Use Instead |
|----------|-------------|
| Short sequences (< 100 tokens) | **Transformers** for long sequences |
| Time series forecasting | **Transformers** for NLP tasks |
| Simple sequence tasks | **Transformers** for state-of-the-art results |
| Low-resource environments | **Transformers** if you have GPU |
| Real-time streaming data | **Transformers** for batch processing |

**Important reality check:** For most NLP tasks today, **Transformers have replaced RNNs/LSTMs**. But understanding RNNs is essential because:
1. They explain why Transformers were invented
2. They're still used for time series and streaming
3. LSTM concepts (gating) appear everywhere in modern architectures

## Python Example

```python
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        # Convert word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM processes the sequence
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,          # stack 2 LSTM layers
            batch_first=True,      # input shape: (batch, sequence, features)
            dropout=0.3,           # regularization between layers
            bidirectional=True,    # read both directions
        )

        # Classify based on final hidden state
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # × 2 because bidirectional
            nn.ReLU(),
            nn.Linear(64, 1),               # 1 output: positive/negative
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length) — word indices

        embedded = self.embedding(x)        # (batch, seq, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Take the last hidden state from both directions
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        output = self.classifier(last_hidden)
        return output

# Create model
model = SentimentLSTM(vocab_size=10000)

# Dummy input: batch of 4 sentences, each 50 words (as word indices)
x = torch.randint(0, 10000, (4, 50))
output = model(x)
print(f"Output shape: {output.shape}")  # [4, 1] — one score per sentence
```

## Key Takeaway

RNNs process sequences by maintaining a **hidden state** that carries information from step to step — like `Array.reduce()`. Basic RNNs forget too quickly, so **LSTM** adds gates to control memory (what to keep, what to forget, what to output). While Transformers have largely replaced LSTMs for NLP, understanding RNNs teaches you **why sequence modeling is hard** and prepares you for the Transformer architecture. LSTMs are still the go-to for time series and streaming data.
