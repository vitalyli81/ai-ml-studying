# Activation Functions

## 1. TL;DR

Activation functions are the "squish" operations applied after every neuron's weighted sum. Without them, stacking 100 layers is mathematically identical to 1 layer — they're what makes deep learning *deep*. ReLU is the default for hidden layers (fast, effective). Sigmoid is for binary output. Softmax is for multi-class output. That covers 95% of cases.

---

## 2. The Mental Model

> 💡 **Without activations, stacking 100 layers is mathematically the same as 1 layer. Activations are what make depth actually mean something.**

Here's the proof in two lines. If each layer is `output = W × input + b` (pure linear), then stacking two layers gives:

```
layer2(layer1(x)) = W₂ × (W₁ × x + b₁) + b₂
                  = (W₂ × W₁) × x + (W₂ × b₁ + b₂)
                  = W_combined × x + b_combined     ← still one linear layer!
```

No matter how many layers you stack, it collapses to a single matrix multiply — so you can only draw straight decision boundaries. Activation functions break this collapse by inserting a **non-linear bend** between layers.

**Analogy — a light dimmer switch vs. an on/off switch.** A plain linear neuron is a pass-through wire. An activation function is a dimmer with a custom curve: clamp negatives to zero (ReLU), compress to 0–1 (sigmoid), or bend smoothly near zero (GELU). The *shape* of the bend is what lets the network carve curves, wiggles, and complex boundaries through the data.

- **Electrical signal strength** → raw neuron output (z = weighted sum)
- **Dimmer's response curve** → activation function shape
- **Light output** → activated neuron output
- **Different dimmers for different rooms** → different activations for different layers
- **Why you need the dimmer at all** → without non-linearity, depth gives you zero extra expressive power

---

## 3. Why It Exists

**The problem:** Matrix multiplication (what a neural network does) is linear — you can stack 1000 linear layers and they're mathematically equivalent to one. You can only model straight-line relationships.

**What came before:** Early networks used step functions (neuron is fully on or off) — these had zero gradient everywhere, making training impossible via backpropagation.

**What changed:** Sigmoid enabled gradient flow in the 1980s. ReLU (2010s) solved sigmoid's vanishing gradient problem and made deep networks trainable. GELU and SiLU now power modern Transformers.

---

## 4. Core Concepts

### ReLU (Rectified Linear Unit)

**One-line definition:** Output the input unchanged if positive; output 0 if negative.

**Analogy:** A one-way valve — water (positive signal) flows through; a vacuum (negative signal) is blocked to zero.

**Technical explanation:**
```
output = max(0, x)

x = -5  →  output = 0   (neuron "silent")
x = 0   →  output = 0
x = 3   →  output = 3   (neuron "active")
```

```
Output
  |          ╱
  |        ╱
  |      ╱
  |    ╱
  |  ╱
  |╱____________  Input
  0
```

**Code:**
```python
import torch.nn as nn
relu = nn.ReLU()
```

**Common misconception:** ❌ "ReLU is always safe to use" → ✅ The "Dying ReLU" problem: if a neuron's input is always negative during training, its gradient is always 0 — it permanently stops learning. Fix: use Leaky ReLU.

---

### Sigmoid

**One-line definition:** Squish any number into the (0, 1) range — perfect for probabilities.

**Analogy:** A volume knob that's physically capped between 0 and 1. No matter how hard you turn it, you can't go below 0 or above 1.

**Technical explanation:**
```
output = 1 / (1 + e^(-x))

x = -10  →  output ≈ 0.00005
x = 0    →  output = 0.5
x = 10   →  output ≈ 0.99995
```

```
Output
 1|            ___________
  |          ╱
  |        ╱
0.5|      ╱   ← S-shaped curve
  |    ╱
  |  ╱
 0|╱___________  Input
```

**Common misconception:** ❌ "Use sigmoid in hidden layers for probabilities" → ✅ Only use sigmoid on the **output** layer for binary classification. In hidden layers it causes vanishing gradients — gradients near 0 or 1 are nearly zero, so weights stop updating.

---

### Softmax

**One-line definition:** Converts a vector of raw scores into probabilities that sum to exactly 1.0.

**Analogy:** A vote counter — it doesn't matter if one candidate has 10 votes vs 100; softmax shows each candidate's *share* of the total.

**Technical explanation:**
```
Input:  [2.0, 1.0, 0.1]           ← raw scores (logits)
Output: [0.65, 0.24, 0.11]        ← probabilities (sum = 1.0)

Formula: softmax(xᵢ) = e^xᵢ / Σe^xⱼ
```

**Code:**
```python
import torch
import torch.nn.functional as F
logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
# tensor([0.6590, 0.2424, 0.0986])
```

**Common misconception:** ❌ "Add softmax to output layer when using CrossEntropyLoss in PyTorch" → ✅ `nn.CrossEntropyLoss` applies softmax internally. Adding it yourself applies softmax twice — outputs will be wrong.

---

### Leaky ReLU

**One-line definition:** Like ReLU, but negative inputs get a tiny slope instead of hard zero.

**Analogy:** The one-way valve that still lets a tiny trickle through when closed — enough to keep the gradient alive.

```
output = max(0.01 × x, x)

x = -5  →  output = -0.05  (tiny negative, not zero)
x = 3   →  output = 3
```

**Common misconception:** ❌ "Leaky ReLU is always better than ReLU" → ✅ In most cases they perform similarly. Only switch if you see evidence of dying neurons (many zero-gradient weights).

---

### GELU (Gaussian Error Linear Unit)

**One-line definition:** A smooth version of ReLU used in modern Transformers (GPT, BERT).

**Analogy:** ReLU is a sharp corner at 0; GELU is a smooth curve — it lets small negative values partially through.

```
GELU ≈ x × sigmoid(1.702 × x)

ReLU:   ______╱     (sharp corner)
GELU:   _____╱      (smooth transition near 0)
```

Used in GPT-2, GPT-3, BERT, and most modern architectures. You'll see it in Transformer configs.

**Common misconception:** ❌ "GELU is always better than ReLU" → ✅ For CNNs and standard MLPs, ReLU is usually equivalent and faster. GELU's advantage shows mainly in Transformers.

---

## 5. How It Actually Works — Step by Step

Processing a single neuron in a hidden layer:

```
Step 1: Receive weighted sum from previous layer
  z = 1.5×0.3 + (-0.8)×0.7 + 2.1×0.4 + bias(0.1)
  z = 0.45 - 0.56 + 0.84 + 0.1 = 0.83

Step 2: Pass through activation function
  ReLU(0.83) = max(0, 0.83) = 0.83    ✓ positive, passes through

  If z had been -0.83:
  ReLU(-0.83) = max(0, -0.83) = 0     ✗ blocked — neuron "silent"

Step 3: 0.83 goes forward to the next layer as input

Step 4: During backpropagation
  If neuron was active (z > 0): gradient flows through unchanged (derivative = 1)
  If neuron was silent (z < 0): gradient is blocked (derivative = 0) ← dying ReLU risk
```

The key insight: gradient flows through active ReLU neurons unimpeded, which is why they train much faster than sigmoid (which compresses gradients to near-zero).

---

## 6. Code in Practice

### Minimal — Apply activations manually
```python
import torch
import torch.nn as nn

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

relu    = nn.ReLU()
sigmoid = nn.Sigmoid()
leaky   = nn.LeakyReLU(0.01)
gelu    = nn.GELU()

print(f"Input:      {x.tolist()}")
print(f"ReLU:       {relu(x).tolist()}")      # [0, 0, 0, 1, 2]
print(f"Sigmoid:    {sigmoid(x).tolist()}")   # [0.12, 0.27, 0.5, 0.73, 0.88]
print(f"LeakyReLU:  {leaky(x).tolist()}")    # [-0.02, -0.01, 0, 1, 2]
print(f"GELU:       {gelu(x).tolist()}")      # [-0.045, -0.158, 0, 0.841, 1.954]
```

### Practical — Correct activation per layer type
```python
class Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),              # hidden layer → ReLU
            nn.Linear(64, 32),
            nn.ReLU(),              # hidden layer → ReLU
            nn.Linear(32, num_classes),
            # NO softmax here — CrossEntropyLoss handles it
        )

    def forward(self, x):
        return self.net(x)

# Binary classification output
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),           # output → probability 0-1
        )
```

### Activation in Transformers
```python
# Modern Transformer encoder layer uses GELU
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    activation='gelu',   # not relu!
    batch_first=True,
)
```

---

## 7. Gotchas & Pitfalls

| ❌ Wrong Assumption | ✅ Reality |
|---|---|
| Use sigmoid in hidden layers | Causes vanishing gradients — stick to ReLU for hidden layers |
| Always add softmax to the output | `nn.CrossEntropyLoss` already includes softmax — adding it doubles it |
| Sigmoid works for multi-class output | Sigmoid outputs don't sum to 1; use softmax for multi-class |
| ReLU on regression output | Clips negative predictions to 0; use no activation (linear) for regression |
| Dying ReLU can't be detected | Check: if many weights have zero gradients after training, neurons are dead |
| GELU is always better | ReLU is simpler, faster, and equally good for non-Transformer architectures |
| Activation functions are just cosmetic | Without them, depth provides zero benefit — they are fundamental |

---

## 8. When to Use / When NOT to Use

**Use ReLU when:**
- Default choice for hidden layers in any network
- CNNs, MLPs, general architectures
- You want fast computation with good performance

**Use Sigmoid when:**
- Binary classification output (predict yes/no probability)
- Gating mechanisms inside LSTM cells

**Use Softmax when:**
- Multi-class classification output (pick exactly one of N classes)

**Use GELU/SiLU when:**
- Building or fine-tuning Transformer models
- Replicating modern architectures (GPT, BERT, LLaMA)

**Do NOT use Sigmoid in hidden layers** — ever. Vanishing gradients will slow or halt training.

---

## 9. Related Concepts (The Map)

- **Backpropagation** — gradients flow backward through activation functions; this is why activation choice matters so much (see `backpropagation.md`)
- **Vanishing gradients** — sigmoid and tanh cause this; ReLU was invented to solve it (see `backpropagation.md`)
- **Loss functions** — the output layer activation pairs with the loss function: sigmoid + BCELoss, softmax + CrossEntropyLoss (see `loss-functions-optimizers.md`)
- **Neural network basics** — activations are the non-linear step in every neuron (see `neural-networks-basics.md`)
- **Transformers** — use GELU instead of ReLU; understanding why requires knowing what GELU's smooth curve offers (see `transformers.md`)

---

## 10. Cheat Sheet

| Function | Formula | Range | Use When |
|---|---|---|---|
| **ReLU** | max(0, x) | [0, ∞) | Hidden layers (default) |
| **Leaky ReLU** | max(0.01x, x) | (-∞, ∞) | Dying ReLU fix |
| **GELU** | ≈ x · sigmoid(1.702x) | (-∞, ∞) | Transformers |
| **Sigmoid** | 1/(1+e⁻ˣ) | (0, 1) | Binary output |
| **Softmax** | eˣⁱ/Σeˣʲ | (0, 1), sums to 1 | Multi-class output |
| **None (linear)** | x | (-∞, ∞) | Regression output |

**Output layer decision tree:**
```
Regression?          → No activation (linear)
Binary classification? → Sigmoid
Multi-class (pick 1)? → Softmax (or none if using CrossEntropyLoss)
```

**Remember these 3 things:**
1. Without activations, deep networks = one linear layer
2. ReLU for hidden layers, sigmoid/softmax for output — that's 95% of all cases
3. `nn.CrossEntropyLoss` already includes softmax — don't add it yourself

---

## 11. Self-Check Questions

1. What happens mathematically if you remove all activation functions from a 5-layer network?
2. Why shouldn't you use sigmoid activation in hidden layers?
3. When would you choose Leaky ReLU over standard ReLU?
4. You're building a classifier that picks one of 10 categories. What output activation do you use?
5. Why does `nn.CrossEntropyLoss` warn you not to apply softmax before passing logits to it?

<details>
<summary>Brief Answers</summary>

1. The entire network collapses into a single linear transformation — no matter how many layers you stack, the output is just a linear function of the input. You lose all ability to model non-linear patterns.

2. Sigmoid squishes values to (0,1), so gradients are always < 1. Multiplied through many layers via the chain rule (backpropagation), they shrink exponentially toward zero. Early layers stop receiving useful gradient signal and stop learning — the vanishing gradient problem.

3. When you observe "dying ReLU" — weights that output 0 for all training samples and have zero gradients, meaning they've permanently stopped learning. Leaky ReLU keeps a small slope for negative inputs (0.01×x) so gradients can still flow.

4. Softmax — it converts the raw output scores (logits) into probabilities that sum to 1.0, which is exactly what you want for a single-label multi-class problem.

5. Because `CrossEntropyLoss` applies `log(softmax(x))` internally for numerical stability. If you apply softmax first, the loss then computes `log(softmax(softmax(x)))` — double-applying it distorts the probabilities and produces wrong gradients.

</details>

---

## 12. Go Deeper

- **CS231n Lecture Notes — Neural Networks Part 1** (cs231n.github.io): The Stanford notes on activation functions are the gold standard. Includes detailed visualizations of dying ReLU and comparisons of all functions. [Why: rigorous but readable, with the math and intuition together.]

- **"Empirical Evaluation of Rectified Activations" (Xu et al., 2015)** — the paper that systematically compared ReLU, Leaky ReLU, and variants. [Why: understanding the empirical evidence helps you make principled choices, not cargo-cult ones.]

- **Andrej Karpathy — "Yes you should understand backprop"** (medium): Explains exactly how activation function choice affects gradient flow. [Why: makes the vanishing gradient problem concrete and tangible.]

- **PyTorch docs — torch.nn activation functions** (pytorch.org/docs/stable/nn.html#non-linear-activations): Complete API reference with formulas. Bookmark this — you'll check it constantly. [Why: the authoritative source for every parameter and edge case.]

- **"Gaussian Error Linear Units" paper (Hendrycks & Gimpel, 2016)** — original GELU paper. Short and readable. [Why: if you're using Transformers in production, knowing the activation they use at a deep level matters.]
