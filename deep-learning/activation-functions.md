# Activation Functions

## What Is It?

An activation function decides **whether a neuron should "fire" or not** — it adds non-linearity so the network can learn complex patterns, not just straight lines.

Without activation functions, stacking 100 layers would still be equivalent to 1 layer of linear math. Activations are what make deep learning *deep*.

## Frontend Analogy

Activation functions are like **CSS transitions/easing functions** — they transform a linear input into a shaped output:

```javascript
// linear:      output = input             (boring straight line)
// ease-in:     output = input²            (slow start, fast end)
// ease-in-out: output = smooth S-curve    (that's basically sigmoid!)
// step:        output = input > 0 ? 1 : 0 (that's a step function)
```

## The Big Three (You'll Use These 95% of the Time)

### 1. ReLU (Rectified Linear Unit) — The Default Choice

```
Output = max(0, x)

If input is negative → output 0 (neuron is "off")
If input is positive → output the input unchanged (neuron is "on")
```

```
Output
  |          /
  |        /
  |      /
  |    /
  |  /
  |/__________ Input
  0
 (negative = 0, positive = unchanged)
```

**Why it's popular:**
- Super fast to compute (just a max operation)
- Doesn't have vanishing gradient problem for positive values
- Works great in practice

**The problem — "Dying ReLU":**
If a neuron's input is always negative, it **always outputs 0** and never updates (gradient = 0). It's permanently dead.

**Fix → Leaky ReLU:**
```
Output = max(0.01 * x, x)

Negative inputs get a tiny slope (0.01) instead of hard 0
```

### 2. Sigmoid — For Probabilities (0 to 1)

```
Output = 1 / (1 + e^(-x))

Any input → output between 0 and 1
```

```
Output
 1|           ___________
  |         /
  |       /
0.5|     /     ← S-shaped curve
  |   /
  | /
 0|/___________  Input
```

**Use for:** Output layer when you need a probability (binary classification).

**Don't use in:** Hidden layers. Causes vanishing gradient (output is always between 0 and 1, so gradients shrink when multiplied).

### 3. Softmax — For Multiple Classes

```
Converts a vector of numbers into probabilities that sum to 1

Input:  [2.0, 1.0, 0.1]
Output: [0.65, 0.24, 0.11]  ← sums to 1.0
```

**Use for:** Output layer for multi-class classification (pick one of N classes).

```python
# "Is this image a cat, dog, or bird?"
# Softmax output: [0.85, 0.10, 0.05]
#                  cat   dog   bird → predict CAT (85%)
```

## Quick Reference — When to Use What

### Hidden Layers (inside the network):

| Function | When | Note |
|----------|------|------|
| **ReLU** | Default for everything | Start here, switch only if needed |
| **Leaky ReLU** | If you see dying neurons | Safe default alternative |
| **GELU** | Transformers / NLP | Used in BERT, GPT — smoother ReLU |
| **SiLU / Swish** | Modern architectures | Used in newer models |

### Output Layer (final layer):

| Task | Function | Output |
|------|----------|--------|
| **Regression** (predict a number) | None (linear) | Any value |
| **Binary classification** (yes/no) | Sigmoid | 0 to 1 |
| **Multi-class** (pick one of N) | Softmax | N probabilities summing to 1 |

## All Functions at a Glance

```
ReLU:         ____/        Range: [0, ∞)     Hidden layers default
Leaky ReLU:  __/⟋         Range: (-∞, ∞)    Fixes dying ReLU
Sigmoid:     __/‾‾         Range: (0, 1)     Binary output
Tanh:        _/‾‾          Range: (-1, 1)    Centered sigmoid
Softmax:     [0.7, 0.2, 0.1]                 Multi-class output
```

## GELU (Gaussian Error Linear Unit) — The Transformer Favorite

```
GELU ≈ x × sigmoid(1.702 × x)
```

It's like ReLU but **smoother** — instead of a hard cutoff at 0, there's a smooth curve. Small negative values can still pass through a little bit.

```
ReLU:   ______/     (sharp corner at 0)
GELU:   ____⟋/     (smooth curve near 0)
```

Used in GPT, BERT, and most modern transformer models. You'll see it a lot.

## Common Mistakes

| Mistake | Why It's Bad | Fix |
|---------|-------------|-----|
| Sigmoid in hidden layers | Vanishing gradients, slow training | Use ReLU |
| No activation on output for classification | Model outputs raw numbers, not probabilities | Add sigmoid/softmax |
| Sigmoid for multi-class | Only works for 2 classes | Use softmax |
| ReLU on output for regression | Clips negative predictions to 0 | Use no activation (linear) |

## PyTorch Examples

```python
import torch
import torch.nn as nn

# Individual activation functions
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=-1)

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

print(f"Input:   {x.tolist()}")
print(f"ReLU:    {relu(x).tolist()}")       # [0, 0, 0, 1, 2]
print(f"Sigmoid: {sigmoid(x).tolist()}")     # [0.12, 0.27, 0.5, 0.73, 0.88]

# In a network — typical pattern
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),          # hidden layer → ReLU
            nn.Linear(64, 32),
            nn.ReLU(),          # hidden layer → ReLU
            nn.Linear(32, 3),   # output layer → 3 classes
            # No softmax here! PyTorch's CrossEntropyLoss includes it.
        )

    def forward(self, x):
        return self.net(x)
```

**Important PyTorch note:** `nn.CrossEntropyLoss` already applies softmax internally. Don't add softmax to your output layer if using this loss — you'll apply it twice!

## Key Takeaway

Activation functions add **non-linearity** so neural networks can learn complex patterns. Use **ReLU for hidden layers** (it's fast and works), **sigmoid for binary output**, and **softmax for multi-class output**. That covers 95% of cases. The only decision you really need to make is what goes on the output layer — and that's determined by your task type.
