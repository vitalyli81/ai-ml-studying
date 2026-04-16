# Backpropagation

## What Is It?

Backpropagation is how a neural network **figures out which weights caused the error** and how to fix them. It works backwards — from the output error back through each layer — calculating how much each weight contributed to the mistake.

Think of it like debugging: your app shows a wrong result (the error), and you trace backwards through the code to find which function (which weight) caused the bug.

## Frontend Analogy — Source Maps

```
Minified error in production:    "TypeError at bundle.js:1:34521"
                                          ↓ source map
Original source:                 "TypeError at CartItem.tsx:42"
                                          ↓ trace back
Root cause:                      "price is null in fetchProduct()"
```

Backpropagation does the same thing:
1. You see the error at the **output** (prediction was wrong)
2. Trace back through **each layer** (which neurons contributed?)
3. Find the **root cause** (which weights need changing?)

## The Chain Rule (The Math Behind It)

You don't need to memorize formulas, but the core idea is simple.

### The Chain Rule in Everyday Terms

If you change the **price of flour** → it changes the **cost of bread** → which changes the **cost of a sandwich**.

How much does flour price affect sandwich cost? **Multiply the effects through each step.**

```
flour → bread → sandwich

If flour goes up $1 → bread goes up $0.50 → sandwich goes up $0.25

Effect of flour on sandwich = 0.50 × 0.25 = 0.125 (per dollar of flour)
```

### In Neural Network Terms

```
weight₁ → neuron₁ → neuron₂ → loss

How much does weight₁ affect the loss?
= (effect of weight₁ on neuron₁)
× (effect of neuron₁ on neuron₂)
× (effect of neuron₂ on loss)
```

That's the **chain rule**: multiply the local effects through each step.

## Step by Step Example

### A Tiny Network

```
x=2 →[×w1=3]→ h=6 →[×w2=0.5]→ y_pred=3    (actual y=5)
```

**Forward pass:** 2 × 3 = 6, then 6 × 0.5 = 3. Predicted 3, actual is 5.

**Loss:** (5 - 3)² = 4

**Backward pass — find how each weight affects the loss:**

```
How much does w2 affect the loss?
  ∂Loss/∂w2 = ∂Loss/∂y_pred × ∂y_pred/∂w2
             = 2×(y_pred - y) × h
             = 2×(3 - 5) × 6
             = -24
  → w2 should INCREASE (negative gradient = go up)

How much does w1 affect the loss?
  ∂Loss/∂w1 = ∂Loss/∂y_pred × ∂y_pred/∂h × ∂h/∂w1
             = 2×(3 - 5) × w2 × x
             = 2×(-2) × 0.5 × 2
             = -4
  → w1 should also INCREASE
```

**Update weights** (learning rate = 0.01):
```
w2_new = 0.5 - 0.01 × (-24) = 0.5 + 0.24 = 0.74
w1_new = 3.0 - 0.01 × (-4)  = 3.0 + 0.04 = 3.04
```

Now the prediction will be closer to 5. Repeat this thousands of times.

## The Full Picture

```
FORWARD PASS (left → right):
Input → Layer 1 → Layer 2 → Layer 3 → Prediction → Loss
  2       6         3         3                      4

BACKWARD PASS (right → left):
Input ← Layer 1 ← Layer 2 ← Layer 3 ← Prediction ← Loss
         ∂L/∂w1    ∂L/∂w2    ∂L/∂w3                  ∂L/∂pred

UPDATE:
w1 = w1 - lr × ∂L/∂w1
w2 = w2 - lr × ∂L/∂w2
w3 = w3 - lr × ∂L/∂w3
```

## Why "Back" Propagation?

Because we go **backwards** through the network:

```
Forward:  Input → → → → → → Prediction
                                  ↓
                              Calculate Error
                                  ↓
Backward: Update ← ← ← ← ← Gradients flow back
```

We can't go forward because we don't know the gradients until we see the error at the end.

## The Vanishing Gradient Problem

In deep networks, gradients get **multiplied at each layer** (chain rule). If the multipliers are small (< 1), the gradient shrinks exponentially:

```
Layer 10 gradient: 0.5
Layer 9 gradient:  0.5 × 0.5 = 0.25
Layer 8 gradient:  0.25 × 0.5 = 0.125
...
Layer 1 gradient:  0.5^10 = 0.001  ← almost zero! Layer doesn't learn.
```

**Early layers barely update** → the network can't learn deep patterns.

### Solutions:

| Solution | How It Helps |
|----------|-------------|
| **ReLU activation** | Gradient is 1 (not < 1) for positive values |
| **Batch normalization** | Keeps values in a good range at each layer |
| **Residual connections (skip connections)** | Gradient can skip layers (used in ResNet, Transformers) |
| **LSTM/GRU** | Special gates that preserve gradients (for sequences) |

## Exploding Gradient Problem

The opposite — gradients get too big and weights blow up to infinity:

```
Layer 1 gradient: 2
Layer 2 gradient: 2 × 2 = 4
Layer 3 gradient: 4 × 2 = 8
...
Layer 10 gradient: 2^10 = 1024  ← way too big!
```

### Solutions:

| Solution | How It Helps |
|----------|-------------|
| **Gradient clipping** | Cap gradients at a maximum value |
| **Lower learning rate** | Smaller updates = more stable |
| **Proper weight initialization** | Start with sensible weight values |

## In PyTorch (You Never Write Backprop Manually)

```python
import torch
import torch.nn as nn

# Simple network
model = nn.Linear(3, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
x = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([10.0])

# 1. Forward pass
y_pred = model(x)
loss = loss_fn(y_pred, y_true)

# 2. Backward pass (PyTorch does ALL the chain rule math for you)
loss.backward()  # ← this is backpropagation, one line!

# 3. Update weights
optimizer.step()  # ← this adjusts weights using the computed gradients

# 4. Reset gradients for next iteration
optimizer.zero_grad()
```

**You never implement backprop yourself.** PyTorch's `loss.backward()` does it all. But understanding how it works helps you debug when training goes wrong (vanishing gradients, NaN losses, etc.).

## Common Mistakes That Break Training

| Symptom | Likely Cause |
|---------|-------------|
| Loss = NaN | Exploding gradients. Lower learning rate or add gradient clipping. |
| Loss doesn't decrease | Vanishing gradients, learning rate too low, or bug in data pipeline. |
| Loss oscillates wildly | Learning rate too high. Reduce it. |
| Loss decreases then increases | Overfitting. Add regularization or get more data. |

## Key Takeaway

Backpropagation is just the **chain rule applied backwards through the network** to figure out how each weight affected the error. You never code it manually (PyTorch handles it), but understanding it helps you diagnose training issues. The two big problems — vanishing and exploding gradients — are solved by modern architectures (ReLU, batch norm, skip connections).
