# Backpropagation

## 1. TL;DR

Backpropagation is how a neural network figures out which weights caused the error and by how much. After a forward pass produces a wrong prediction, backprop walks backwards through the network using the chain rule of calculus — computing each weight's "blame" for the error. PyTorch does this automatically with `loss.backward()`. You never write it manually, but understanding it helps you diagnose training failures.

---

## 2. The Mental Model

> 💡 **Think of it as a source-map trace for a production bug.**

You ship minified JavaScript. An error fires at `bundle.js:1:34521`. Your source map traces it back to `CartItem.tsx:42`, then to `fetchProduct()` returning null. You found the root cause by walking backwards through the stack.

- **Wrong output in production** → wrong prediction (high loss)
- **Source map** → chain rule (maps output error back to each weight)
- **Stack frames traced backwards** → layers traversed in reverse
- **Root cause function** → specific weights that contributed most to the error
- **Deploy the fix** → weight update (`optimizer.step()`)
- **CI pipeline re-running** → next training iteration

---

## 3. Why It Exists

**The problem:** A network with millions of weights needs to know exactly how to adjust *each* weight to reduce error. You can't just try random changes — there are too many.

**What came before:** Before backprop, networks were trained with finite difference approximation (perturb each weight, measure change in loss). For N weights this means N+1 forward passes per update — completely unscalable.

**What changed:** Backprop (Rumelhart, Hinton & Williams, 1986) showed you can compute all gradients in just *two* passes — one forward, one backward — using the chain rule. Training went from impossible to practical. Combined with GPUs and ReLU, it enabled the deep learning revolution.

---

## 4. Core Concepts

### Gradient

**One-line definition:** A number that tells you which direction and how much to change a weight to reduce the loss.

**Analogy:** You're hiking in fog. The gradient tells you the slope under your feet — positive slope means you're going uphill (loss increases), negative means downhill. You always want to walk downhill (reduce loss).

**Technical explanation:** For a weight `w`, the gradient `∂Loss/∂w` means "if I increase `w` by a tiny amount, how much does the loss change?" Negative gradient = increasing w reduces loss. Positive gradient = increasing w increases loss. So you move opposite the gradient.

**Common misconception:** ❌ "The gradient tells you the right value for the weight" → ✅ It only tells you the *direction* to move. You take a small step (controlled by learning rate), then recompute.

---

### The Chain Rule

**One-line definition:** To find how a far-upstream change affects a downstream output, multiply all the local effects together.

**Analogy:** Flour price affects bread cost affects sandwich cost. How much does flour affect sandwich price? Multiply the effects: (bread per flour) × (sandwich per bread).

```
flour → bread → sandwich

flour↑$1 → bread↑$0.50 → sandwich↑$0.25

Effect of flour on sandwich = 0.50 × 0.25 = 0.125
```

In neural networks:
```
weight → neuron₁ → neuron₂ → loss

∂Loss/∂weight = (∂neuron₁/∂weight) × (∂neuron₂/∂neuron₁) × (∂Loss/∂neuron₂)
```

**Common misconception:** ❌ "Backprop requires new math beyond calculus" → ✅ It's literally just the chain rule applied repeatedly — something covered in first-year calculus.

---

### Gradient Descent

**One-line definition:** The optimization algorithm that uses gradients to iteratively move weights toward lower loss.

**Analogy:** Finding the lowest point in a hilly landscape by always stepping in the steepest downhill direction.

```
new_weight = old_weight - learning_rate × gradient
```

- `learning_rate` controls step size (too big = overshoot, too small = too slow)
- You repeat this for every weight, every iteration

**Common misconception:** ❌ "Gradient descent finds the global minimum" → ✅ It finds *a* local minimum. For deep networks this is usually fine in practice, but there's no guarantee of the best possible solution.

---

### Vanishing Gradients

**One-line definition:** When gradients shrink to near-zero as they travel back through many layers, early layers stop learning.

**Analogy:** A game of telephone where each person whispers quieter. By the 10th person, the message is inaudible — the signal (gradient) is lost.

```
Layer 10 gradient: 0.5
Layer 9:  0.5 × 0.5 = 0.25
Layer 8:  0.25 × 0.5 = 0.125
...
Layer 1:  0.5¹⁰ = 0.001  ← nearly zero, layer barely updates
```

**Solutions:** ReLU (gradient = 1 for positive inputs), skip connections (ResNets), batch normalization, LSTMs.

**Common misconception:** ❌ "Vanishing gradients only happen with many layers" → ✅ They happen with sigmoid/tanh even in shallow networks, because those activations saturate and produce tiny gradients.

---

### Exploding Gradients

**One-line definition:** When gradients grow exponentially large, weights update wildly and training diverges.

**Analogy:** Compound interest gone wrong — each layer multiplies the gradient by >1, and after 10 layers you have 2¹⁰ = 1024× the original value.

**Symptoms:** Loss becomes `NaN`, weights blow up to infinity.

**Solutions:** Gradient clipping (`torch.nn.utils.clip_grad_norm_`), lower learning rate, proper weight initialization.

**Common misconception:** ❌ "Exploding gradients only happen in RNNs" → ✅ They can happen in any deep network with bad initialization or too-high learning rate.

---

### Computational Graph

**One-line definition:** The record PyTorch keeps of every operation, used to compute gradients automatically.

**Analogy:** A Git commit history — every operation is a node, every tensor is an edge. Backprop replays this graph in reverse.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2        # PyTorch records: y = x²
z = y + 3         # PyTorch records: z = y + 3

z.backward()      # walks backwards: dz/dx = 2x = 4
print(x.grad)     # tensor(4.)
```

**Common misconception:** ❌ "`requires_grad=True` makes training slow for all tensors" → ✅ Only set it for parameters (weights). Input data and labels don't need gradients.

---

## 5. How It Actually Works — Step by Step

A tiny network: `x=2 → [×w1=3] → h=6 → [×w2=0.5] → ŷ=3`, actual `y=5`.

```
FORWARD PASS:
  Step 1: h = x × w1 = 2 × 3 = 6
  Step 2: ŷ = h × w2 = 6 × 0.5 = 3
  Step 3: Loss = (y - ŷ)² = (5 - 3)² = 4

BACKWARD PASS (chain rule, right to left):
  Step 4: ∂Loss/∂ŷ = 2(ŷ - y) = 2(3 - 5) = -4

  Step 5: ∂Loss/∂w2 = ∂Loss/∂ŷ × ∂ŷ/∂w2
                    = -4 × h
                    = -4 × 6 = -24
          → w2 should INCREASE (negative gradient = go up)

  Step 6: ∂Loss/∂h = ∂Loss/∂ŷ × ∂ŷ/∂h
                   = -4 × w2
                   = -4 × 0.5 = -2

  Step 7: ∂Loss/∂w1 = ∂Loss/∂h × ∂h/∂w1
                    = -2 × x
                    = -2 × 2 = -4
          → w1 should also INCREASE

WEIGHT UPDATE (learning_rate = 0.01):
  Step 8: w2_new = 0.5 - 0.01 × (-24) = 0.74
          w1_new = 3.0 - 0.01 × (-4)  = 3.04

NEXT FORWARD PASS with updated weights:
  h = 2 × 3.04 = 6.08
  ŷ = 6.08 × 0.74 = 4.5   ← closer to 5! ✓
```

Repeat thousands of times → prediction converges to 5.

---

## 6. Code in Practice

### Minimal — Backprop on a single tensor
```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1   # y = x² + 2x + 1

y.backward()              # compute dy/dx
print(x.grad)             # dy/dx = 2x + 2 = 2(3) + 2 = 8.0
```

### Practical — Full training loop (PyTorch style)
```python
import torch
import torch.nn as nn

model = nn.Linear(3, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

x = torch.randn(32, 3)   # batch of 32 samples
y = torch.randn(32, 1)   # targets

for step in range(100):
    # 1. Forward pass
    pred = model(x)
    loss = loss_fn(pred, y)

    # 2. Backward pass — PyTorch computes ALL gradients automatically
    optimizer.zero_grad()   # clear gradients from previous step
    loss.backward()         # backpropagation (one line!)

    # 3. Update weights using computed gradients
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step}: loss={loss.item():.4f}")
```

### Gradient clipping — prevent exploding gradients
```python
optimizer.zero_grad()
loss.backward()

# Clip gradient norm to max 1.0 before updating
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

---

## 7. Gotchas & Pitfalls

| ❌ Wrong Assumption | ✅ Reality |
|---|---|
| You need to call `optimizer.zero_grad()` after `step()` | Call it **before** `backward()` — stale gradients from the last batch will corrupt the new ones |
| `loss.backward()` updates the weights | It only computes gradients. `optimizer.step()` does the actual update |
| You can call `backward()` twice on the same graph | The graph is freed after the first call. Set `retain_graph=True` only if you genuinely need it |
| Gradients accumulate automatically between batches | They DO accumulate — which is usually a bug. Always `zero_grad()` before each backward pass |
| Loss = NaN means a data problem | Usually exploding gradients. Lower learning rate or add gradient clipping |
| Backprop is slow in PyTorch | PyTorch uses C++/CUDA under the hood — it's fast. The bottleneck is usually data loading |
| You need to understand the math to use it | You use `loss.backward()` — but knowing the math helps you debug training failures |

---

## 8. When to Use / When NOT to Use

Backpropagation is not optional — it's the only practical algorithm for training neural networks. The question is about its failure modes:

**Watch out for vanishing gradients when:**
- Using sigmoid/tanh in deep hidden layers
- Training very deep networks (50+ layers) without skip connections
- RNN/LSTM on very long sequences

**Watch out for exploding gradients when:**
- Loss suddenly becomes NaN
- Weights blow up to very large values
- Training RNNs on long sequences

**Use gradient clipping when:**
- Training RNNs, LSTMs, or Transformers
- You see NaN losses or wildly oscillating loss curves

**Use `torch.no_grad()` when:**
- Running inference (validation, prediction) — no gradients needed, saves memory and compute

---

## 9. Related Concepts (The Map)

- **Activation functions** — the choice of activation directly affects gradient flow; ReLU was invented specifically to fix vanishing gradients from sigmoid (see `activation-functions.md`)
- **Loss functions** — backprop starts at the loss; the loss function determines the initial gradient `∂Loss/∂ŷ` (see `loss-functions-optimizers.md`)
- **Optimizers** — gradient descent is the simplest optimizer; Adam improves on it by adapting the step size per weight (see `loss-functions-optimizers.md`)
- **Regularization** — skip connections (ResNets) solve vanishing gradients architecturally; batch norm stabilizes gradient magnitudes (see `regularization.md`)
- **RNN/LSTM** — LSTMs were designed specifically to solve the vanishing gradient problem in sequences (see `rnn-lstm.md`)

---

## 10. Cheat Sheet

| Term | One-Line Definition |
|---|---|
| **Gradient** | Derivative of loss w.r.t. a weight — direction + magnitude to adjust |
| **Chain rule** | Multiply local gradients backward through each layer |
| **Gradient descent** | `w = w - lr × gradient`, repeated until convergence |
| **Learning rate** | Step size for weight updates (typical: 0.001 for Adam) |
| **Vanishing gradient** | Gradient shrinks to ~0 in early layers; they stop learning |
| **Exploding gradient** | Gradient grows huge; loss becomes NaN |
| **`loss.backward()`** | PyTorch call that runs backpropagation |
| **`optimizer.step()`** | Applies computed gradients to update weights |
| **`optimizer.zero_grad()`** | Clears accumulated gradients before next backward pass |

**The 3-line training core:**
```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Remember these 3 things:**
1. Backprop = chain rule applied backwards — multiply local gradients through each layer
2. PyTorch does it all with `loss.backward()` — you never implement it manually
3. Loss = NaN → exploding gradients (lower LR or clip); loss stagnates → vanishing gradients (use ReLU, batch norm, skip connections)

---

## 11. Self-Check Questions

1. What are the three PyTorch calls that form the core of every training step, and what does each do?
2. Why must you call `optimizer.zero_grad()` before `loss.backward()`?
3. Your training loss suddenly becomes NaN after epoch 3. What's most likely happening and how do you fix it?
4. A network has 10 layers with sigmoid activations. What problem will it likely have, and why?
5. What is the difference between `loss.backward()` and `optimizer.step()`?

<details>
<summary>Brief Answers</summary>

1. **`optimizer.zero_grad()`** — clears gradients accumulated from the previous batch. **`loss.backward()`** — runs backpropagation, computing the gradient of the loss with respect to every parameter. **`optimizer.step()`** — uses those computed gradients to update each weight.

2. PyTorch **accumulates** gradients by default (adds to existing `.grad` values). Without zeroing, the gradients from batch 1, batch 2, batch 3... all pile up, making each update wrong. You must zero them before each new backward pass to get clean per-batch gradients.

3. Most likely **exploding gradients** — weights are updating so drastically they blow up to infinity. Fixes: (a) lower the learning rate, (b) add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` before `optimizer.step()`, (c) check for very large values in your input data.

4. **Vanishing gradients** — sigmoid squashes all values to (0,1), so its gradient is always < 0.25. Multiplied through 10 layers via the chain rule: 0.25¹⁰ ≈ 0.000001. Early layers receive essentially zero gradient and stop learning entirely.

5. `loss.backward()` **computes** gradients and stores them in each parameter's `.grad` attribute — it doesn't change any weights. `optimizer.step()` **reads** those `.grad` values and applies the update rule (`w = w - lr × grad`) to change the weights. They must happen in sequence.

</details>

---

## 12. Go Deeper

- **Andrej Karpathy — "micrograd"** (github.com/karpathy/micrograd): A backpropagation engine in ~100 lines of Python. Building this makes the chain rule click permanently. Essential reading. [Why: nothing beats implementing it yourself once.]

- **Andrej Karpathy — "Yes you should understand backprop"** (Medium, 2016): A practitioner's argument for why understanding backprop matters even when PyTorch handles it. Includes sigmoid saturation and dead ReLU worked examples. [Why: directly relevant to debugging real training failures.]

- **3Blue1Brown — "Backpropagation calculus"** (YouTube): The visual explanation using the chain rule on a tiny network. Watch after reading this doc to solidify the math. [Why: the visual of gradients flowing backwards is unforgettable.]

- **CS231n Notes — "Backpropagation, Intuitions"** (cs231n.github.io/optimization-2): Stanford's written explanation with gate-level intuition (add gates, multiply gates, max gates). Rigorous and practical. [Why: explains backprop in terms of circuit gates — a great mental model.]

- **PyTorch Autograd tutorial** (pytorch.org/tutorials/beginner/blitz/autograd_tutorial): Official explanation of `requires_grad`, the computation graph, and how `.backward()` works internally. [Why: knowing the tool at this level prevents subtle bugs in custom training loops.]
