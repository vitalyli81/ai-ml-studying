# Loss Functions & Optimizers

## 1. TL;DR

Loss functions measure how wrong your model's predictions are. Optimizers use that error signal to update weights in the right direction. Think of them as a pair: the loss says "you're this wrong," the optimizer says "here's how to fix it." Use MSE for regression, CrossEntropy for classification. Use AdamW as your default optimizer with `lr=0.001`. That covers 90% of all cases.

---

## 2. The Mental Model

> 💡 **Think of a GPS navigation system.**

You're driving to a destination (the global minimum of the loss). The GPS constantly recalculates your position (loss function — how far off are you?), then tells you which turns to make (optimizer — which direction to update weights). Different GPS routing algorithms (optimizers) use different strategies: some go straight-line, some adapt to traffic, some remember your past routes.

- **Destination** → minimum possible loss (perfect predictions)
- **Current position** → current model weights
- **Distance from destination** → loss value
- **GPS recalculating** → computing the gradient
- **Turn-by-turn directions** → gradient descent step
- **Adaptive routing (Google Maps)** → Adam optimizer (adapts per weight)
- **Simple routing (paper map)** → SGD (one fixed step size for all weights)

---

## 3. Why It Exists

**The problem:** Training needs a single number that measures "how wrong is this model?" — a scalar you can take a derivative of to know which direction to adjust weights.

**What came before:** Mean Squared Error was used even before neural networks (in linear regression). The key insight for classification was that MSE doesn't penalize confident wrong answers nearly enough — cross-entropy does.

**What changed:** The right loss function is critical. Cross-entropy loss (paired with softmax) made classification networks trainable. Adam (2014) made training much faster and more reliable than plain SGD, dramatically lowering the barrier to training new models.

---

## 4. Core Concepts

### MSE (Mean Squared Error)

**One-line definition:** Average of squared differences between predictions and true values — penalizes large errors heavily.

**Analogy:** A golf scorecard where every stroke over par is counted squared — a double bogey (2 over) is penalized 4× more than a bogey (1 over). Big mistakes hurt disproportionately.

**Technical explanation:**
```
MSE = (1/n) × Σ(actual - predicted)²

Actual:    [100, 200, 300]
Predicted: [110, 190, 280]
Errors:    [ 10, -10,  -20]
Squared:   [100,  100,  400]
MSE = (100 + 100 + 400) / 3 = 200
```

**Common misconception:** ❌ "MSE works for classification too" → ✅ MSE doesn't properly penalize confident wrong classifications. A model predicting 0.99 confidence for the wrong class barely gets penalized by MSE — cross-entropy destroys it.

---

### Cross-Entropy Loss

**One-line definition:** Measures how wrong your *probability distribution* is — punishes confident wrong predictions most severely.

**Analogy:** A quiz where being confidently wrong costs you 10× more than being unsure and wrong. If you say "I'm 99% sure it's A" and it's B, you lose almost everything.

**Technical explanation:**
```
Binary CE:  Loss = -(y × log(p) + (1-y) × log(1-p))

  y=1 (spam),  p=0.9  → Loss = -log(0.9)  = 0.10  (small, correct direction)
  y=1 (spam),  p=0.1  → Loss = -log(0.1)  = 2.30  (large, badly wrong)
  y=1 (spam),  p=0.01 → Loss = -log(0.01) = 4.60  (HUGE, confidently wrong)
```

The log function is the key — as predicted probability approaches 0 for the correct class, loss approaches infinity.

**Common misconception:** ❌ "Add softmax before `nn.CrossEntropyLoss`" → ✅ PyTorch's `CrossEntropyLoss` applies log-softmax internally. Passing raw logits is correct and more numerically stable.

---

### MAE (Mean Absolute Error / L1 Loss)

**One-line definition:** Average of absolute differences — treats all error magnitudes proportionally (no squaring).

**Analogy:** A cab meter that charges exactly by distance, not distance-squared. A 10-mile trip costs exactly 2× a 5-mile trip.

```
MAE = (1/n) × Σ|actual - predicted|

vs MSE: a 10× error is penalized 10× more (not 100×)
→ MAE is robust to outliers; MSE amplifies them
```

**Common misconception:** ❌ "MAE is always safer than MSE" → ✅ MAE has zero gradient when prediction = actual (kink at 0), which can cause optimization issues. MSE has smooth gradients everywhere.

---

### Gradient Descent

**One-line definition:** The core optimization loop — move weights opposite the gradient direction by a small step.

**Analogy:** Finding the lowest point in a hilly landscape by always stepping in the steepest downhill direction.

```
new_weight = old_weight - learning_rate × gradient
```

**Common misconception:** ❌ "Gradient descent finds the global minimum" → ✅ It finds *a* local minimum. For deep networks, local minima are usually good enough in practice.

---

### Learning Rate

**One-line definition:** Controls how big each weight update step is — the single most important hyperparameter.

**Analogy:** Step size while hiking downhill in fog. Too large: you overshoot the valley and climb the other side. Too small: you'll be hiking for days.

```
Too high  (0.1):    Loss bounces wildly, never converges
Too low   (0.00001): Converges painfully slowly
Just right (0.001): Smooth, steady decrease

Loss
│╲
│ ╲╱╲       ← too high (oscillating)
│
│╲
│  ╲___      ← just right
│      ╲___
│
│╲__________  ← too low (barely moving)
└──────────── Epochs
```

**Common misconception:** ❌ "The right learning rate is fixed for all models" → ✅ It depends on the model, dataset, and optimizer. Always tune it. Start with `1e-3` for Adam, `1e-2` for SGD.

---

### SGD (Stochastic Gradient Descent)

**One-line definition:** The simplest optimizer — one fixed step per gradient, applied to random batches.

**Analogy:** Walking straight downhill with equal-length steps regardless of terrain. Simple, but can bounce around in valleys.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Momentum (0.9) adds "velocity" — like a ball rolling downhill, it builds speed in consistent directions and rolls through small bumps.

**Common misconception:** ❌ "Adam is always better than SGD" → ✅ SGD + momentum often achieves *better final accuracy* for image classification with careful tuning. Adam gets you 95% of the way with zero effort.

---

### Adam / AdamW

**One-line definition:** An adaptive optimizer that gives each weight its own auto-adjusted learning rate based on gradient history.

**Analogy:** Google Maps routing — it adapts to traffic conditions (gradient history) per road (per weight), not just following a fixed speed limit everywhere.

**Technical explanation:** Adam tracks two things per weight:
- **Momentum** (m): running average of past gradients (direction)
- **Variance** (v): running average of squared gradients (magnitude)

Effective learning rate = `lr / (sqrt(v) + ε)` — weights with large historical gradients get smaller steps (stable), weights with small gradients get larger steps (faster learning).

**AdamW** fixes a subtle weight decay bug in Adam. Use AdamW over Adam — it's strictly better and what GPT, BERT, and LLaMA use.

**Common misconception:** ❌ "AdamW's weight_decay is the same as L2 regularization in Adam" → ✅ In Adam, weight decay is incorrectly absorbed into the adaptive learning rate scaling. AdamW decouples them — that's the fix.

---

## 5. How It Actually Works — Step by Step

Training a binary spam classifier:

```
Step 1: FORWARD PASS
  Input email → model → raw score (logit): 2.3
  BCEWithLogitsLoss applies sigmoid: σ(2.3) = 0.91  → "91% spam"

Step 2: COMPUTE LOSS
  True label: y = 1 (is spam)
  Loss = -log(0.91) = 0.094  (small — model is correctly confident)

  Counter-example: if model output 0.09 (9% spam) for actual spam:
  Loss = -log(0.09) = 2.41   (large — model is confidently wrong)

Step 3: BACKWARD PASS
  loss.backward() — computes ∂Loss/∂w for every weight

Step 4: OPTIMIZER STEP (AdamW)
  For each weight w:
    m = 0.9 × m + 0.1 × gradient      (momentum update)
    v = 0.999 × v + 0.001 × gradient² (variance update)
    effective_lr = lr / sqrt(v)        (adaptive step)
    w = w - effective_lr × m           (update)
    w = w × (1 - weight_decay)         (weight decay — AdamW's fix)

Step 5: REPEAT
  Each batch: loss decreases, model becomes more accurate
```

---

## 6. Code in Practice

### Minimal — Loss functions
```python
import torch
import torch.nn as nn

pred = torch.tensor([2.5, 1.0, 0.5])
target = torch.tensor([3.0, 1.0, 0.0])

mse = nn.MSELoss()(pred, target)     # 0.1667
mae = nn.L1Loss()(pred, target)      # 0.3333
print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

# Classification — pass raw logits (no sigmoid/softmax)
logits = torch.tensor([[2.0, 0.5, -1.0]])   # 3-class, batch=1
labels = torch.tensor([0])                   # correct class = 0
ce = nn.CrossEntropyLoss()(logits, labels)
print(f"Cross-entropy: {ce:.4f}")
```

### Practical — Full training loop with AdamW
```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64), nn.ReLU(),
    nn.Linear(64, 3),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

X = torch.randn(100, 10)
y = torch.randint(0, 3, (100,))

for epoch in range(50):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: {loss.item():.4f}")
```

### Real-world — Learning rate scheduler
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Cosine annealing: smoothly decay LR from 1e-3 → ~0 over training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    train_one_epoch(model, optimizer)
    scheduler.step()   # update LR after each epoch
    print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()[0]:.6f}")
```

---

## 7. Gotchas & Pitfalls

| ❌ Wrong Assumption | ✅ Reality |
|---|---|
| Apply softmax before `CrossEntropyLoss` | It already includes log-softmax — applying softmax first gives wrong gradients |
| Use MSE for classification | MSE doesn't penalize confident wrong answers enough — use CrossEntropy |
| Higher learning rate = faster training | Too high and training diverges; the loss oscillates or becomes NaN |
| AdamW weight_decay=0 is fine | Without weight decay, large weights accumulate and the model overfits |
| The loss going down means the model is good | Monitor validation loss too — training loss can drop while val loss rises (overfitting) |
| `optimizer.zero_grad()` after `step()` | Call it BEFORE `backward()` — stale gradients corrupt the current batch's update |
| SGD is obsolete | For image classification, SGD + momentum with tuned LR often beats Adam on final accuracy |

---

## 8. When to Use / When NOT to Use

**MSELoss when:**
- Predicting a continuous number (price, temperature, score)
- Your outputs should be any real value

**L1Loss (MAE) when:**
- Predicting continuous values but data has outliers that would distort MSE
- You want predictions less sensitive to extreme values

**BCEWithLogitsLoss when:**
- Binary classification (yes/no, spam/not-spam)
- Always pass raw logits, not sigmoid output

**CrossEntropyLoss when:**
- Multi-class classification (pick one of N)
- Always pass raw logits

**AdamW when:**
- Default for everything — transformers, MLPs, CNNs when getting started
- Fine-tuning pretrained models (use very small lr: 1e-5 to 5e-5)

**SGD + Momentum when:**
- Training CNNs for image classification from scratch (can beat Adam with tuning)
- You need deterministic, reproducible optimization

---

## 9. Related Concepts (The Map)

- **Backpropagation** — the optimizer uses gradients computed by backprop; the loss function determines the starting gradient `∂Loss/∂output` (see `backpropagation.md`)
- **Activation functions** — loss function + output activation are a pair: sigmoid + BCE, softmax (internal to CrossEntropy) + CE, linear + MSE (see `activation-functions.md`)
- **Regularization** — weight decay (in AdamW) is a form of regularization; early stopping monitors the loss curve (see `regularization.md`)
- **Transfer learning** — fine-tuning uses the same loss functions but requires much smaller learning rates (1e-5 vs 1e-3) (see `transfer-learning.md`)
- **Training loop** — loss functions and optimizers are the core of PyTorch's training loop (see `pytorch-basics.md`)

---

## 10. Cheat Sheet

| Task | Loss Function | PyTorch |
|---|---|---|
| Predict a number | MSELoss | `nn.MSELoss()` |
| Predict a number (outliers) | L1Loss | `nn.L1Loss()` |
| Yes/No classification | BCEWithLogitsLoss | `nn.BCEWithLogitsLoss()` |
| Pick one of N classes | CrossEntropyLoss | `nn.CrossEntropyLoss()` |

| Situation | Optimizer | Learning Rate |
|---|---|---|
| Default / first try | AdamW | 1e-3 |
| Fine-tuning pretrained | AdamW | 1e-5 to 5e-5 |
| Image classification (best accuracy) | SGD + Momentum | 0.01–0.1 |
| Transformers | AdamW | 3e-4 |

**The minimal training step:**
```python
optimizer.zero_grad()
loss = loss_fn(model(X), y)
loss.backward()
optimizer.step()
```

**Remember these 3 things:**
1. Loss function = task-specific (regression → MSE, classification → CrossEntropy)
2. AdamW is your default optimizer — it works well out of the box with `lr=1e-3`
3. Learning rate is the most important hyperparameter — too high = NaN, too low = forever

---

## 11. Self-Check Questions

1. Why should you NOT apply softmax before passing logits to `nn.CrossEntropyLoss`?
2. Why is cross-entropy preferred over MSE for classification tasks?
3. What does AdamW's `weight_decay` parameter do, and why use AdamW instead of Adam?
4. Your training loss is decreasing smoothly, but validation loss starts increasing at epoch 10. What's happening?
5. You're training a regression model that predicts house prices. Prices range from $50K to $5M, with a few extreme values at $20M. Should you use MSE or MAE? Why?

<details>
<summary>Brief Answers</summary>

1. `nn.CrossEntropyLoss` internally applies `log(softmax(logits))` in a numerically stable combined operation. If you apply softmax first, it computes `log(softmax(softmax(logits)))` — the double softmax squashes the distribution, making all values close to 1/N and producing nearly flat gradients. Training stalls.

2. Cross-entropy uses the log function, which creates an asymmetric penalty: being confidently wrong (predicting 0.99 for the wrong class) gives loss ≈ 4.6, while being uncertain (predicting 0.6) gives loss ≈ 0.5. MSE treats them nearly the same (0.99² ≈ 0.98 vs 0.60² = 0.36). Cross-entropy correctly trains the model to be confident AND accurate.

3. `weight_decay` adds a penalty proportional to the magnitude of weights, shrinking them toward zero each step. This prevents weights from growing large and overfitting. AdamW vs Adam: in Adam, weight decay interacts incorrectly with the adaptive learning rate scaling (it gets divided by the gradient variance, making it weaker for frequent parameters). AdamW applies weight decay separately after the adaptive update, making it work as intended.

4. **Overfitting** — the model has memorized the training data and stopped generalizing. The training loss is optimizing for patterns specific to training examples; validation loss reflects performance on unseen data. Fix: add regularization (dropout, stronger weight decay), reduce model capacity, get more data, or use early stopping.

5. **MAE (L1Loss)**. The extreme outliers ($20M properties) would dominate MSE — their squared error is enormous, causing the model to overfit its predictions toward those extremes. MAE penalizes outliers linearly, not quadratically, making the model more robust to the skewed distribution of house prices.

</details>

---

## 12. Go Deeper

- **"Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)**: The original Adam paper — short, readable, explains the momentum + variance idea clearly. [Why: understanding the algorithm at paper-level helps you tune it and interpret optimizer behavior.]

- **"Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)**: The AdamW paper — explains exactly why Adam's weight decay is broken and how AdamW fixes it. [Why: you'll use AdamW constantly; knowing why it's better than Adam is useful.]

- **fast.ai — "Optimizers" lecture**: Jeremy Howard's practical explanation of optimizers from the practitioner's lens — which to use when, learning rate finders, 1-cycle policy. [Why: most practical optimizer advice you'll find anywhere.]

- **Andrej Karpathy — "A Recipe for Training Neural Networks"** (karpathy.github.io): A checklist for debugging training runs, including loss curve interpretation and optimizer choices. [Why: when your loss does something unexpected, this is your first reference.]

- **PyTorch lr_scheduler docs** (pytorch.org/docs/stable/optim.html): Complete reference for all schedulers — StepLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau. [Why: learning rate scheduling is an easy win; bookmark this for the exact API signatures.]
