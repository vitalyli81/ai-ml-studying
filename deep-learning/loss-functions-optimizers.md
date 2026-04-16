# Loss Functions & Optimizers

## Two Questions Every Neural Network Answers

1. **Loss function** → "How wrong am I?" (measures the error)
2. **Optimizer** → "How should I fix it?" (updates the weights)

They work together: the loss function tells you the **score**, the optimizer tells you the **direction to improve**.

---

# Part 1: Loss Functions

## Frontend Analogy

A loss function is like a **Lighthouse score** for your model:
- Score = 100 → perfect (loss = 0)
- Score = 50 → needs improvement (loss is high)
- You optimize by making changes that improve the score

## The Three Loss Functions You Need

### 1. MSE (Mean Squared Error) — For Regression

"How far off are my number predictions?"

```
MSE = average of (actual - predicted)²

Actual:    [100, 200, 300]
Predicted: [110, 190, 280]
Errors:    [ 10, -10, -20]
Squared:   [100, 100, 400]
MSE = (100 + 100 + 400) / 3 = 200
```

- Squares the errors → **big mistakes are penalized much more**
- Predicted 110 vs 100 → error 100. Predicted 140 vs 100 → error 1600 (16x worse, not 4x)
- Use when: predicting prices, temperatures, scores — any **continuous number**

```python
loss_fn = nn.MSELoss()
```

### 2. Cross-Entropy Loss — For Classification

"How confident was I in the wrong answer?"

```
Binary:     actual = 1 (spam),  predicted probability = 0.9 → small loss
            actual = 1 (spam),  predicted probability = 0.1 → HUGE loss
            actual = 0 (not spam), predicted = 0.1 → small loss

Multi-class: actual = "cat", predicted = [cat: 0.8, dog: 0.1, bird: 0.1] → small loss
             actual = "cat", predicted = [cat: 0.1, dog: 0.7, bird: 0.2] → HUGE loss
```

**Why not just use MSE for classification?**
MSE doesn't care about confidence properly. Cross-entropy **destroys** confident wrong answers.

```
Confident and wrong: predicted 0.99 for wrong class → loss ≈ 4.6 (massive!)
Unsure and wrong:    predicted 0.60 for wrong class → loss ≈ 0.5 (moderate)
```

```python
# Binary classification (spam or not)
loss_fn = nn.BCEWithLogitsLoss()

# Multi-class classification (cat/dog/bird)
loss_fn = nn.CrossEntropyLoss()  # includes softmax internally!
```

### 3. MAE (Mean Absolute Error) — Robust Regression

```
MAE = average of |actual - predicted|

Same as MSE but without squaring → outliers don't dominate
```

Use MAE when your data has **outliers** that would distort MSE.

## Cheat Sheet — Which Loss Function?

| Task | Loss Function | PyTorch |
|------|--------------|---------|
| Predict a **number** | MSELoss | `nn.MSELoss()` |
| Predict a **number** (with outliers) | L1Loss (MAE) | `nn.L1Loss()` |
| **Yes/No** classification | BCEWithLogitsLoss | `nn.BCEWithLogitsLoss()` |
| **Pick one of N** classes | CrossEntropyLoss | `nn.CrossEntropyLoss()` |

---

# Part 2: Optimizers

## Frontend Analogy

An optimizer is like choosing how to navigate down a hill:
- **SGD** → walk straight downhill, constant step size (simple but can get stuck)
- **SGD + Momentum** → roll a ball downhill (builds speed, carries through small bumps)
- **Adam** → GPS navigation that adapts speed to terrain (the smart default)

## How Optimization Works

All optimizers do the same core thing:

```
new_weight = old_weight - learning_rate × gradient
```

The gradient says "go this direction," the learning rate says "go this far." Different optimizers are just smarter about **how far** and **in what direction**.

## The Optimizers You Need to Know

### 1. SGD (Stochastic Gradient Descent) — The Classic

```
weight = weight - lr × gradient
```

Simple: move in the steepest downhill direction.

**Problem:** Can oscillate back and forth and get stuck in local minima.

**SGD + Momentum:** Adds "velocity" — remembers the previous direction:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

Think of it like a ball rolling downhill — it builds speed in consistent directions and rolls through small bumps.

### 2. Adam (Adaptive Moment Estimation) — The Default Choice

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Adam combines two ideas:
- **Momentum** — remembers which direction it's been going
- **Adaptive learning rate** — each weight gets its own learning rate that auto-adjusts

**Why it's the default:**
- Works well out of the box with minimal tuning
- Handles different feature scales automatically
- Converges faster than plain SGD

**When NOT to use Adam:**
- When you need the absolute best final accuracy (SGD + momentum with careful tuning often wins for image classification)
- But Adam gets you 95% of the way with zero effort

### 3. AdamW — Adam but Better

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

Fixes a subtle bug in Adam's weight decay. **Use AdamW over Adam** — it's strictly better. This is what most modern models use (GPT, BERT, etc.).

## Learning Rate — The Most Important Hyperparameter

```
Too high (0.1):     Loss bounces around wildly, never converges
Too low (0.00001):  Loss decreases painfully slowly
Just right (0.001): Smooth, steady decrease

Loss
  |╲
  | ╲  ╱╲        ← too high (oscillating)
  |  ╲╱  ╲╱
  |
  |╲
  |  ╲___         ← just right (smooth convergence)
  |      ╲___
  |
  |
  |╲___________   ← too low (barely moving)
  |________________ Epochs
```

**Rules of thumb:**
- **Adam/AdamW:** Start with `lr=0.001` or `lr=3e-4`
- **SGD:** Start with `lr=0.01` or `lr=0.1`
- **Fine-tuning pretrained models:** Use much smaller `lr=1e-5` to `lr=5e-5`

## Learning Rate Schedulers — Change LR During Training

Start with a higher LR (explore broadly), then decrease it (fine-tune):

```python
# Step: multiply LR by 0.1 every 30 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine: smoothly decrease LR following a cosine curve
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# In training loop:
for epoch in range(100):
    train_one_epoch()
    scheduler.step()  # update learning rate
```

## Complete Training Loop

Everything together:

```python
import torch
import torch.nn as nn

# Model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
)

# Loss + Optimizer
loss_fn = nn.CrossEntropyLoss()                      # multi-class classification
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    predictions = model(X_train)
    loss = loss_fn(predictions, y_train)

    # Backward pass
    optimizer.zero_grad()   # reset gradients from last step
    loss.backward()         # compute gradients (backpropagation)
    optimizer.step()        # update weights

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

## Cheat Sheet — Which Optimizer?

| Situation | Optimizer | Learning Rate |
|-----------|-----------|---------------|
| **Default / first try** | AdamW | 0.001 |
| **Fine-tuning pretrained model** | AdamW | 1e-5 to 5e-5 |
| **Image classification (best accuracy)** | SGD + Momentum | 0.01-0.1 |
| **Transformers / NLP** | AdamW | 3e-4 to 1e-3 |

## Key Takeaway

The loss function measures **how wrong** the model is (MSE for numbers, cross-entropy for categories). The optimizer decides **how to fix it** (Adam/AdamW is the safe default). The learning rate is the **single most important setting** — too high and training explodes, too low and it never converges. Start with AdamW + lr=0.001 and adjust from there.
