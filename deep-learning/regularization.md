# Regularization (Preventing Overfitting)

## What Is It?

Regularization is a set of techniques that prevent a model from **memorizing the training data** instead of learning general patterns. Without it, your model gets perfect scores on training data but fails on new data.

Think of it like studying for an exam: memorizing the exact practice questions (overfitting) vs understanding the concepts (generalization).

## Frontend Analogy — Overfitting is Over-Engineering

```javascript
// OVERFIT: Style rule that only works for one specific page
.homepage > div:nth-child(3) > ul > li:first-child > a.nav-link-blue-v2 {
  color: blue;
}

// GENERALIZED: Style rule that works everywhere
.nav-link { color: blue; }

// The overfit version perfectly matches one case but breaks on every other page.
// Regularization is like a linter that says "your selector is too specific."
```

## How to Spot Overfitting

```
Training accuracy: 99.5%    ← memorized the training data
Validation accuracy: 72.0%  ← fails on new data
Gap: 27.5%                  ← OVERFITTING

Training accuracy: 85.0%    ← good but not perfect
Validation accuracy: 83.0%  ← holds up on new data
Gap: 2.0%                   ← HEALTHY
```

```
Loss
  |╲
  |  ╲______ training loss keeps going down
  |         ╲_______________
  |
  |    ╲___
  |        ╲___
  |            ╲_______ validation loss starts going UP ← stop here!
  |                    ╱
  |___________________╱_______ Epochs
            ↑
       Stop training here (early stopping)
```

## The Regularization Toolkit

### 1. Dropout — The Most Common Technique

Randomly **turn off neurons** during training (set their output to 0). Each training step uses a different random subset of neurons.

```
Without dropout:           With dropout (p=0.5):
 ○ → ○ → ○ → ○             ○ → ○ → ✕ → ○
 ○ → ○ → ○ → ○             ✕ → ○ → ○ → ○
 ○ → ○ → ○ → ○             ○ → ✕ → ○ → ○
 ○ → ○ → ○ → ○             ○ → ○ → ○ → ✕

All neurons active          Random neurons disabled (✕)
                            Next batch: different neurons disabled
```

**Why it works:** Forces the network to not rely on any single neuron. Like studying with different friends each time — you learn the material, not just copy one person's notes.

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(0.3),        # 30% of neurons randomly zeroed
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),        # apply before each layer
            nn.Linear(128, 10),     # no dropout on output layer!
        )
```

**Important:** Dropout is only active during training. During inference (prediction), all neurons are used. PyTorch handles this automatically with `model.train()` and `model.eval()`.

**Typical values:**
- Input layers: `0.1 - 0.2`
- Hidden layers: `0.2 - 0.5`
- Never on the output layer

### 2. Weight Decay (L2 Regularization)

Add a penalty for large weights to the loss function. Big weights = overly specific rules.

```
Total Loss = Prediction Loss + λ × sum(weights²)

λ (lambda) controls how strong the penalty is
```

Think of it as: "I want accurate predictions, BUT I also want simple weights." Large weights = complex model = overfitting.

```python
# Built into the optimizer — just add weight_decay
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
#                                                           ↑ this is λ
```

**Typical values:** `0.01` to `0.1`

### 3. Batch Normalization

Normalize the outputs of each layer to have **mean ≈ 0** and **standard deviation ≈ 1**. This stabilizes training and acts as mild regularization.

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),     # normalize after linear
            nn.ReLU(),               # then activate
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
```

**Why it helps:**
- Keeps values in a reasonable range (no exploding/vanishing)
- Smooths the loss landscape (easier to optimize)
- Mild regularization effect (each mini-batch has slightly different statistics)

### 4. Early Stopping

Stop training **before** the model overfits. Monitor validation loss and stop when it stops improving.

```python
best_val_loss = float('inf')
patience = 5
no_improve_count = 0

for epoch in range(100):
    train_loss = train_one_epoch()
    val_loss = validate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_count = 0
        torch.save(model.state_dict(), 'best_model.pt')  # save best
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Load the best model (not the last one!)
model.load_state_dict(torch.load('best_model.pt'))
```

### 5. Data Augmentation (For Images)

Create **modified versions** of your training images. The model sees more variety without needing more real data.

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # flip left-right
    transforms.RandomRotation(15),          # rotate up to 15°
    transforms.RandomCrop(224, padding=16), # random crop position
    transforms.ColorJitter(                 # random color changes
        brightness=0.2, contrast=0.2
    ),
    transforms.ToTensor(),
])

# Same image → different variations each epoch
# Dog facing left, dog facing right, dog zoomed in, darker dog...
# All from ONE original image
```

**Think of it like responsive design testing:** same content, different viewports/conditions.

### 6. Label Smoothing

Instead of hard labels (0 or 1), use soft labels (0.05 or 0.95). Prevents the model from being **overconfident**.

```
Hard labels: [0, 0, 1, 0, 0]          "I'm 100% sure it's class 3"
Soft labels: [0.01, 0.01, 0.95, 0.01, 0.01]  "I'm 95% sure it's class 3"
```

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

## Cheat Sheet — Which Techniques to Use

| Technique | When | Typical Config |
|-----------|------|----------------|
| **Dropout** | Almost always for dense layers | 0.1-0.5 |
| **Weight decay** | Almost always | 0.01 in AdamW |
| **Early stopping** | Always — no reason not to | patience=5-10 |
| **Batch norm** | Most architectures (not Transformers) | After linear, before activation |
| **Data augmentation** | Image tasks (always) | Flips, rotations, crops |
| **Label smoothing** | Classification tasks | 0.1 |

## The Typical Recipe

```python
class WellRegularizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),     # batch norm
            nn.ReLU(),
            nn.Dropout(0.3),         # dropout

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 10),      # no dropout/batchnorm on output
        )

# AdamW with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Label smoothing
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# Early stopping in training loop
# Data augmentation in data loader
```

## Underfitting vs Overfitting

| Signal | Underfitting | Overfitting |
|--------|-------------|-------------|
| Training accuracy | Low | High |
| Validation accuracy | Low | Low |
| Gap | Small | Large |
| Fix | Bigger model, more layers, train longer | More regularization, more data |

```
Underfitting:          Just right:           Overfitting:
  Model too simple      Good balance          Model too complex
  ____                  ___                   ∿∿∿∿∿
      ____             /   \                 /\  /\  /\
          \___        /     \___            /  \/  \/  \
```

## Key Takeaway

Regularization is **how you make models that work in the real world**, not just on training data. The essentials: use **dropout** (0.3), **weight decay** (0.01 in AdamW), **early stopping** (always), and **data augmentation** (for images). You can stack multiple techniques — they complement each other. If your training accuracy is way higher than validation accuracy, you need more regularization or more data.
