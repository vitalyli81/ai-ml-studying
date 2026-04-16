# Regularization (Preventing Overfitting)

## 1. TL;DR

Overfitting is when your model memorizes training data instead of learning general patterns — it scores 99% on training data and 70% on new data. Regularization is the set of techniques that prevent this. The essentials: dropout (randomly disable neurons during training), weight decay (penalize large weights via AdamW), early stopping (stop when validation loss stops improving), and data augmentation (for image tasks). Stack these together — they complement each other.

---

## 2. The Mental Model

> 💡 **Think of it as the difference between cramming for an exam vs. actually learning the material.**

A student who memorizes every practice question (overfitting) aces the practice tests but fails the real exam when questions are worded differently. A student who understands the underlying concepts (generalization) handles novel questions fine. Regularization is the set of study habits that force genuine understanding rather than memorization.

- **Cramming exact practice questions** → memorizing training data (overfitting)
- **Understanding concepts** → learning general patterns (generalization)
- **Studying with different friends each day** → dropout (can't rely on any single connection)
- **Keeping notes brief and clean** → weight decay (penalize complex, large weights)
- **Stopping when you're ready, not when time's up** → early stopping
- **Practicing with varied question phrasings** → data augmentation

---

## 3. Why It Exists

**The problem:** Neural networks are extremely high-capacity — a large network can memorize the entire training set, including noise and quirks specific to those examples. This is useless for making predictions on new data.

**What came before:** Classic ML models (linear regression, SVMs) had built-in regularization as mathematical constraints. Neural networks are so flexible that without explicit regularization, they overfit aggressively.

**What changed:** Dropout (Srivastava et al., 2014) was a breakthrough — simple, effective, and model-agnostic. Batch normalization (2015) solved both training instability and acted as mild regularization. Together with data augmentation, these techniques allowed training of very deep networks without overfitting.

---

## 4. Core Concepts

### Overfitting vs. Underfitting

**One-line definition:** Overfitting = too much memorization; underfitting = too little capacity to learn.

**Analogy:** A tailor who makes a suit perfectly fitted to one person (overfitting) vs. one who makes a one-size-fits-all suit (underfitting) vs. one who makes a well-fitted suit that still works after you gain 5 pounds (generalization).

```
Training accuracy: 99.5%   Validation accuracy: 72%   → OVERFIT (big gap)
Training accuracy: 75%     Validation accuracy: 73%   → HEALTHY (small gap)
Training accuracy: 60%     Validation accuracy: 59%   → UNDERFIT (both low)
```

**Diagnosis via loss curves:**
```
Loss
│╲  training loss      validation loss
│  ╲_________                          ← underfit: both high
│
│╲
│  ╲___  ← train      ╱ ← val rising  ← overfit: diverging
│              ╲____╱
│
│╲
│  ╲_______________  ← both converging ← healthy
└────────────────── Epochs
```

**Common misconception:** ❌ "Zero training loss is the goal" → ✅ The goal is low validation loss. Training loss = 0 with high validation loss = completely overfit.

---

### Dropout

**One-line definition:** Randomly zero out a fraction of neurons during each training step, forcing the network not to rely on any single neuron.

**Analogy:** A sports team that practices with random players benched each session. Every player must be capable of covering multiple roles — no single point of failure.

```
Without dropout:           With dropout (p=0.3):
 ●─●─●─●                   ●─●─✕─●
 ●─●─●─●                   ✕─●─●─●     ✕ = zeroed out
 ●─●─●─●                   ●─✕─●─●
 ●─●─●─●                   ●─●─●─✕

All neurons active          30% randomly disabled (different each step)
```

**Technical explanation:** During training, each neuron has probability `p` of being set to 0 for that forward pass. At inference, all neurons are active but outputs are scaled by `(1-p)` to maintain the same expected magnitude.

```python
nn.Dropout(p=0.3)   # 30% of neurons zeroed per forward pass
```

**Common misconception:** ❌ "Dropout slows inference" → ✅ Dropout is disabled during `model.eval()` — inference uses all neurons at full speed.

---

### Weight Decay (L2 Regularization)

**One-line definition:** Add a penalty to the loss for large weights, pushing the model toward simpler solutions.

**Analogy:** A rental agreement that charges extra for each piece of furniture you own — you keep only what you really need. Large weights = unnecessary complexity.

**Technical explanation:**
```
Total Loss = Prediction Loss + λ × Σ(w²)

λ controls strength (typical: 0.01)
Large weights add to the loss → optimizer shrinks them
```

In PyTorch, pass it directly to AdamW:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

**Common misconception:** ❌ "Weight decay and dropout do the same thing" → ✅ Dropout regularizes by introducing noise during training; weight decay regularizes by constraining weight magnitude. They address different aspects of overfitting and work well together.

---

### Batch Normalization

**One-line definition:** Normalize each layer's outputs to have mean ≈ 0 and std ≈ 1, stabilizing training and providing mild regularization.

**Analogy:** A standardized intake form for every hospital — regardless of how a patient arrived (different data distributions), they fill out the same form in the same format. Each layer sees "normal" inputs.

```python
nn.Sequential(
    nn.Linear(100, 256),
    nn.BatchNorm1d(256),   # normalize: mean→0, std→1
    nn.ReLU(),             # then activate
)
```

**Why it helps:**
- Prevents internal covariate shift (layer inputs shifting distributions during training)
- Smooths the loss landscape → easier optimization
- Acts as mild regularization (each mini-batch has slightly different statistics)

**Common misconception:** ❌ "BatchNorm replaces dropout" → ✅ They're complementary. BatchNorm stabilizes training; dropout prevents reliance on individual neurons. Use both.

---

### Early Stopping

**One-line definition:** Monitor validation loss during training and stop when it stops improving — don't wait for training loss to hit zero.

**Analogy:** Knowing when to stop studying. Once you can answer every practice question correctly, studying more doesn't help — it just makes you more rigid and test-specific.

```
Epoch 1: train_loss=2.0, val_loss=2.1  ← both improving
Epoch 5: train_loss=0.8, val_loss=0.9  ← healthy
Epoch 10: train_loss=0.3, val_loss=1.1  ← val_loss rising → STOP HERE
Epoch 15: train_loss=0.1, val_loss=1.8  ← overfitting badly (if you continued)
```

**Common misconception:** ❌ "Use the final model weights" → ✅ Save the best model (lowest val_loss) and reload those weights after stopping. The final epoch's weights are overfit.

---

### Data Augmentation

**One-line definition:** Artificially create variations of training examples so the model sees more diversity without needing more real data.

**Analogy:** A language learner who practices the same conversation in different accents, speeds, and contexts — the underlying meaning is the same, but the surface variation forces deeper understanding.

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(224, padding=16),
    transforms.ToTensor(),
])
# Each epoch, every image looks slightly different → model can't memorize exact pixels
```

**Common misconception:** ❌ "Data augmentation is only for small datasets" → ✅ Even large datasets benefit from augmentation — it's always used in state-of-the-art image models.

---

### Label Smoothing

**One-line definition:** Instead of hard 0/1 labels, use soft labels (0.05/0.95) to prevent the model from being overconfident.

**Analogy:** A teacher who says "this is almost certainly the answer, but never be 100% certain" — it trains intellectual humility.

```
Hard labels: [0, 0, 1, 0, 0]           "100% sure it's class 3"
Soft labels: [0.01, 0.01, 0.95, 0.01, 0.01]  "95% sure, but keeping options open"
```

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**Common misconception:** ❌ "Label smoothing makes training less accurate" → ✅ It often *improves* final accuracy by preventing overconfident predictions that overfit training labels.

---

## 5. How It Actually Works — Step by Step

Training with dropout + weight decay + early stopping:

```
Setup:
  model with Dropout(0.3) in hidden layers
  optimizer = AdamW(lr=1e-3, weight_decay=0.01)
  best_val_loss = infinity
  patience = 5, no_improve = 0

Epoch 1:
  model.train()
  For each batch:
    → 30% of hidden neurons randomly zeroed (dropout)
    → Forward pass with remaining 70%
    → Loss = prediction_loss + 0.01 × Σ(w²)  ← weight decay implicit in AdamW
    → Gradients flow back through active neurons only
    → Weights updated (large weights penalized more)
  model.eval()
  val_loss = 1.2   → new best! Save weights. no_improve = 0

Epoch 5:
  val_loss = 0.85  → new best! Save weights. no_improve = 0

Epoch 10:
  val_loss = 0.91  → worse! no_improve = 1

Epoch 14:
  val_loss = 1.05  → worse! no_improve = 5  → STOP
  Load saved weights from Epoch 5 (val_loss=0.85)
```

---

## 6. Code in Practice

### Minimal — Dropout in a network
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Dropout(0.3),        # 30% dropout after hidden layer
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10),     # no dropout on output layer!
)
```

### Practical — Full regularized model
```python
class RegularizedModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),   # batch norm before activation
            nn.ReLU(),
            nn.Dropout(0.3),       # dropout after activation

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# AdamW = weight decay built in
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
# Label smoothing
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Real-world — Early stopping implementation
```python
best_val_loss = float('inf')
patience = 7
no_improve = 0

for epoch in range(200):
    # --- train ---
    model.train()
    for bX, by in train_loader:
        optimizer.zero_grad()
        loss_fn(model(bX), by).backward()
        optimizer.step()

    # --- validate ---
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), y_val).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), 'best_model.pt')  # save best
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
            break

# Always load the best checkpoint, not the last epoch
model.load_state_dict(torch.load('best_model.pt'))
```

---

## 7. Gotchas & Pitfalls

| ❌ Wrong Assumption | ✅ Reality |
|---|---|
| Keep training until training loss = 0 | Stop when validation loss stops improving — training loss = 0 usually means overfit |
| Dropout on the output layer | Never apply dropout to the final layer — it corrupts the probability distribution |
| Forgetting `model.eval()` at inference | Dropout stays active and randomly zeros outputs — predictions are random and wrong |
| BatchNorm in Transformers | Transformers use LayerNorm, not BatchNorm — BatchNorm is for CNNs and MLPs |
| More dropout = always better | Too much dropout (p > 0.5) makes the model underfit — it can't learn complex patterns |
| Data augmentation is always safe | Augmentations must preserve the label. Horizontally flipping "6" → "9" is not safe for digit recognition |
| Early stopping with training loss | Monitor **validation** loss, not training loss — training loss always decreases |

---

## 8. When to Use / When NOT to Use

**Always use:**
- Early stopping — zero downside, always helps
- Weight decay (via AdamW) — almost always beneficial, default: `0.01`
- `model.eval()` + `torch.no_grad()` during inference — always required

**Use dropout when:**
- Training dense (fully connected) layers
- Seeing a large gap between training and validation accuracy

**Use batch normalization when:**
- Training CNNs or MLPs with many layers
- Training is unstable (loss oscillating)

**Use data augmentation when:**
- Image or audio tasks — always
- Your dataset is small (< 10,000 samples)

**Use label smoothing when:**
- Classification tasks where you want better-calibrated confidence
- Fine-tuning large pretrained models

**Skip batch normalization when:**
- Using Transformers — use LayerNorm instead
- Batch size is very small (< 8) — BN statistics become unreliable

---

## 9. Related Concepts (The Map)

- **Overfitting ↔ Model capacity**: Larger models overfit more easily; regularization allows you to use higher-capacity models safely (see `neural-networks-basics.md`)
- **Dropout ↔ Ensemble methods**: Dropout approximates training an ensemble of exponentially many networks simultaneously — each forward pass uses a different sub-network
- **Weight decay ↔ AdamW optimizer**: Weight decay is built into AdamW — just set `weight_decay=0.01` in the optimizer; no separate loss term needed (see `loss-functions-optimizers.md`)
- **Data augmentation ↔ Transfer learning**: With pretrained models, you need less data augmentation because the model already knows general features; but augmentation still helps (see `transfer-learning.md`)
- **BatchNorm ↔ LayerNorm**: CNNs use BatchNorm; Transformers use LayerNorm — they normalize over different dimensions (see `transformers.md`)

---

## 10. Cheat Sheet

| Technique | When | Config |
|---|---|---|
| **Dropout** | Dense/hidden layers | `p=0.2–0.5`; never on output |
| **Weight decay** | Almost always | `0.01` in AdamW |
| **Early stopping** | Always | `patience=5–10` epochs |
| **Batch normalization** | CNN/MLP (not Transformers) | After linear, before activation |
| **Data augmentation** | Image tasks | Flips, crops, color jitter |
| **Label smoothing** | Classification | `0.05–0.1` |

**Diagnosing with accuracy gap:**
```
train=99%, val=72% → heavy overfitting → more regularization / more data
train=85%, val=83% → healthy
train=65%, val=64% → underfitting → bigger model, train longer
```

**Remember these 3 things:**
1. Monitor validation loss, not training loss — that's the real score
2. Always use early stopping + weight decay — zero cost, always helps
3. Save the best model checkpoint, not the last epoch

---

## 11. Self-Check Questions

1. Your model achieves 98% training accuracy and 71% validation accuracy. Name two techniques you'd add and explain why each helps.
2. Why should you never use dropout on the output layer?
3. What does `model.eval()` do to dropout specifically?
4. Explain what weight decay does mathematically and how to apply it in PyTorch.
5. You implement early stopping and it triggers at epoch 12, but you saved no checkpoints. What do you load?

<details>
<summary>Brief Answers</summary>

1. **Dropout** (`p=0.3` in hidden layers) — forces the network to distribute learning across all neurons rather than relying on specific ones, preventing over-specialization to training samples. **Weight decay** (`weight_decay=0.01` in AdamW) — adds a penalty proportional to weight magnitude, pushing the optimizer toward simpler solutions (smaller weights = less complex decision boundaries = less memorization).

2. The output layer produces probability distributions (via softmax) or raw scores for your classes. Applying dropout here randomly zeros class scores, corrupting the probability distribution and making predictions meaningless. Regularization on the output directly harms task performance — apply it only to hidden representations.

3. `model.eval()` completely disables dropout — all neurons are active for every forward pass. During training, each neuron has probability `p` of being zeroed. During eval, all neurons contribute with their weights scaled by `(1-p)` to match the expected activation magnitude from training (PyTorch handles this scaling automatically).

4. Weight decay adds `λ × Σ(w²)` to the total loss. Since the optimizer minimizes total loss, it now has an incentive to keep weights small. In PyTorch: `torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)` — the `weight_decay=0.01` parameter applies the penalty; you don't modify the loss function yourself.

5. **The weights from your last epoch** — but these are almost certainly overfit since early stopping triggered because val_loss was worsening. This is exactly why you should always save the best checkpoint during training: `torch.save(model.state_dict(), 'best.pt')` whenever `val_loss < best_val_loss`. Without that, you've lost your best model and must retrain.

</details>

---

## 12. Go Deeper

- **"Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (Srivastava et al., 2014)**: The original dropout paper — readable, has great visualizations of why it works. [Why: understanding the ensemble interpretation of dropout is the "aha" moment for why it works so well.]

- **"Batch Normalization: Accelerating Deep Network Training" (Ioffe & Szegedy, 2015)**: The BatchNorm paper. Explains why internal covariate shift was a problem and how normalization solves it. [Why: BatchNorm is in nearly every modern CNN; knowing why it works makes you a better debugger.]

- **Andrej Karpathy — "A Recipe for Training Neural Networks"** (karpathy.github.io/2019/04/25/recipe): The practitioner's debugging guide — section on overfitting is excellent. [Why: practical, battle-tested advice on diagnosing and fixing training problems.]

- **fast.ai — Data Augmentation chapter**: Jeremy Howard's comprehensive treatment of augmentation techniques for images. Includes `albumentations` library and modern augmentation strategies (Mixup, CutMix). [Why: augmentation is an easy win that most beginners under-use — this shows you the full range of what's possible.]

- **"Three Mechanisms of Weight Decay Regularization" (Zhang et al., 2018)**: Explains why L2 regularization / weight decay works — not just "keeps weights small" but three distinct mechanisms. [Why: if you want to go beyond cargo-culting `weight_decay=0.01`, this gives you the theoretical backing.]
