# PyTorch Basics

## 1. TL;DR

PyTorch is the dominant framework for building and training neural networks — think of it as the React of deep learning. It has four core ideas: tensors (multi-dimensional arrays that run on GPU), `nn.Module` (the building block for all models), autograd (automatic gradient computation), and DataLoader (efficient batching). The training loop is always the same 4 steps: forward → loss → backward → update. Learn this loop by heart.

---

## 2. The Mental Model

> 💡 **PyTorch is React for neural networks.**

In React, you define components (reusable UI pieces), they receive props (data), maintain state, and React handles the DOM updates. In PyTorch, you define modules (reusable network pieces), they receive tensors (data), maintain parameters (learned weights), and PyTorch handles the gradient updates.

| React | PyTorch |
|---|---|
| Component (`React.Component`) | Module (`nn.Module`) |
| Props (data in) | Tensors (data flowing through) |
| State (component memory) | Parameters (learned weights) |
| `render()` | `forward()` |
| Virtual DOM diffing | Autograd (automatic gradients) |
| `npm install` a library | `torchvision`, `transformers`, `timm` |
| `useState`, `useEffect` | `model.train()`, `model.eval()` |

---

## 3. Why It Exists

**The problem:** Writing matrix math for neural networks manually is error-prone. Computing derivatives for backpropagation by hand for millions of parameters is impossible.

**What came before:** Theano (2007) first enabled GPU computation and symbolic differentiation, but had a steep learning curve. Early TensorFlow used static computation graphs — you had to define the full graph before running it, making debugging nightmarish.

**What changed:** PyTorch (2016) introduced dynamic computation graphs — the graph is built as you run Python code, so you can use regular Python debuggers (`print()`, `pdb`). Research productivity exploded. It became the standard in academia and has since taken over industry too (Meta, OpenAI, Anthropic all use it).

---

## Build the Intuition From Zero

PyTorch is mostly practical, but one thing feels like magic and shouldn't: **how `loss.backward()` knows the gradient of every weight without you ever writing calculus.** The answer is autograd, and it's simpler than it looks.

### Idea 1: PyTorch secretly records every operation (the tape)

As your `forward()` runs, PyTorch quietly writes down each math operation in order — like a security camera taping every step. This recording is the **computational graph**:

```
you write:        loss = ((w * x + b) − y) ** 2

PyTorch records:  x ──┐
                      ×──→ ──+──→ ──−──→ ──²──→ loss
                  w ──┘     ↑      ↑
                            b      y
                  (it remembers every node and how they connect)
```

Any tensor you mark with `requires_grad=True` (all model parameters are, automatically) gets tracked. The graph is rebuilt fresh on every forward pass — that's the "dynamic" part that lets you use plain Python `if`/`for` inside a model.

### Idea 2: backward() = replay the tape in reverse, multiplying

When you call `loss.backward()`, PyTorch plays that tape **backwards**, applying the chain rule from [backpropagation.md](backpropagation.md) — multiplying local gradients along the way — to hand every tracked tensor its `.grad` (its share of the blame for the loss):

```
forward:   x, w, b ───────────────────────────→ loss     (build the tape)
backward:  loss ─── chain-rule multiply back ──→ fills w.grad, b.grad   (.backward())
then:      optimizer.step()  →  w −= lr * w.grad           (take the downhill step)
```

> 💡 **One line:** autograd tapes every operation in the forward pass, then `loss.backward()` replays it in reverse to compute every gradient automatically — so you only ever write the *forward* math, and PyTorch derives the backward for free. That's why the training loop is always the same 4 lines: `forward → loss → backward → step`.

This is also why you must call `optimizer.zero_grad()` each iteration: `.grad` *accumulates* (adds up) by default, so you wipe last step's gradients before computing this step's. Forgetting it is the #1 beginner bug. The tensor, autograd, and training-loop sections below put this into real code.

---

## 4. Core Concepts

### Tensors

**One-line definition:** Multi-dimensional arrays — like NumPy arrays, but they can run on GPU and support automatic differentiation.

**Analogy:** Think of them as typed, GPU-capable JavaScript arrays with shape metadata. A 2D tensor is a matrix; a 4D tensor is a batch of RGB images.

```python
import torch

scalar = torch.tensor(42)                    # 0D — shape: []
vector = torch.tensor([1, 2, 3])             # 1D — shape: [3]
matrix = torch.tensor([[1, 2], [3, 4]])      # 2D — shape: [2, 2]
image  = torch.randn(3, 224, 224)            # 3D — [channels, H, W]
batch  = torch.randn(32, 3, 224, 224)        # 4D — [batch, C, H, W]
```

**Common misconception:** ❌ "Tensors are just NumPy arrays with a different name" → ✅ They support GPU acceleration (`tensor.to('cuda')`) and gradient tracking (`requires_grad=True`) — both impossible with plain NumPy.

---

### `nn.Module`

**One-line definition:** The base class for all PyTorch models — defines the architecture and how data flows through it.

**Analogy:** A React component. You define `__init__` (what layers exist) and `forward` (how data flows) — PyTorch handles the rest.

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)   # define layers in __init__
        self.fc2 = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):              # define data flow in forward
        x = self.relu(self.fc1(x))
        return self.fc2(x)
```

**Common misconception:** ❌ "You call `forward()` directly" → ✅ Call the model like a function: `model(x)`. This invokes `__call__`, which runs hooks and then calls `forward()`.

---

### Autograd (Automatic Differentiation)

**One-line definition:** PyTorch records every operation on tensors and can compute gradients automatically by walking the computation graph in reverse.

**Analogy:** A detailed receipt for every mathematical operation — when you need to know "how much did flour contribute to the final cost?", you just trace the receipt backwards.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x     # y = x² + 2x, dy/dx = 2x + 2

y.backward()            # compute gradient
print(x.grad)           # 2×3 + 2 = 8.0 ✓
```

**Common misconception:** ❌ "`requires_grad=True` on all tensors for safety" → ✅ Only model parameters need gradients. Input data and labels should NOT have `requires_grad=True` — it wastes memory and compute.

---

### DataLoader

**One-line definition:** Handles batching, shuffling, and parallel data loading automatically.

**Analogy:** A waiter who batches your table's orders (batch_size), shuffles the seating each round (shuffle=True), and has multiple staff loading food in parallel (num_workers).

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for batch_X, batch_y in loader:
    print(batch_X.shape)  # [32, num_features]
```

**Common misconception:** ❌ "Just process the full dataset at once" → ✅ Most datasets don't fit in GPU memory. Batching also makes gradient descent stochastic, which helps escape local minima.

---

### `model.train()` vs `model.eval()`

**One-line definition:** Two modes that change behavior of dropout and batch normalization — you MUST switch between them correctly.

**Analogy:** A factory machine in "production mode" (all systems active, strict QC) vs "test mode" (certain systems bypassed for calibration).

```python
model.train()   # dropout active, batchnorm uses batch statistics
model.eval()    # dropout disabled, batchnorm uses running statistics
```

**Common misconception:** ❌ "These modes affect what gets computed in `loss.backward()`" → ✅ They only affect dropout and batch normalization behavior. Gradients are controlled by `torch.no_grad()`, not `eval()`.

---

### `torch.no_grad()`

**One-line definition:** A context manager that disables gradient computation — use during inference to save memory and speed up predictions.

**Analogy:** Read-only mode in a database — you can query freely but no writes are recorded (no gradient tape).

```python
model.eval()
with torch.no_grad():          # no gradient tape = 2× faster, less memory
    predictions = model(X_test)
    accuracy = (predictions.argmax(1) == y_test).float().mean()
```

**Common misconception:** ❌ "`model.eval()` disables gradient computation" → ✅ Only `torch.no_grad()` does that. `eval()` only changes dropout/batchnorm behavior.

---

> 🧠 **Quick recall — answer out loud before scrolling on** (all answers are above):
> 1. What does autograd record during `forward()`, and what does `backward()` do with it?
> 2. Why must `optimizer.zero_grad()` run every iteration — what accumulates otherwise?
> 3. `model.eval()` vs `torch.no_grad()` — what does each actually change?
> 4. Why call `model(x)` instead of `model.forward(x)`?
> 5. Why save `state_dict()` rather than the whole model object?

---

## 5. How It Actually Works — Step by Step

Full lifecycle of training a classifier:

```
Step 1: DEFINE MODEL
  model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 3))
  → PyTorch registers all parameters (weights + biases) automatically

Step 2: MOVE TO DEVICE
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = model.to(device)
  → Parameters now live on GPU (if available)

Step 3: DEFINE LOSS + OPTIMIZER
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

Step 4: TRAINING LOOP (repeat per epoch)
  for batch_X, batch_y in train_loader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    pred = model(batch_X)           # forward pass → builds computation graph
    loss = loss_fn(pred, batch_y)   # compute scalar loss

    optimizer.zero_grad()           # clear old gradients
    loss.backward()                 # walk graph backwards → fill .grad
    optimizer.step()                # update weights using .grad

Step 5: VALIDATION (per epoch)
  model.eval()
  with torch.no_grad():
    val_pred = model(val_X.to(device))
    val_loss = loss_fn(val_pred, val_y.to(device))
  model.train()

Step 6: SAVE
  torch.save(model.state_dict(), 'model.pt')
```

---

## 6. Code in Practice

### Minimal — 10-line training loop
```python
import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

X, y = torch.randn(100, 4), torch.randint(0, 3, (100,))

for _ in range(100):
    opt.zero_grad()
    loss_fn(model(X), y).backward()
    opt.step()
```

### Practical — Full training + validation loop
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Data
X_train = torch.randn(800, 10)
y_train = torch.randint(0, 3, (800,))
X_val   = torch.randn(200, 10)
y_val   = torch.randint(0, 3, (200,))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# Model
model = nn.Sequential(
    nn.Linear(10, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 3),
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Training loop
for epoch in range(20):
    model.train()
    total_loss = 0
    for bX, by in train_loader:
        bX, by = bX.to(device), by.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(bX), by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val.to(device)), y_val.to(device))

    print(f"Epoch {epoch+1}: train={total_loss/len(train_loader):.4f}, val={val_loss:.4f}")
```

### Real-world — Save, load, and inference
```python
# Save
torch.save(model.state_dict(), 'model.pt')

# Load
model = nn.Sequential(
    nn.Linear(10, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 3),
)
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
model.eval()

# Inference
with torch.no_grad():
    single_input = torch.randn(1, 10)
    logits = model(single_input)
    predicted_class = logits.argmax(dim=1).item()
    print(f"Predicted class: {predicted_class}")
```

---

## 7. Gotchas & Pitfalls

| ❌ Wrong Assumption | ✅ Reality |
|---|---|
| Call `model.forward(x)` directly | Call `model(x)` — the `__call__` wrapper runs essential hooks first |
| `model.eval()` disables gradient computation | Use `torch.no_grad()` for that; `eval()` only affects dropout/batchnorm |
| Forget `optimizer.zero_grad()` | Gradients accumulate by default — old gradients corrupt the current batch |
| Forget to move data to GPU | Model on GPU + data on CPU → runtime error. Always `.to(device)` both |
| Forget `model.train()` after validation | If you stay in eval mode, dropout is disabled during training — model won't regularize |
| `torch.save(model, ...)` vs `torch.save(model.state_dict(), ...)` | Save `state_dict()` — saving the whole model is fragile across Python versions |
| Shape errors are mysterious | Print `.shape` at every step when debugging — 90% of PyTorch bugs are shape mismatches |

---

## 8. When to Use / When NOT to Use

PyTorch is the tool for all deep learning in this study path. But within PyTorch:

**Use `nn.Sequential` when:**
- Architecture is a simple linear chain (most MLPs, simple classifiers)
- Prototyping quickly

**Use custom `nn.Module` class when:**
- You have skip connections, multiple inputs/outputs, or conditional logic
- You're implementing a research paper architecture

**Use `torch.no_grad()` when:**
- Running validation or inference — always, without exception
- You want 2× faster inference and lower memory usage

**Use `DataLoader` when:**
- Your dataset doesn't fit in RAM or GPU memory
- You need shuffling, batching, or parallel data loading (always)

**Skip PyTorch and use Hugging Face `Trainer` when:**
- Fine-tuning a pretrained language or vision model
- You want training, evaluation, and checkpointing managed for you

---

## 9. Related Concepts (The Map)

- **Tensors ↔ NumPy**: PyTorch tensors and NumPy arrays share memory when on CPU — convert with `.numpy()` and `torch.from_numpy()`
- **Autograd ↔ Backpropagation**: Autograd is the implementation; backpropagation is the algorithm (see `backpropagation.md`)
- **`nn.Module` ↔ every architecture**: CNNs, RNNs, Transformers are all `nn.Module` subclasses (see `cnn.md`, `rnn-lstm.md`, `transformers.md`)
- **DataLoader ↔ Regularization**: `shuffle=True` in DataLoader is a subtle form of regularization — random ordering prevents the model from memorizing batch order (see `regularization.md`)
- **`state_dict` ↔ Transfer learning**: Pretrained models are distributed as state dicts loaded into an `nn.Module` (see `transfer-learning.md`)

---

## 10. Cheat Sheet

**Tensor creation:**
```python
torch.zeros(3, 3)          # all zeros
torch.ones(3, 3)           # all ones
torch.randn(3, 3)          # random normal
torch.arange(0, 10)        # [0, 1, ..., 9]
torch.tensor([1, 2, 3])    # from Python list
```

**Tensor operations:**
```python
x.shape                    # size of each dimension
x.reshape(6, 4)            # change shape (same data)
x.unsqueeze(0)             # add dimension at position 0
x.squeeze()                # remove size-1 dimensions
x.to('cuda')               # move to GPU
x.cpu().numpy()            # convert to NumPy (CPU only)
```

**Model lifecycle:**
```python
model.train()              # enable dropout/batchnorm training behavior
model.eval()               # disable dropout/batchnorm for inference
model.parameters()         # iterator over all learnable params
model.state_dict()         # dict of all param tensors (for saving)
sum(p.numel() for p in model.parameters())  # count total params
```

**The training core — memorize this:**
```python
optimizer.zero_grad()
loss = loss_fn(model(X), y)
loss.backward()
optimizer.step()
```

**Remember these 3 things:**
1. The 4-step training loop: zero_grad → forward → backward → step
2. Always switch `model.train()` / `model.eval()` and use `torch.no_grad()` for inference
3. Shape mismatches cause 90% of bugs — print `.shape` at every layer when debugging

---

## 11. Self-Check Questions

1. What are the four steps of the PyTorch training loop, and what does each do?
2. What is the difference between `model.eval()` and `torch.no_grad()`?
3. Why do you call `optimizer.zero_grad()` before `loss.backward()`, not after `optimizer.step()`?
4. You move your model to GPU but forget to move your input data. What happens?
5. What is `state_dict()` and why should you save it instead of the full model?

<details>
<summary>Brief Answers</summary>

1. **`optimizer.zero_grad()`** — clears accumulated gradients from the previous iteration. **`loss = loss_fn(model(X), y)`** — runs the forward pass and computes the scalar loss. **`loss.backward()`** — walks the computation graph backwards, computing `∂loss/∂param` for every parameter and storing it in `param.grad`. **`optimizer.step()`** — reads each `param.grad` and updates the weights according to the optimizer's update rule.

2. **`model.eval()`** changes the *behavior* of dropout (disabled) and batch normalization (uses running statistics instead of batch statistics). **`torch.no_grad()`** disables the gradient *tape* — no computation graph is built, saving memory and making inference ~2× faster. You typically use both together during validation: `model.eval()` then `with torch.no_grad():`.

3. PyTorch **accumulates** gradients — each call to `.backward()` adds to existing `.grad` values rather than replacing them. This is useful for gradient accumulation tricks, but normally you want fresh gradients per batch. Calling `zero_grad()` before `backward()` ensures each batch's gradients are computed cleanly without contamination from previous batches.

4. A **RuntimeError** at the first operation that tries to combine them: "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!" Always move both model and data to the same device: `X = X.to(device)`.

5. `state_dict()` is an ordered dict of all parameter tensors (weights and biases), keyed by layer name. Saving the full model with `torch.save(model, ...)` pickles the entire Python object — this breaks when you rename classes, change the module structure, or upgrade Python/PyTorch versions. `state_dict()` is just data (tensors), making it stable, portable, and compatible with future code changes.

</details>

---

## 12. Go Deeper

- **PyTorch official "60-minute blitz"** (pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html): The canonical first PyTorch tutorial. Covers tensors, autograd, and a full training loop. [Why: learn the tool from the source — it's well-maintained and accurate.]

- **Andrej Karpathy — "Neural Networks: Zero to Hero"** (YouTube): Builds everything from scratch in pure Python, then rebuilds in PyTorch. The best way to truly understand what PyTorch is doing. [Why: after watching, `loss.backward()` will never feel like magic again.]

- **fast.ai — Practical Deep Learning** (course.fast.ai): Top-down approach — you train a state-of-the-art image classifier in lesson 1, then learn how PyTorch works in later lessons. [Why: gets you productive fast; complements the bottom-up approach of this doc.]

- **PyTorch documentation — `torch.nn`** (pytorch.org/docs/stable/nn.html): The complete reference for every layer, loss, and activation. Bookmark it — you'll use it weekly. [Why: authoritative source; includes every parameter and edge case.]

- **"effective_pytorch"** (github.com/vahidk/effective_pytorch): A collection of PyTorch best practices and gotchas from production experience. [Why: covers the non-obvious things that bite you in real projects — gradient accumulation, mixed precision, memory management.]
