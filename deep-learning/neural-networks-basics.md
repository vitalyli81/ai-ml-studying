# Neural Networks — The Basics

## 1. TL;DR

A neural network is layers of simple math (multiply, add, activate) stacked together. Data goes in one end, a prediction comes out the other. The network learns by making a prediction, measuring how wrong it was, then adjusting its internal numbers (weights) to be less wrong next time. Repeat millions of times. That's it.

---

## 2. The Mental Model

> 💡 **Think of it as an assembly line with quality control.**

A car factory has stations: cut metal → weld frame → install engine → paint → QA check. Each station transforms the car a little bit. At the end, QA compares the result to specs and sends improvement notes back to every station.

- **Assembly station** → Layer (transforms the data)
- **Part going through stations** → Data (the input being transformed)
- **QA result vs. specs** → Loss (how wrong the prediction is)
- **Improvement notes sent backwards** → Backpropagation (adjusting weights)
- **Station settings that change** → Weights (the numbers the network learns)
- **Running the factory 10,000 times** → Training epochs

---

## 3. Why It Exists

**The problem:** Classical ML required you to manually engineer features — you had to decide what to look for (edges, colors, frequencies). For images or language, this was impossibly hard.

**What came before:** Decision trees, SVMs, and logistic regression all needed hand-crafted feature columns. If you missed an important feature, the model failed.

**What changed:** Neural networks learn features automatically. You feed in raw pixels or raw text, and the network figures out what to look for. You design the architecture; it designs the features.

---

## 4. Core Concepts

### The Neuron (Perceptron)

**One-line definition:** A single unit that takes weighted inputs, sums them, and passes through a squishing function.

**Analogy:** A judge scoring a diving competition — multiplies each criteria by its importance, sums the weighted scores, then outputs a final rating.

**Technical explanation:** For inputs x₁, x₂, x₃ with weights w₁, w₂, w₃:

```
z = (x1 × w1) + (x2 × w2) + (x3 × w3) + bias
output = activation_function(z)
```

**Code:**
```python
import torch
import torch.nn as nn

# One neuron: 3 inputs → 1 output
neuron = nn.Linear(3, 1)
x = torch.tensor([1.0, 2.0, 3.0])
output = neuron(x)
```

**Common misconception:** ❌ "Neurons are like real brain neurons" → ✅ They're just a weighted sum + squish. The brain analogy is loose at best.

---

### Weights and Biases

**One-line definition:** Weights are the numbers the network learns; bias is a default offset that shifts the output.

**Analogy:** Weights are the EQ knobs on a mixer — each controls how much a signal contributes. Bias is the volume knob's baseline level.

**Technical explanation:** Every connection between neurons has a weight. During training, weights get adjusted to minimize prediction error. The bias lets the neuron activate even when all inputs are zero.

**Common misconception:** ❌ "Bias is bad — it means the model is biased" → ✅ Bias here is a mathematical term for an offset parameter, not statistical bias.

---

### Layers

**One-line definition:** Groups of neurons stacked in sequence — input → hidden → output.

**Analogy:** A pipeline in Express.js — each middleware transforms the request a bit and passes it on.

```javascript
app.use(parseRawInput)       // Input layer:  raw data
app.use(findPatterns)        // Hidden layer 1: low-level features
app.use(combinePatterns)     // Hidden layer 2: high-level features
app.use(predict)             // Output layer: final answer
```

**Technical explanation:**
- **Input layer** — your raw data (pixels, numbers, word embeddings)
- **Hidden layers** — where features are learned (you choose how many)
- **Output layer** — the prediction (1 neuron for regression, N for N classes)

**Common misconception:** ❌ "More layers always = better" → ✅ More layers = more capacity, but also more risk of overfitting and harder to train. Start small.

---

### The Training Loop

**One-line definition:** The cycle of predict → measure error → adjust weights, repeated until the model is good.

**Analogy:** Like debugging with hot reload: run → see the bug → trace the root cause → fix → repeat.

```
┌─────────────────────────────────────────┐
│  1. FORWARD PASS   → get prediction     │
│  2. LOSS           → measure error      │
│  3. BACKWARD PASS  → find root cause    │
│  4. UPDATE WEIGHTS → apply the fix      │
│  5. REPEAT         → next iteration     │
└─────────────────────────────────────────┘
```

**Common misconception:** ❌ "The network randomly tries values until it works" → ✅ It follows the mathematical gradient — it always knows which direction to adjust each weight.

---

### Epochs, Batches, and Iterations

**One-line definition:** An epoch is one full pass through all training data; a batch is a chunk processed at once.

**Analogy:** Reading a textbook. One epoch = reading the whole book once. A batch = one chapter per sitting. Iterations = total chapter sittings to finish the book.

```python
dataset = 1000       # total samples
batch_size = 100     # items per batch
# 1 iteration = 1 batch (100 samples)
# 1 epoch = 10 iterations = full dataset
# 10 epochs = 100 total iterations
```

**Common misconception:** ❌ "One epoch is enough" → ✅ Networks usually need 10–100+ epochs to converge. Each pass through the data further refines the weights.

---

### Universal Approximation Theorem

**One-line definition:** A neural network with enough neurons can approximate any mathematical function.

**Analogy:** You can approximate any curve with enough line segments. More segments = smoother approximation.

**Technical explanation:** Even a single hidden layer with enough neurons can theoretically approximate any continuous function. In practice, deep networks (many layers) learn the same functions more efficiently with fewer neurons.

**Common misconception:** ❌ "This means neural nets can solve anything" → ✅ It means they CAN represent the function — actually finding it through training is a separate (hard) problem.

---

## 5. How It Actually Works — Step by Step

Let's trace a simple 2-input network predicting house prices:

**Input:** `[size=1500sqft, bedrooms=3]`  
**Goal:** Predict price

```
Step 1: INPUT LAYER
  x = [1500, 3]

Step 2: HIDDEN LAYER 1 (say 4 neurons)
  For each neuron:
    z = x1×w1 + x2×w2 + bias
    h = ReLU(z)              ← activation makes it non-linear
  Output: h = [42, 0, 81, 17]  ← 4 neuron outputs

Step 3: HIDDEN LAYER 2 (say 2 neurons)
  z = h1×w... + h2×w... + ...
  h2 = ReLU(z)
  Output: [124, 35]

Step 4: OUTPUT LAYER (1 neuron)
  price_pred = 124×w1 + 35×w2 + bias
  price_pred = 250,000

Step 5: CALCULATE LOSS
  actual_price = 310,000
  loss = (310,000 - 250,000)² = 3,600,000,000

Step 6: BACKWARD PASS (backpropagation)
  ← Gradients flow from output back through each layer
  ← Each weight gets a number: "change by this much"

Step 7: UPDATE WEIGHTS
  w = w - learning_rate × gradient
  Now predictions will be slightly closer to 310,000

Step 8: REPEAT 10,000 times
  → Loss gets smaller and smaller
  → Predictions get more accurate
```

---

## 6. Code in Practice

### Minimal "Hello World"
```python
import torch
import torch.nn as nn

# Simplest possible network
model = nn.Sequential(
    nn.Linear(2, 4),   # 2 inputs → 4 neurons
    nn.ReLU(),
    nn.Linear(4, 1),   # 4 neurons → 1 output
)

x = torch.tensor([[1500.0, 3.0]])  # size, bedrooms
print(model(x))  # random prediction (untrained)
```

### Practical — Define with Custom Class
```python
class HousePriceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output(x)

model = HousePriceNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"Learnable parameters: {total_params}")
# 2×16+16 + 16×8+8 + 8×1+1 = 297 parameters
```

### Full Training Loop
```python
import torch
import torch.nn as nn

model = HousePriceNet()
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Fake data: 100 houses
X = torch.randn(100, 2)
y = torch.randn(100, 1)

for epoch in range(50):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

---

## 7. Gotchas & Pitfalls

| ❌ Wrong Assumption | ✅ Reality |
|---|---|
| More layers = always better | Too many layers → overfitting + vanishing gradients. Start with 2-3 hidden layers. |
| Random initial weights don't matter | Bad initialization causes vanishing/exploding gradients. PyTorch's defaults are usually fine. |
| Training longer = better model | After a point, training longer just overfits the training data. |
| Bigger batch size is always faster | Very large batches → worse generalization, even if training is faster. |
| Loss going down = everything is good | Loss can decrease while validation accuracy stagnates — monitor both. |
| Neural networks need GPU always | Small networks (< 1M params) train fine on CPU. GPU matters for big models. |
| One pass through data is enough | Networks need many epochs (10–200+) to converge. |

---

## 8. When to Use / When NOT to Use

**Use neural networks when:**
- Your data is unstructured (images, audio, text, video)
- You have thousands+ samples
- The pattern is complex and non-linear
- Hand-engineering features would take months
- You have access to a GPU for training

**Do NOT use neural networks when:**
- You have small tabular data (< 1000 rows) — use XGBoost instead
- You need full explainability (medical, legal decisions)
- You have a deadline in hours — classical ML trains faster
- You don't have compute — a decision tree can run on a Raspberry Pi
- The pattern is actually simple — a logistic regression might be all you need

---

## 9. Related Concepts (The Map)

- **Backpropagation** — the algorithm that computes gradients for weight updates; it's what makes training possible (see `backpropagation.md`)
- **Activation functions** — the "squish" functions (ReLU, sigmoid) that add non-linearity; without them, you just have linear regression (see `activation-functions.md`)
- **Loss functions** — how you measure "how wrong" the prediction is; different tasks use different losses (see `loss-functions-optimizers.md`)
- **CNNs** — neural networks with a special layer type designed for spatial data (images); same core principles, different layer arrangement (see `cnn.md`)
- **Regularization** — techniques to prevent the network from just memorizing training data (see `regularization.md`)

---

## 10. Cheat Sheet

| Term | One-Line Definition |
|---|---|
| **Neuron** | Weighted sum of inputs + activation function |
| **Weight** | Learned number controlling connection strength |
| **Bias** | Offset that lets neurons activate independently of inputs |
| **Layer** | Group of neurons that transform data together |
| **Epoch** | One full pass through the entire training dataset |
| **Batch** | Subset of data processed per gradient update |
| **Loss** | How wrong the predictions are (lower = better) |
| **Gradient** | Direction + magnitude to adjust each weight |
| **Inference** | Using a trained model to make predictions (no learning) |

**Core pattern:**
```
input → [Linear → Activation] × N → output
```

**Remember these 3 things:**
1. A neural network is just layers of weighted sums + activations
2. Training = forward pass → loss → backward pass → weight update, repeated
3. More data + right architecture + right hyperparameters = better model

---

## 11. Self-Check Questions

1. What are the three types of layers in a neural network, and what does each do?
2. Why do we need activation functions? What happens without them?
3. What is one epoch? How does it differ from one iteration?
4. If your training loss is 0.01 but validation loss is 2.5, what's happening?
5. Why would you choose XGBoost over a neural network for tabular data?

<details>
<summary>Brief Answers</summary>

1. **Input** (receives raw data), **hidden** (learns features through weights + activations), **output** (produces the final prediction). Hidden layers are where the actual learning happens.

2. Without activation functions, any number of layers collapses into a single linear transformation — you can only model straight-line relationships. Activations (ReLU, sigmoid) introduce non-linearity so the network can learn curves and complex patterns.

3. One **epoch** = one complete pass through every sample in the training set. One **iteration** = one forward + backward pass on a single batch. If you have 1000 samples and batch size 100, one epoch = 10 iterations.

4. **Overfitting** — the model memorized the training data instead of learning general patterns. The gap between training and validation loss is the red flag. Fix: add regularization (dropout, weight decay), get more data, or reduce model size.

5. On structured/tabular data, XGBoost typically outperforms neural networks and trains orders of magnitude faster with less data. Neural networks need large datasets to shine and are better suited to unstructured data (images, text, audio).

</details>

---

## 12. Go Deeper

- **3Blue1Brown — Neural Networks series** (YouTube): The best visual intuition for how networks learn. Watch chapters 1-4 before writing any code. [Why it's worth it: the gradient descent visualization alone is worth 20 hours of reading.]

- **fast.ai Practical Deep Learning** (course.fast.ai): Top-down, code-first approach. You build real things before diving into theory. Perfect for your background as a developer. [Why: gets you productive fast, then fills in the math.]

- **PyTorch official tutorials** (pytorch.org/tutorials): The "60-minute blitz" is the canonical first PyTorch tutorial. Solid, accurate, well-maintained. [Why: you'll use PyTorch for everything — learn it from the source.]

- **"Deep Learning" by Goodfellow, Bengio, Courville** (deeplearningbook.org): Free online. Chapter 6 covers feedforward networks with rigorous math. [Why: the reference book — go here when you need precise answers.]

- **Andrej Karpathy — micrograd** (github.com/karpathy/micrograd): 100-line implementation of a neural network from scratch. Reading this makes backpropagation click permanently. [Why: nothing beats building it yourself to understand it.]
