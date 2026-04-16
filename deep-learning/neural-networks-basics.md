# Neural Networks — The Basics

## What Is It?

A neural network is a function that learns patterns by passing data through **layers of connected nodes** (neurons). Each neuron takes inputs, multiplies them by weights, adds them up, and passes the result through an activation function.

Think of it as a **series of filters**: raw data goes in one side, each layer extracts more abstract patterns, and a prediction comes out the other side.

## Frontend Analogy — The Pipeline

A neural network is like a **middleware chain** in Express or a series of data transforms:

```javascript
// Middleware chain: each layer transforms the data
app.use(parseRawInput)       // Input layer:  raw pixels, text, numbers
app.use(findEdges)           // Hidden layer 1: low-level patterns
app.use(findShapes)          // Hidden layer 2: combine patterns
app.use(recognizeObject)     // Hidden layer 3: high-level concepts
app.use(sendResponse)        // Output layer: final prediction
```

Each layer takes the previous layer's output, transforms it, and passes it forward. That's literally what a neural network does.

## The Single Neuron (Perceptron)

The simplest unit — one neuron:

```
Inputs        Weights       Sum + Bias      Activation     Output
  x1 ——→ × w1 ——↘
  x2 ——→ × w2 ——→ Σ + b ——→ f(z) ——→ output
  x3 ——→ × w3 ——↗
```

In code:

```python
z = (x1 * w1) + (x2 * w2) + (x3 * w3) + bias
output = activation_function(z)
```

That's it. A neuron is just:
1. **Multiply** each input by a weight
2. **Sum** them up and add a bias
3. **Pass through** an activation function

## Building a Network — Layers

Stack neurons into layers, connect layers together:

```
INPUT LAYER      HIDDEN LAYER 1     HIDDEN LAYER 2     OUTPUT LAYER
(your data)      (learned features) (combined features) (prediction)

  [x1] ——————→ [n1] ——————————→ [n4] ——————————→ [output1]
       ╲      ╱    ╲          ╱    ╲
  [x2] ——→——→ [n2] ——————————→ [n5] ——————————→ [output2]
       ╱      ╲    ╱          ╲    ╱
  [x3] ——————→ [n3] ——————————→ [n6] ——————————→ [output3]

  3 features    3 neurons         3 neurons        3 classes
```

- **Input layer** — your raw data (pixels, numbers, words)
- **Hidden layers** — where the learning happens (you pick how many)
- **Output layer** — the prediction (1 neuron for regression, N neurons for N classes)

Every neuron in one layer connects to every neuron in the next. That's why it's called a **fully connected** (or dense) layer.

## What Does Each Layer Learn?

Imagine a face recognition network:

```
Layer 1: Learns edges and simple patterns      ╱ ╲ — |
Layer 2: Combines edges into shapes            ○ △ □
Layer 3: Combines shapes into parts            👁 👃 👄
Layer 4: Combines parts into faces             🧑 👩 🧔
```

Each layer builds on the previous one — from simple to complex. This is why deep networks (many layers) can learn sophisticated patterns.

## The Training Loop (How It Learns)

The network learns by repeating this cycle thousands of times:

```
┌─────────────────────────────────────────┐
│  1. FORWARD PASS                        │
│     Feed data through the network       │
│     Get a prediction                    │
│                                         │
│  2. CALCULATE LOSS                      │
│     Compare prediction to correct answer│
│     "How wrong are we?"                 │
│                                         │
│  3. BACKWARD PASS (Backpropagation)     │
│     Calculate how each weight            │
│     contributed to the error            │
│                                         │
│  4. UPDATE WEIGHTS                      │
│     Adjust weights to reduce error      │
│     (using gradient descent)            │
│                                         │
│  5. REPEAT until loss is small enough   │
└─────────────────────────────────────────┘
```

Frontend analogy — it's like **hot module reloading for weights**:
1. Run the app (forward pass)
2. See the bug (calculate loss)
3. Find which code caused it (backpropagation)
4. Fix the code (update weights)
5. HMR reloads (next iteration)

## Key Vocabulary

| Term | Frontend Analogy | Meaning |
|------|-----------------|---------|
| **Epoch** | Full test suite run | One pass through the entire training dataset |
| **Batch** | Chunk of API responses | Subset of data processed at once (e.g., 32 samples) |
| **Learning rate** | Step size in animation | How much weights change per update (too big = chaos, too small = stuck) |
| **Loss** | Error count in console | How wrong the predictions are (lower = better) |
| **Weights** | Config values | The numbers the network learns (what gets updated) |
| **Bias** | Default value | An extra number added at each neuron (like a baseline) |
| **Inference** | Production request | Using a trained model to make predictions (no learning) |

## Epochs, Batches, and Iterations

This confuses everyone. Here's the breakdown:

```
Dataset: 1000 samples
Batch size: 100

1 iteration  = process 1 batch (100 samples)
1 epoch      = process all batches (10 iterations = 1000 samples)
10 epochs    = go through the full dataset 10 times (100 iterations total)
```

Think of it like pagination:
```javascript
const dataset = 1000;      // total items
const batchSize = 100;     // items per page
const iterations = 10;     // pages per full scroll
const epochs = 10;         // how many times you re-read everything
```

## Why Do We Need Hidden Layers?

Without hidden layers, a neural network is just linear regression — it can only draw straight lines. Hidden layers let it learn **non-linear patterns**:

```
1 layer (no hidden):     Can learn: straight lines / flat planes
2 layers (1 hidden):     Can learn: curves and simple shapes
3+ layers (2+ hidden):   Can learn: almost any pattern
```

This is the **universal approximation theorem**: a neural network with enough neurons can approximate any function. More layers = more complex patterns.

## How Many Layers / Neurons?

No exact formula, but rules of thumb:

| Task | Architecture |
|------|-------------|
| Simple tabular data | 1-2 hidden layers, 32-128 neurons |
| Image classification | Use CNN (next doc) |
| Text/sequence data | Use RNN or Transformer (later docs) |
| Complex patterns | 3-5 hidden layers, 128-512 neurons |

**Start small, increase if underfitting.** Too many neurons = overfitting + slow training.

## Python Example (from scratch feel, using PyTorch)

```python
import torch
import torch.nn as nn

# Define a simple network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 16)    # 3 inputs → 16 neurons
        self.layer2 = nn.Linear(16, 8)    # 16 → 8 neurons
        self.layer3 = nn.Linear(8, 1)     # 8 → 1 output
        self.relu = nn.ReLU()             # activation function

    def forward(self, x):
        x = self.relu(self.layer1(x))     # layer 1 + activation
        x = self.relu(self.layer2(x))     # layer 2 + activation
        x = self.layer3(x)                # output (no activation for regression)
        return x

# Create the model
model = SimpleNet()
print(model)

# Count parameters (weights + biases)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total learnable parameters: {total_params}")
# 3×16 + 16 + 16×8 + 8 + 8×1 + 1 = 48 + 16 + 128 + 8 + 8 + 1 = 209 parameters
```

## The Big Picture

```
Classical ML:  You design features → Algorithm learns patterns
Deep Learning: Raw data → Network learns features AND patterns

ML:            [Handcrafted features] → [Simple model] → Prediction
Deep Learning: [Raw data] → [Layer][Layer][Layer] → Prediction
```

The magic of deep learning: **you don't have to tell it what to look for**. It figures out the important features on its own.

## Key Takeaway

A neural network is just **layers of simple math operations** (multiply, add, activate) stacked together. Each layer transforms data into slightly more useful representations. Training adjusts the weights so these transformations produce correct predictions. Everything else in deep learning (CNNs, RNNs, Transformers) is just clever variations on how these layers are arranged.
