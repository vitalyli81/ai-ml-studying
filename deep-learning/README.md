# Deep Learning

## What Is Deep Learning?

Deep learning is a **subset of machine learning** that uses neural networks with many layers (hence "deep"). Instead of you designing features by hand, the network learns to extract features automatically from raw data — pixels, words, audio waves.

```
Artificial Intelligence  (broad field)
  └── Machine Learning   (learns from data)
       └── Deep Learning  (neural networks with many layers)
```

### The Key Difference from Classical ML

```
Classical ML (e.g., Random Forest):
  Raw Image → [YOU manually extract features] → Pixel histograms → Model → "cat"
                                                 Color averages
                                                 Edge counts
              ↑ You decide what matters

Deep Learning (e.g., CNN):
  Raw Image → [Network learns features automatically] → "cat"
              Layer 1: edges
              Layer 2: textures
              Layer 3: shapes
              Layer 4: objects
              ↑ The model decides what matters
```

With ML, you're the feature engineer. With deep learning, you're the **architect** — you design the network, and it figures out the rest.

### Frontend Analogy

```javascript
// Classical ML: like manually building a webpack config
// You choose every loader, plugin, and optimization rule

// Deep Learning: like using Vite
// You give it the raw source files, it figures out
// how to bundle, split, and optimize automatically
// More powerful, but you need a beefy machine (GPU) to run it
```

## When Deep Learning Wins (and When It Doesn't)

| Use Deep Learning | Use Classical ML |
|-------------------|-----------------|
| Images, video | Spreadsheet/tabular data |
| Text, NLP | Small datasets (< 1000 samples) |
| Audio, speech | When you need explainability |
| Games, robotics | When you need fast training |
| When you have lots of data | When you don't have a GPU |
| When patterns are complex | When patterns are simple |

**Important:** Deep learning is NOT always better. On a Kaggle tabular competition, XGBoost consistently beats neural networks. Use the right tool for the job.

## The Building Blocks

A deep learning model is made of **layers** stacked together. Each layer transforms the data a little bit, extracting increasingly abstract features:

```
Input (raw data)
  ↓
[Layer 1] → low-level patterns     (edges, simple sounds, common words)
  ↓
[Layer 2] → mid-level patterns     (shapes, phonemes, phrases)
  ↓
[Layer 3] → high-level patterns    (objects, words, meaning)
  ↓
[Layer N] → abstract concepts      (faces, sentences, intent)
  ↓
Output (prediction)
```

The "deep" in deep learning refers to having **many layers** — modern models have dozens to hundreds.

## The Three Major Architectures

### 1. CNN (Convolutional Neural Networks) — For Images

Slides small filters across images to detect visual patterns.

```
Photo → [detect edges] → [detect shapes] → [detect objects] → "dog" (95%)
```

**Used for:** image classification, object detection, medical imaging, self-driving cars

### 2. RNN/LSTM (Recurrent Neural Networks) — For Sequences

Processes data step-by-step, maintaining memory of what came before.

```
"I" → "love" → "coding" → state carries info about full sentence → "positive"
```

**Used for:** time series, simple text tasks, streaming data
**Mostly replaced by Transformers** for NLP tasks.

### 3. Transformers — For Everything (The Current King)

Processes entire sequences in parallel using attention — every element can look at every other element directly.

```
"The cat sat on the mat" → [all words attend to all words simultaneously] → understanding
```

**Used for:** GPT, Claude, BERT, Stable Diffusion, code generation, translation — basically everything modern in AI.

## How Training Works

Every deep learning model trains the same way:

```
1. FORWARD PASS     Feed data through the network → get prediction
2. LOSS             Compare prediction to correct answer → "how wrong?"
3. BACKWARD PASS    Backpropagation: trace error back through layers
4. UPDATE           Adjust weights to reduce error
5. REPEAT           Do this millions of times
```

```
Loss (error)
  |╲
  |  ╲
  |    ╲____
  |         ╲____
  |              ╲_________    ← training: error decreases over time
  |________________________ Epochs (iterations)
```

## What You Need to Train Deep Learning

| Resource | Classical ML | Deep Learning |
|----------|-------------|---------------|
| **Data** | Hundreds-thousands | Thousands-millions |
| **Hardware** | CPU is fine | GPU needed (NVIDIA, Apple M-series) |
| **Training time** | Seconds-minutes | Minutes-days |
| **Framework** | scikit-learn | PyTorch (primary), TensorFlow |
| **Memory** | Low | High (models can be GBs) |

### GPU Options for Learning

| Option | Cost | Good For |
|--------|------|----------|
| **Google Colab** | Free (limited) | Getting started, small experiments |
| **Kaggle Notebooks** | Free (30h/week GPU) | Competitions, practice |
| **Your Mac (M1/M2/M3)** | Already own it | Small-medium models via MPS |
| **Cloud GPU** (Lambda, AWS) | $1-3/hour | Serious training |

## The Deep Learning Revolution — Why Now?

Deep learning theory existed since the 1980s. Three things changed:

1. **Data** — the internet produced massive datasets (ImageNet, Common Crawl)
2. **Compute** — GPUs made parallel computation affordable
3. **Algorithms** — ReLU, batch norm, Adam, skip connections, attention

```
1986: Backpropagation invented        → works in theory
2012: AlexNet wins ImageNet with GPU   → deep learning actually works!
2017: "Attention Is All You Need"      → Transformers change everything
2022: ChatGPT                          → deep learning goes mainstream
```

## Docs in This Folder

Read in this order — each doc builds on the previous:

### Foundations (Read First)

| # | File | Topic | Why It Matters |
|---|------|-------|---------------|
| 1 | [neural-networks-basics.md](neural-networks-basics.md) | Neurons, layers, training | Everything builds on this |
| 2 | [backpropagation.md](backpropagation.md) | How networks learn | Core of all training |
| 3 | [activation-functions.md](activation-functions.md) | ReLU, sigmoid, softmax | What makes networks non-linear |
| 4 | [loss-functions-optimizers.md](loss-functions-optimizers.md) | MSE, cross-entropy, Adam | How to measure and reduce error |
| 5 | [regularization.md](regularization.md) | Dropout, early stopping | Prevent overfitting |

### Architectures (The Big Three)

| # | File | Topic | Why It Matters |
|---|------|-------|---------------|
| 6 | [cnn.md](cnn.md) | Convolutional Neural Networks | Image/visual understanding |
| 7 | [rnn-lstm.md](rnn-lstm.md) | Recurrent Networks | Sequence understanding (and why Transformers replaced them) |
| 8 | [transformers.md](transformers.md) | Attention & Transformers | Powers GPT, Claude, BERT — the most important architecture |

### Practical Skills

| # | File | Topic | Why It Matters |
|---|------|-------|---------------|
| 9 | [transfer-learning.md](transfer-learning.md) | Fine-tuning pretrained models | How you'll actually use DL in practice (not from scratch) |
| 10 | [pytorch-basics.md](pytorch-basics.md) | PyTorch coding patterns | Hands-on implementation reference |

## What Comes After Deep Learning?

After these docs, you'll be ready for:

- **Phase 5: LLMs & AI Engineering** — prompt engineering, RAG, agents, LLM APIs
- This is where your frontend skills become a **superpower** — building AI-powered products that others can actually use
