# CNN (Convolutional Neural Networks)

## 1. TL;DR

A CNN is a neural network designed for visual data. Instead of connecting every pixel to every neuron (impossibly expensive), it slides small learned filters across the image to detect patterns — edges → shapes → parts → objects. It's dramatically more efficient than a regular network for images. In practice you'll almost never build one from scratch — you'll fine-tune a pretrained ResNet or EfficientNet on your data.

---

## 2. The Mental Model

> 💡 **Think of it as a factory inspection line with specialized cameras.**

A car factory QA line has different cameras: one checks for dents, one checks paint consistency, one checks part alignment. Each camera is small and focused — it doesn't see the whole car at once, just a small region. The results from all cameras combine to produce a final quality verdict.

- **Factory inspection camera** → convolutional filter (small, focused)
- **Camera sliding across the car surface** → filter sliding across the image
- **One camera type looking for dents** → one filter detecting a specific pattern (e.g., edges)
- **Many different camera types** → many different filters (32, 64, 128...)
- **Camera result sheet per region** → feature map (where was the pattern found?)
- **Final quality verdict** → output class prediction

---

## Build the Intuition From Zero

The one idea to truly get is **convolution: what "sliding a filter" actually computes, and why sharing one small filter everywhere is the breakthrough.** Let's build it on a tiny grid.

### Idea 1: A filter is a small pattern-matcher that slides

A **filter** (or kernel) is a tiny grid of numbers — say 3×3 — that represents a pattern to look for, like a vertical edge. You slide it over every position of the image, and at each spot you **multiply-and-sum**: overlap the filter, multiply each pair, add them up. A big result means "the pattern is here."

```
Image patch (bright left, dark right)   Vertical-edge filter      Overlap, multiply, sum:
   ┌──┬──┬──┐                              ┌──┬──┬──┐
   │ 9│ 9│ 0│                              │+1│ 0│−1│           (9×1)+(9×0)+(0×−1)
   │ 9│ 9│ 0│        slide over     ×      │+1│ 0│−1│      =    +(9×1)+(9×0)+(0×−1)
   │ 9│ 9│ 0│                              │+1│ 0│−1│           +(9×1)+(9×0)+(0×−1)
   └──┴──┴──┘                              └──┴──┴──┘           = 27  → "STRONG edge here!"
```

Slide the same filter to a flat region (all 9s) and the +1s and −1s cancel → result ≈ 0 → "no edge here." So one filter produces a **feature map**: a grid showing *where* its pattern appears in the image. Different filters detect different things; the network *learns* the filter numbers during training rather than you designing them.

### Idea 2: Why sharing one filter is the whole point

Here's the breakthrough. A dense network would learn a *separate* weight for every pixel position — millions of them — and a cat detector trained on cats in the top-left would have no idea what to do with a cat in the bottom-right. A CNN uses the **same small filter at every position**:

```
Dense layer:   a separate weight per pixel  → 150,000+ params, position-blind
CNN filter:    ONE 3×3 filter reused everywhere → 9 params, finds the pattern ANYWHERE
```

This buys two huge things at once: **far fewer parameters** (9 numbers, not millions) and **translation invariance** (an edge detector works in every corner of the image, because it's literally the same detector slid everywhere). That's why CNNs scaled to real images when dense nets couldn't.

### Idea 3: Stacking builds a hierarchy of meaning

Stack convolution layers and the patterns compose — exactly the "simple parts → complex parts" idea from [neural-networks-basics.md](neural-networks-basics.md), but spatial:

```
Layer 1 filters → edges & color blobs      ("there's a diagonal line here")
Layer 2 filters → corners, curves, texture (combinations of edges)
Layer 3 filters → eyes, wheels, fur        (combinations of those)
Layer 4+        → whole objects: "cat"
```

Early layers see tiny regions; deeper layers (after pooling shrinks the map) effectively see larger and larger areas, so the network goes from "edge" to "cat" through depth. The convolution, pooling, and feature-map sections below formalize these three ideas — and in practice you'll fine-tune a pretrained CNN ([transfer-learning.md](transfer-learning.md)) rather than learn all these filters yourself.

---

## 3. Why It Exists

**The problem:** A 224×224 RGB image has 150,528 pixel values. A fully-connected (dense) network's first layer connecting to 1024 neurons would need 154 million parameters — just for the first layer. Impossible to train efficiently. Also, a dense layer treats `pixel[0,0]` and `pixel[223,223]` as completely unrelated — it doesn't know they're spatially adjacent.

**What came before:** Dense networks applied to raw pixel values. They couldn't scale to real images and had no concept of spatial structure.

**What changed:** LeCun's LeNet (1989) introduced convolution for neural networks — sharing the same small filter across all positions. One edge detector works everywhere in the image. This reduced parameters from millions to thousands and gave the network spatial awareness. AlexNet (2012) proved deep CNNs beat everything on ImageNet and started the deep learning revolution.

---

## 4. Core Concepts

### Convolution (The Sliding Filter)

**One-line definition:** A small matrix (filter) slides across the image, computing a dot product at each position to detect a specific pattern.

**Analogy:** A magnifying glass with a pattern template — you slide it across the image and wherever the template matches, the score is high.

**Technical explanation:** A 3×3 filter slides across the image with a stride of 1. At each position, multiply filter values by image values element-wise and sum all 9 products — this produces **one number** in the output feature map.

```
Image patch:     Filter (diagonal-edge detector):
1  0  1          1  0  -1
0  1  0    ⊙     0  0   0     (⊙ = element-wise multiply)
1  0  1         -1  0   1

Step 1 — element-wise multiply:
  (1·1) + (0·0) + (1·-1)     =  1 + 0 - 1  =  0
+ (0·0) + (1·0) + (0·0)      =  0 + 0 + 0  =  0
+ (1·-1) + (0·0) + (1·1)     = -1 + 0 + 1  =  0

Step 2 — sum them all  →  output pixel = 0
(This patch is symmetric, so a diagonal-edge filter produces zero —
 exactly what we want: no diagonal edge here.)
```

> 💡 **Output size formula:** For an input of spatial size `W`, kernel `K`, padding `P`, stride `S`:
> `output_size = ⌊(W - K + 2P) / S⌋ + 1`
>
> Common setup: `K=3, P=1, S=1` → output same size as input ("same" padding).
> `K=3, P=0, S=1` → output shrinks by 2 per side.
> `K=3, P=1, S=2` → output halves (downsampling via stride).

**Common misconception:** ❌ "The filters are handcrafted" → ✅ The filter values (weights) are *learned* during training via backpropagation, just like any other weights.

---

### Feature Maps

**One-line definition:** The output image produced by applying one filter across the entire input — a map of "where was this pattern found?"

**Analogy:** A heat map overlay — bright spots show where the pattern (e.g., a horizontal edge) was detected strongly.

```
Input image (5×5)  →  [Apply edge filter]  →  Feature map (3×3)
                                               High values = edge found here
                                               Low values  = no edge here
```

**Common misconception:** ❌ "One filter produces one number" → ✅ One filter produces a full feature map (one value per position the filter visited).

---

### Multiple Filters = Multiple Channels

**One-line definition:** Apply N different filters in parallel → N feature maps → N channels of learned patterns.

**Analogy:** N different camera types scanning the same scene simultaneously — each one looking for something different.

```
Input image (1 image)
       ↓
Filter 1 → Feature map 1  (horizontal edges)
Filter 2 → Feature map 2  (vertical edges)
Filter 3 → Feature map 3  (diagonal patterns)
...
Filter 32 → Feature map 32

Output: 32 feature maps stacked = image with 32 channels
```

**Common misconception:** ❌ "More filters always helps" → ✅ More filters = more capacity but more compute. Early layers: 32-64 filters. Later layers: 128-512.

---

### Pooling

**One-line definition:** Reduce the spatial size of feature maps by keeping only the most important value per region.

**Analogy:** Thumbnail generation — you shrink a 1000×1000 image to 100×100 by sampling. Max pooling keeps the strongest signal from each region.

```
Feature map (4×4):          After 2×2 Max Pooling → (2×2):
┌────┬────┬────┬────┐        ┌────┬────┐
│  1 │  3 │  2 │  1 │        │  6 │  8 │   ← max of each 2×2 block
├────┼────┼────┼────┤  →     ├────┼────┤
│  6 │  2 │  8 │  4 │        │  7 │  5 │
├────┼────┼────┼────┤        └────┴────┘
│  3 │  7 │  1 │  5 │
├────┼────┼────┼────┤
│  2 │  4 │  3 │  2 │
└────┴────┴────┴────┘
```

**Why it helps:** Smaller maps = less compute. Also adds translation invariance — a cat shifted 2 pixels still activates the same pooled value.

**Common misconception:** ❌ "Pooling loses important information" → ✅ Pooling discards *where exactly* the pattern was, but keeps *whether* it was there — which is usually what you want for classification.

---

### Parameter Sharing

**One-line definition:** The same filter weights are used at every position in the image — massively reducing the number of parameters.

**Analogy:** One spell-checker rule ("fix this typo") applied to the entire document — not a separate rule for each word position.

```
Dense layer:  150,528 inputs × 1024 neurons = 154M params (first layer alone)
Conv layer:   32 filters × 3×3×3 = 864 params (first layer)

Same expressive power. 178,000× fewer parameters.
```

**Common misconception:** ❌ "Fewer parameters = less powerful" → ✅ Parameter sharing enforces a useful inductive bias: patterns should be detectable regardless of position.

---

### Stride and Padding

**One-line definition:** Stride controls how far the filter jumps each step; padding adds zeros around the border to control output size.

**Analogy:** Stride = how many steps you skip when reading a page. Padding = adding blank margins so you can read edge content.

```python
# stride=1: filter moves 1 pixel at a time (default, full detail)
# stride=2: filter jumps 2 pixels (halves the output size, faster)
# padding=1 with 3×3 filter: output same size as input ("same" padding)
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
```

**Common misconception:** ❌ "Stride=2 is the same as pooling" → ✅ Strided convolution learns how to downsample; max pooling always takes the max. Modern networks increasingly prefer strided conv over explicit pooling.

---

## 5. How It Actually Works — Step by Step

Classifying a 28×28 grayscale image (e.g., handwritten digit):

```
Step 1: INPUT
  Image shape: [1, 28, 28]  → 1 channel (grayscale), 28×28 pixels

Step 2: CONV LAYER 1 (8 filters, 3×3, padding=1)
  Each filter slides across the 28×28 image
  Output: [8, 28, 28]  → 8 feature maps, same spatial size (padding=1)
  → Detects: edges, simple patterns
  Apply ReLU: negative values → 0

Step 3: MAX POOL (2×2)
  Output: [8, 14, 14]  → halved spatial size, kept all 8 channels

Step 4: CONV LAYER 2 (16 filters, 3×3, padding=1)
  Input: [8, 14, 14], output: [16, 14, 14]
  → Detects: combinations of edges (curves, corners)
  Apply ReLU

Step 5: MAX POOL (2×2)
  Output: [16, 7, 7]

Step 6: FLATTEN
  [16, 7, 7] → [784]  → 1D vector for the classifier

Step 7: DENSE LAYER (784 → 10)
  Linear layer: 784 inputs → 10 outputs (one per digit class)
  Output: [10]  → raw scores (logits)

Step 8: LOSS (CrossEntropyLoss)
  Compare logits to true label, compute error

Step 9: BACKPROP
  Gradients flow back through dense → pool → conv → pool → conv
  Filter weights update to detect better patterns
```

---

## 6. Code in Practice

### Minimal — Single conv layer
```python
import torch
import torch.nn as nn

# One conv layer: 3 input channels (RGB), 32 filters, 3×3 kernel
conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

img = torch.randn(1, 3, 224, 224)  # batch=1, RGB, 224×224
output = conv(img)
print(output.shape)  # [1, 32, 224, 224] — 32 feature maps, same spatial size
```

### Practical — Full CNN for image classification
```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: RGB → 32 feature maps, downsample
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 224 → 112

            # Block 2: 32 → 64 feature maps, downsample
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 112 → 56

            # Block 3: 64 → 128 feature maps, downsample
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 56 → 28
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = SimpleCNN(num_classes=10)
x = torch.randn(4, 3, 224, 224)   # batch of 4 images
print(model(x).shape)              # [4, 10]
```

### Real-world — Use a pretrained model (recommended)
```python
import torchvision.models as models

# Load ResNet-50 pretrained on ImageNet
model = models.resnet50(weights='IMAGENET1K_V1')

# Adapt for your number of classes
model.fc = nn.Linear(2048, num_your_classes)

# Fine-tune with a small learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

---

## 7. Gotchas & Pitfalls

| ❌ Wrong Assumption | ✅ Reality |
|---|---|
| Building CNN from scratch is the normal workflow | In practice, always start from a pretrained model (ResNet, EfficientNet) — it's faster and better |
| More conv layers always helps | Each layer halves spatial size (with pooling); too many layers → feature maps become 1×1 |
| Input image size doesn't matter | CNNs have a minimum input size based on how many times you pool; check your architecture |
| Padding is optional | Without padding, each 3×3 conv shrinks the image by 2px per side — stack 5 convs and you lose 10px |
| Conv2d input needs flattening first | NO — Conv2d expects [batch, channels, height, width]; flatten is only before the dense layers |
| Filter weights are handcrafted | They're learned during training via backprop, just like any other weights |
| CNNs work for text | Use Transformers for text. CNNs work for spatial/grid data (images, audio spectrograms) |

---

## 8. When to Use / When NOT to Use

**Use CNNs when:**
- Your data has spatial structure (images, video frames, audio spectrograms)
- You need to detect patterns regardless of where they appear (translation invariance)
- You're working on image classification, object detection, or segmentation
- You have limited compute — CNNs are efficient per parameter

**Do NOT use CNNs when:**
- Your data is tabular/structured → use XGBoost or an MLP
- Your task is text/NLP → use Transformers
- Your sequences are non-spatial (time series without grid structure) → use LSTM or Transformer
- You're starting a new vision project from scratch → use pretrained ResNet/EfficientNet via transfer learning instead

---

## 9. Related Concepts (The Map)

- **Transfer learning** — in practice you almost never train a CNN from scratch; you load a pretrained ResNet and fine-tune it (see `transfer-learning.md`)
- **Pooling vs. stride** — strided convolutions (stride=2) are increasingly replacing explicit pooling layers in modern architectures like ResNet
- **Transformers for vision** — Vision Transformers (ViT) treat image patches like tokens; they're replacing CNNs at scale but CNNs still win on small/medium datasets (see `transformers.md`)
- **Regularization** — CNNs overfit too; dropout and data augmentation (flips, crops, color jitter) are essential (see `regularization.md`)
- **Backpropagation** — gradients flow through conv layers just like dense layers; the filter weights update via the same chain rule (see `backpropagation.md`)

---

## 10. Cheat Sheet

| Term | One-Line Definition |
|---|---|
| **Filter / Kernel** | Small learned matrix (e.g., 3×3) that detects one pattern |
| **Feature map** | Output of applying one filter across the entire image |
| **Channels** | Depth of a tensor (RGB=3, after conv=N filters) |
| **Stride** | How many pixels the filter moves each step (stride=2 halves output size) |
| **Padding** | Zeros added around image border to preserve spatial size |
| **Max pooling** | Keep the max value in each spatial region — shrinks the map |
| **Flatten** | Convert [C, H, W] feature maps to a 1D vector for dense layers |
| **Parameter sharing** | Same filter weights reused at every position — huge efficiency gain |

**Standard CNN block pattern:**
```
Conv2d → ReLU → MaxPool2d  (repeat N times)
→ Flatten → Linear → ReLU → Linear (classifier)
```

**Remember these 3 things:**
1. Filters are learned, not handcrafted — backprop figures out what to detect
2. Early layers detect edges; deep layers detect objects — features get increasingly abstract
3. In practice: load pretrained ResNet, replace the final layer, fine-tune — don't build from scratch

---

## 11. Self-Check Questions

1. Why are CNNs dramatically more parameter-efficient than fully-connected networks for images?
2. What does a feature map represent?
3. What does max pooling do, and why is it useful?
4. You have a `[1, 3, 224, 224]` input tensor and apply `Conv2d(3, 64, kernel_size=3, padding=1)`. What is the output shape?
5. Why do early CNN layers detect edges while deep layers detect objects?

<details>
<summary>Brief Answers</summary>

1. Two reasons: **parameter sharing** (the same 3×3 filter is applied at every position — 9 weights covers the whole image instead of one weight per pixel per neuron) and **local connectivity** (each neuron only sees a small patch, not the whole image). A 3×3 conv layer with 32 filters has 864 parameters; the equivalent dense layer would have 154 million.

2. A feature map is a 2D grid showing where a specific learned pattern was detected in the input. High values = strong match at that position. Low values = pattern absent. Each filter produces its own feature map — 32 filters → 32 feature maps stacked into a [32, H, W] tensor.

3. Max pooling divides the feature map into small regions (e.g., 2×2) and keeps only the maximum value from each region, halving the spatial dimensions. It's useful because: (a) reduces computation in subsequent layers, (b) adds translation invariance (small shifts in pattern position don't change the max), (c) focuses on "was the pattern there?" not "exactly where?".

4. **[1, 64, 224, 224]**. The batch size stays 1. The 3 input channels become 64 (one per filter). With `padding=1` and `kernel_size=3`, the spatial dimensions stay 224×224 (padding compensates for the filter border).

5. Because learning is compositional: Layer 1 filters see raw pixels — the simplest distinguishable patterns are edges and color gradients. Layer 2 filters see Layer 1's edge maps as input — combining edges makes corners and curves. Layer 3 combines those into parts. Each layer builds on the previous one's abstractions, climbing from pixels → edges → shapes → parts → objects.

</details>

---

## 12. Go Deeper

- **CS231n — Convolutional Neural Networks for Visual Recognition** (cs231n.github.io): Stanford's canonical CNN course. Module 2 covers conv layers with excellent visualizations. [Why: the best written explanation of CNNs — rigorous, visual, and free.]

- **"ImageNet Classification with Deep CNNs" — AlexNet paper (Krizhevsky et al., 2012)**: The paper that started the deep learning revolution. Short and readable. [Why: understanding what AlexNet did differently (ReLU, dropout, GPU training) gives you the "why" behind modern conventions.]

- **fast.ai — Lesson 1** (course.fast.ai): Builds an image classifier using transfer learning in ~10 lines of code. Best hands-on first CNN experience. [Why: you build something real and working immediately, then learn the internals.]

- **PyTorch torchvision models** (pytorch.org/vision/stable/models.html): All pretrained models available with one import. ResNet, EfficientNet, ViT, and more. [Why: this is your starting point for any real vision project — bookmark it.]

- **"Visualizing and Understanding Convolutional Networks" (Zeiler & Fergus, 2013)**: The paper that showed what CNN filters actually learn at each layer (edges → textures → objects). Includes beautiful visualizations. [Why: makes the "what does each layer learn?" intuition concrete and visual.]
