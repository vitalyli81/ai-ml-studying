# CNN (Convolutional Neural Networks)

## What Is It?

A CNN is a neural network designed for **visual data** (images, video). Instead of looking at every pixel independently, it slides small filters across the image to detect **patterns** — edges, textures, shapes, and eventually whole objects.

Think of it like scanning an image with a magnifying glass, looking for specific patterns at each position.

## Frontend Analogy — CSS Filters

You already know image filters from CSS/Canvas:

```css
/* CSS applies filters to the whole image */
filter: blur(5px);        /* smoothing filter */
filter: contrast(200%);   /* edge-enhancing filter */
filter: grayscale(100%);  /* color reduction filter */
```

A CNN learns its own filters, but instead of blur or contrast, it learns **pattern detectors**:

```
Filter 1: detects horizontal edges  ───
Filter 2: detects vertical edges    │
Filter 3: detects diagonal edges    ╱
Filter 4: detects corners           ┐
...
Filter 64: detects dog ears         🐕
```

The magic: **the network figures out which filters to use on its own** during training.

## How It Works (Step by Step)

### 1. Convolution — The Sliding Filter

A small filter (e.g., 3x3) slides across the image, computing a dot product at each position:

```
Image (5x5):                Filter (3x3):
┌─────────────────┐         ┌─────────┐
│ 1  0  1  0  1   │         │ 1  0  1 │
│ 0  1  0  1  0   │    *    │ 0  1  0 │  =  Output (3x3)
│ 1  0  1  0  1   │         │ 1  0  1 │
│ 0  1  0  1  0   │         └─────────┘
│ 1  0  1  0  1   │
└─────────────────┘

Step 1: Place filter at top-left     Step 2: Slide right by 1
┌───────┐                            ┌───────┐
│1  0  1│ 0  1                       1 │0  1  0│ 1
│0  1  0│ 1  0   → sum = 5          0 │1  0  1│ 0   → sum = 3
│1  0  1│ 0  1                       1 │0  1  0│ 1
 0  1  0  1  0                        0  1  0  1  0
 1  0  1  0  1                        1  0  1  0  1
```

The filter slides across every position, producing a **feature map** — a new smaller image that highlights where the pattern was found.

### 2. Multiple Filters = Multiple Feature Maps

Each filter detects a different pattern:

```
Input Image (1 image)
      ↓
┌─────────────────────────────────────┐
│ Filter 1 → Feature map 1 (edges)    │
│ Filter 2 → Feature map 2 (corners)  │
│ Filter 3 → Feature map 3 (textures) │
│ ...                                 │
│ Filter 32 → Feature map 32          │
└─────────────────────────────────────┘
      ↓
32 feature maps (like 32 different views of the image)
```

### 3. Pooling — Shrink the Image

After convolution, **max pooling** reduces the size by taking the maximum value in each region:

```
Feature map (4x4):          After 2x2 Max Pooling (2x2):
┌────┬────┬────┬────┐       ┌────┬────┐
│ 1  │ 3  │ 2  │ 1  │       │ 6  │ 8  │
├────┼────┼────┼────┤  →    ├────┼────┤
│ 6  │ 2  │ 8  │ 4  │       │ 7  │ 5  │
├────┼────┼────┼────┤       └────┴────┘
│ 3  │ 7  │ 1  │ 5  │
├────┼────┼────┼────┤       Takes the max from each 2x2 block
│ 2  │ 4  │ 3  │ 2  │
└────┴────┴────┴────┘
```

**Why pool?**
- Reduces computation (smaller image)
- Makes the model more robust to small shifts (the cat moved 2 pixels — doesn't matter)
- Focuses on "was the pattern there?" not "exactly where?"

### 4. The Full CNN Architecture

```
Input Image (224×224×3)    ← RGB image
        ↓
[Conv 3×3, 32 filters] + ReLU   ← detect edges
[Max Pool 2×2]                    ← shrink
        ↓ (112×112×32)
[Conv 3×3, 64 filters] + ReLU   ← detect shapes
[Max Pool 2×2]                    ← shrink
        ↓ (56×56×64)
[Conv 3×3, 128 filters] + ReLU  ← detect parts (eyes, ears)
[Max Pool 2×2]                    ← shrink
        ↓ (28×28×128)
[Flatten]                         ← turn 2D into 1D vector
        ↓ (100352)
[Dense 256] + ReLU                ← combine features
[Dense 10] + Softmax              ← classify into 10 classes
        ↓
Output: [cat: 0.85, dog: 0.10, ...]
```

### 5. What Each Layer Learns (The Key Insight)

```
Layer 1 (early):   Simple patterns     ─  │  ╱  ╲
Layer 2:           Combinations         ○  △  ┘  ╳
Layer 3:           Parts               👁  👃  🦻
Layer 4 (deep):    Whole objects        🐱  🐕  🚗
```

Early layers learn **universal features** (edges work for any image). Deeper layers learn **task-specific features** (cat faces vs dog faces). This is why transfer learning works so well (more on that in the transfer learning doc).

## Key Terms

| Term | Frontend Analogy | Meaning |
|------|-----------------|---------|
| **Filter/Kernel** | CSS filter function | Small matrix that detects a pattern (e.g., 3x3) |
| **Feature Map** | Filtered canvas layer | Output of applying one filter to the image |
| **Stride** | Grid gap | How far the filter moves each step (stride=2 = skip every other position) |
| **Padding** | CSS padding | Add zeros around the border so output size = input size |
| **Channels** | RGBA channels | Depth of the image (RGB=3, grayscale=1, feature maps=N) |
| **Pooling** | Image downsample | Reduce spatial size (usually max pooling) |

## Why CNN Instead of Regular Neural Networks?

A 224×224 RGB image = 150,528 pixels. A regular (dense) network would need:

```
Dense: 150,528 inputs × 1024 neurons = 154 million parameters (first layer alone!)
CNN:   32 filters × 3×3×3 = 864 parameters (first layer)
```

**CNNs are efficient because:**
1. **Parameter sharing** — same filter applied everywhere (one edge detector, not 150K different ones)
2. **Local connectivity** — each neuron only looks at a small patch, not the whole image
3. **Translation invariance** — a cat in the top-left is the same as a cat in the bottom-right

## Famous CNN Architectures

| Model | Year | Key Innovation | Layers |
|-------|------|----------------|--------|
| **LeNet** | 1998 | First CNN (handwritten digits) | 5 |
| **AlexNet** | 2012 | Started the deep learning revolution | 8 |
| **VGG** | 2014 | Simple: just stack 3×3 convolutions | 16-19 |
| **ResNet** | 2015 | Skip connections → train 152 layers! | 50-152 |
| **EfficientNet** | 2019 | Best accuracy/efficiency tradeoff | Varies |

You'll rarely build these from scratch — use pretrained versions (see transfer learning doc).

## Python Example

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extraction (convolution layers)
        self.features = nn.Sequential(
            # Block 1: 3 input channels (RGB) → 32 feature maps
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 224→112

            # Block 2: 32 → 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 112→56

            # Block 3: 64 → 128 feature maps
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),        # 56→28
        )

        # Classification (dense layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),               # 128×28×28 → 100352
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)           # extract patterns
        x = self.classifier(x)         # classify
        return x

# Create model and test
model = SimpleCNN(num_classes=10)
dummy_image = torch.randn(1, 3, 224, 224)  # 1 image, 3 channels, 224×224
output = model(dummy_image)
print(f"Output shape: {output.shape}")  # [1, 10] — 10 class scores
```

## When to Use CNNs

| Good For | Bad For |
|----------|---------|
| Image classification | Text (use Transformers) |
| Object detection | Tabular data (use XGBoost) |
| Image segmentation | Small datasets (use transfer learning) |
| Video analysis | When you don't have GPU |
| Medical imaging | |

## Key Takeaway

CNNs learn **visual features automatically** by sliding small filters across images. Early layers find edges, middle layers find shapes, deep layers find objects. The key innovations — parameter sharing, local connectivity, and pooling — make them efficient enough to process images that would be impossible for regular networks. In practice, you'll almost always use a **pretrained CNN** (ResNet, EfficientNet) rather than building from scratch.
