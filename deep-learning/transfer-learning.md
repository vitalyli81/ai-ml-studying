# Transfer Learning

## 1. TL;DR

Transfer learning means taking a model already trained on a massive dataset and adapting it to your specific task. You almost never train from scratch in practice — you grab a pretrained ResNet, BERT, or LLaMA and fine-tune it. This works because early layers learn universal features (edges, grammar) that transfer to any task. The workflow: pick a pretrained model, replace the output head for your task, fine-tune with a small learning rate (1e-5 to 1e-4), done. Training time: minutes instead of weeks.

---

## 2. The Mental Model

> 💡 **Think of it like hiring a senior developer vs. training a bootcamp grad.**

Hiring senior: they already know programming, system design, and debugging — you just need to onboard them to your specific codebase (fine-tuning). A few weeks and they're productive.

Training junior from scratch: great long-term investment, but takes months before they're autonomous, and you need to teach everything from Hello World up.

- **Hiring senior dev** → fine-tuning a pretrained model
- **Dev's existing programming knowledge** → learned features (edges, grammar, semantics)
- **Onboarding to your codebase** → fine-tuning on your specific dataset
- **Junior from bootcamp** → training from scratch
- **Months of mentoring** → weeks of GPU time + millions of labeled samples
- **Senior's résumé of past projects** → model's pretraining on ImageNet / Common Crawl

---

## 3. Why It Exists

**The problem:** Training a competitive image model from scratch requires ImageNet (1.2M labeled images, 1000 classes) and days on 8 GPUs. Training a language model from scratch requires terabytes of text and thousands of GPU-hours. Most teams have neither.

**What came before:** Every project trained its own model from scratch. Data collection was the bottleneck. Small datasets meant poor models.

**What changed:** The key insight (Yosinski et al., 2014): the first layers of any deep network learn **generic features** that are reusable across tasks — edges and textures for vision, syntax and word meaning for NLP. You can reuse these features freely. Combined with the explosion of pretrained models on Hugging Face and torchvision, the barrier to state-of-the-art dropped dramatically. Fine-tuning a pretrained model became the standard workflow overnight.

---

## 4. Core Concepts

### Why Pretrained Features Transfer

**One-line definition:** Early network layers learn universal patterns that are useful for any task in the same domain — they don't need to be relearned.

**Analogy:** A chef who trained at a French culinary school knows knife skills, sauté technique, and flavor balance — all skills that transfer to Italian cooking. They just need to learn Italian-specific recipes.

```
ResNet-50 trained on ImageNet learns:
  Layer 1–3:   Edges, textures, color gradients    ← UNIVERSAL (useful for any image)
  Layer 4–6:   Shapes, patterns, contours          ← MOSTLY UNIVERSAL
  Layer 7–9:   Object parts (ears, wheels, petals) ← SOMEWHAT SPECIFIC
  Layer 10–12: Full objects (dogs, cars, flowers)   ← TASK SPECIFIC (ImageNet classes)

For your cat-vs-dog classifier:
  Keep layers 1–9 (universal → mostly useful)
  Replace layer 10–12 with your 2-class head
```

**Common misconception:** ❌ "Transfer learning only works if your task is similar to the pretraining task" → ✅ Even transferring from ImageNet (natural images) to medical X-rays works surprisingly well. Generic visual features (edges, shapes) are genuinely universal.

---

### Feature Extraction (Freezing)

**One-line definition:** Lock all pretrained weights, only train a new output layer you've added for your task.

**Analogy:** Using a pre-built API — you don't modify the API's internals, you just write code that calls it. The API does the heavy lifting; you handle the task-specific integration.

```
[Pretrained CNN — ALL FROZEN] → [New linear layer — TRAINABLE]

Frozen: extracts features (acts as a fixed function)
New layer: maps those features to your classes
```

```python
model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False      # freeze everything

model.fc = nn.Linear(2048, num_classes)   # only this trains
```

**When to use:** Very little data (< 500 samples per class). Task is similar to pretraining. Fast iteration needed.

**Common misconception:** ❌ "Frozen layers are wasted compute during training" → ✅ Frozen layers still run during the forward pass (needed to compute features), but their gradients aren't computed during backprop — saving compute.

---

### Fine-Tuning

**One-line definition:** Unfreeze some or all pretrained layers and train them with a very small learning rate alongside your new output head.

**Analogy:** Renovating a house — you keep the foundation and structure (frozen early layers), but update the interior design (fine-tuned later layers) for your specific taste.

```
[Early layers — FROZEN] [Later layers — TRAINABLE (tiny lr)] [New head — TRAINABLE]
```

```python
# Unfreeze only later layers
for name, param in model.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

**Critical rule:** Always use a much smaller learning rate for pretrained layers (1e-5 to 1e-4) vs. the new head (1e-3). A large LR destroys the pretrained knowledge — the network "catastrophically forgets" what it learned.

**Common misconception:** ❌ "Fine-tuning = training for more epochs" → ✅ Fine-tuning specifically means updating pretrained weights (not from scratch) with a controlled, small learning rate to adapt rather than overwrite.

---

### Catastrophic Forgetting

**One-line definition:** When fine-tuning with too large a learning rate, the network overwrites its pretrained knowledge and becomes no better than random initialization.

**Analogy:** Teaching a fluent Spanish speaker English using immersion — if too aggressive, they lose their Spanish fluency in the process.

```
Too high LR (1e-2): weights change drastically → pretrained features destroyed
Too low  LR (1e-8): weights barely change → no adaptation to your task
Just right (1e-5):  gentle adaptation → keep universal features, learn task-specific ones
```

**Common misconception:** ❌ "More epochs compensate for a too-large learning rate" → ✅ Catastrophic forgetting happens within the first few steps of a too-large LR update. More epochs make it worse. Fix the LR first.

---

### Differential Learning Rates

**One-line definition:** Use different (smaller) learning rates for earlier layers vs. later layers in the network.

**Analogy:** When renovating a house, you do minor touch-ups to the foundation (small changes) but completely redesign the interior (larger changes).

```
Early layers (universal):  lr = 1e-6  (barely change — already good)
Middle layers:             lr = 1e-5
Later layers:              lr = 1e-4  (more specific to pretraining task — update more)
New output head:           lr = 1e-3  (brand new — needs full learning)
```

```python
optimizer = torch.optim.AdamW([
    {'params': model.layer1.parameters(), 'lr': 1e-6},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3},
])
```

**Common misconception:** ❌ "Differential LRs are only for advanced use" → ✅ This technique consistently improves fine-tuning results and is simple to implement. Worth using as a default.

---

### Domain Adaptation

**One-line definition:** The process of adapting a model from its pretraining domain to a target domain that has a different data distribution.

**Analogy:** A chef trained in France adapting to Japanese cuisine — the cooking techniques transfer, but flavor profiles, ingredients, and plating style need to be learned.

```
Source domain:  Natural photos (ImageNet)  →  Target domain: Satellite images
Source domain:  English Wikipedia           →  Target domain: Medical clinical notes
Source domain:  General text (Common Crawl) →  Target domain: Legal contracts
```

**Common misconception:** ❌ "The further the domains, the more from-scratch training you need" → ✅ Even very different domains benefit from transfer — training from scratch almost always performs worse than starting from a pretrained model, even if the domains seem unrelated.

---

## 5. How It Actually Works — Step by Step

Fine-tuning ResNet-50 for classifying 5 types of skin lesions (500 images total):

```
Step 1: LOAD PRETRAINED MODEL
  model = resnet50(weights='IMAGENET1K_V1')
  → 25M parameters, trained on 1.2M images, 1000 classes
  → Already knows: edges, textures, shapes, object parts

Step 2: REPLACE OUTPUT HEAD
  model.fc = nn.Linear(2048, 5)   # 5 lesion types, not 1000 ImageNet classes
  → Only 2048×5 + 5 = 10,245 NEW parameters added

Step 3: FREEZE STRATEGY (500 images → feature extraction first)
  for param in model.parameters(): param.requires_grad = False
  model.fc.requires_grad_(True)  # only train the new head

Step 4: FIRST PHASE — Feature Extraction (5 epochs)
  optimizer = AdamW(model.fc.parameters(), lr=1e-3)
  → Train just the new head with full pretrained features
  → Validation accuracy: 78% after 5 epochs

Step 5: UNFREEZE + FINE-TUNE (10 more epochs)
  for param in model.layer4.parameters(): param.requires_grad = True
  optimizer = AdamW([
      {'params': model.layer4.parameters(), 'lr': 1e-5},
      {'params': model.fc.parameters(),     'lr': 1e-4},
  ])
  → Gently adapt later layers to medical imaging domain
  → Validation accuracy: 86% after 10 more epochs

Step 6: TOTAL TRAINING TIME
  ≈ 8 minutes on a single GPU
  vs. training ResNet from scratch: days + millions of labeled images
```

---

## 6. Code in Practice

### Minimal — Feature extraction (freeze all, train head only)
```python
import torch.nn as nn
import torchvision.models as models

model = models.resnet50(weights='IMAGENET1K_V1')

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your task
num_classes = 5
model.fc = nn.Linear(2048, num_classes)   # only this has requires_grad=True

optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
```

### Practical — Two-phase fine-tuning
```python
import torchvision.models as models
import torch.nn as nn

# Phase 1: Feature extraction
model = models.efficientnet_b0(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(1280, num_classes)

optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
train(model, optimizer, epochs=5)   # quick warmup

# Phase 2: Fine-tuning (unfreeze + lower LR)
for param in model.features[-3:].parameters():  # unfreeze last 3 blocks
    param.requires_grad = True

optimizer = torch.optim.AdamW([
    {'params': model.features[-3:].parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(),     'lr': 1e-4},
])
train(model, optimizer, epochs=15)
```

### Real-world — NLP fine-tuning with Hugging Face
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
)

# Tokenize dataset
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = raw_dataset["train"].map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    learning_rate=2e-5,          # small LR — critical for fine-tuning
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="best",
)

trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_dataset, eval_dataset=eval_dataset)
trainer.train()
```

---

## 7. Gotchas & Pitfalls

| ❌ Wrong Assumption | ✅ Reality |
|---|---|
| Use the same learning rate as training from scratch | Fine-tuning requires 10–100× smaller LR; large LR causes catastrophic forgetting |
| Train for many epochs (50–100) | Fine-tuning converges in 2–10 epochs; more epochs = overfitting on small datasets |
| Freeze the whole model and only train the head | The head alone is often not enough; unfreeze later layers after a few warmup epochs |
| Forget to use the model's own preprocessor | Pretrained models expect specific normalization (ImageNet mean/std); use the model's transform |
| Training from scratch when a pretrained model exists | Almost always worse and slower; always check for pretrained alternatives first |
| One learning rate for all layers | Use smaller LR for early layers, larger for later layers — differential LR consistently helps |
| Evaluating on training set | You must have a held-out test set that you never trained on to report honest results |

---

## 8. When to Use / When NOT to Use

**Use feature extraction (full freeze) when:**
- Very little data (< 500 samples per class)
- Your target domain is very similar to the pretraining domain
- You need results quickly (< 1 hour)

**Use partial fine-tuning when:**
- Moderate data (500–5000 samples per class)
- Domains are somewhat different (e.g., natural photos → medical imaging)
- This is the most common and recommended approach

**Use full fine-tuning when:**
- Substantial data (5000+ per class)
- Target domain is very different from pretraining
- You have the compute budget

**Train from scratch when:**
- Your domain is fundamentally unlike any available pretrained model
- You have access to a truly massive dataset (millions of samples)
- Rare in practice — check Hugging Face first, there's almost always something usable

---

## 9. Related Concepts (The Map)

- **CNNs** — ResNet, EfficientNet, ViT are the pretrained models you fine-tune for vision tasks; understanding CNN layers helps you know what to freeze/unfreeze (see `cnn.md`)
- **Transformers** — BERT, GPT, LLaMA are the pretrained models for NLP; Hugging Face `transformers` library is the standard access point (see `transformers.md`)
- **Regularization** — fine-tuning on small datasets overfits easily; dropout, weight decay, and early stopping are essential (see `regularization.md`)
- **Loss functions & optimizers** — AdamW with `lr=2e-5` is the standard for NLP fine-tuning; learning rate schedulers (warmup + cosine decay) are commonly used (see `loss-functions-optimizers.md`)
- **Hugging Face Hub** — the central repository of pretrained models; thousands of models for text, vision, audio, multimodal tasks — check it before building anything

---

## 10. Cheat Sheet

**Strategy selection:**
| Data per class | Strategy | Expected gain |
|---|---|---|
| < 500 | Feature extraction (freeze all) | Huge vs scratch |
| 500–5000 | Partial fine-tuning (unfreeze later layers) | Large |
| 5000+ | Full fine-tuning (unfreeze all, tiny LR) | Moderate |
| 100K+ | Consider training from scratch | Domain-specific |

**Learning rates for fine-tuning:**
| Component | Vision | NLP |
|---|---|---|
| Pretrained early layers | 1e-6 | — |
| Pretrained later layers | 1e-5 | 2e-5 |
| New output head | 1e-3 | 2e-4 |

**The fine-tuning recipe:**
```
1. Load pretrained model (torchvision or huggingface)
2. Replace the output layer for your # of classes
3. Freeze early layers; train head for a few epochs (warmup)
4. Unfreeze later layers with 10–100× smaller LR
5. Use early stopping — fine-tuning converges fast
```

**Remember these 3 things:**
1. Always start from pretrained — training from scratch is almost never the right choice
2. Learning rate must be small (1e-5 to 2e-5 for NLP, 1e-4 for vision fine-tuning)
3. Train for only 2–10 epochs — fine-tuning converges fast and overfits quickly

---

## 11. Self-Check Questions

1. Why does transfer learning work? What specifically transfers between tasks?
2. You have 200 images across 4 classes. Which fine-tuning strategy should you use and why?
3. What is catastrophic forgetting and how do you prevent it?
4. You fine-tune BERT on customer reviews and get 95% training accuracy but 62% validation accuracy. What's happening and how do you fix it?
5. What's the difference between `model.fc = nn.Linear(2048, 5)` and continuing to train the whole model?

<details>
<summary>Brief Answers</summary>

1. Transfer learning works because the first layers of deep networks learn **generic, reusable features** — not task-specific knowledge. For vision: edges, textures, shapes (useful for any image task). For NLP: word meaning, syntax, grammar (useful for any language task). These low-level features take enormous compute to learn from scratch, but once learned they transfer freely. The final layers are task-specific and need to be retrained; the early layers are kept as-is.

2. **Feature extraction (freeze all pretrained layers, only train the new output head)**. With only 200 images (~50 per class), any unfreezing risks overfitting — the pretrained features are already good, and you don't have enough data to fine-tune them meaningfully. The pretrained model's features map your images to a good representation; your new linear head just needs to learn the 4-class decision boundary from 200 examples.

3. **Catastrophic forgetting** is when fine-tuning with too large a learning rate causes the network to drastically overwrite its pretrained weights, losing the useful representations it learned during pretraining. Prevention: (1) Use a small learning rate (1e-5 to 2e-5 for NLP, 1e-4 for vision) — small updates adapt without destroying. (2) Use differential learning rates — even smaller LR for early layers that contain the most universal features. (3) Warm up by training only the new head first before unfreezing pretrained layers.

4. **Overfitting** on the small review dataset. The model has memorized training examples rather than learning general sentiment patterns. Fixes: (a) reduce epochs (fine-tuning typically only needs 3–5); (b) add dropout or stronger weight decay; (c) reduce the learning rate further; (d) collect or augment more training data; (e) use early stopping with the validation set.

5. `model.fc = nn.Linear(2048, 5)` **replaces** the final classification layer with a new randomly initialized one for your 5 classes — the pretrained 1000-class head is discarded. Without this, continuing to train would optimize for the original 1000 ImageNet classes. With it, only the 5-class head has `requires_grad=True` by default (if you've frozen the rest), so backprop only updates those 10,245 parameters. The rest of the network's 25M parameters stay frozen and serve as a fixed feature extractor.

</details>

---

## 12. Go Deeper

- **"How transferable are features in deep neural networks?" (Yosinski et al., 2014)**: The foundational paper proving that early layers are universal and later layers are specific. Includes the experiments showing transfer between different image tasks. [Why: this is the theoretical backbone of why transfer learning works — reading it makes "freeze early layers" feel principled rather than arbitrary.]

- **Hugging Face documentation — Fine-tuning** (huggingface.co/docs/transformers/training): The official guide to fine-tuning any model with the `Trainer` API. Covers data prep, training args, and evaluation. [Why: this is the most used API in NLP fine-tuning — knowing it cold is essential for your role as an AI engineer.]

- **fast.ai — Transfer Learning chapter**: Jeremy Howard's approach: always start from a pretrained model, fine-tune with discriminative learning rates, use the 1-cycle policy. Practical, opinionated, and highly effective. [Why: the best practitioner-level fine-tuning advice; fast.ai won ImageNet competitions using these techniques.]

- **"Universal Language Model Fine-Tuning (ULMFiT)" (Howard & Ruder, 2018)**: Introduced the concept of fine-tuning language models with differential learning rates and gradual unfreezing. The template for all NLP transfer learning. [Why: understanding ULMFiT explains why we fine-tune NLP models the way we do today — BERT fine-tuning is based on these ideas.]

- **Papers With Code — Image Classification Benchmarks** (paperswithcode.com/task/image-classification): Shows the best pretrained models by accuracy on standard benchmarks. [Why: before picking a model, check which ones are state-of-the-art and available for your image size and compute budget.]
