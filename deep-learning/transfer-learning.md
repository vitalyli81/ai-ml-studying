# Transfer Learning

## What Is It?

Transfer learning means taking a model that was **already trained on a huge dataset** and adapting it to your specific task. Instead of training from scratch (expensive, needs tons of data), you start with a model that already understands general patterns and just teach it your specific thing.

Think of it like hiring an **experienced developer** instead of training a junior from scratch. They already know programming — you just need to teach them your codebase.

## Frontend Analogy — npm Packages

```javascript
// WITHOUT transfer learning (training from scratch):
// Build your own React from scratch, your own router, your own state manager...
// Months of work. Needs massive effort.

// WITH transfer learning (fine-tuning):
// npm install react react-router zustand
// Import, configure for your use case, done.
// Someone else did the hard work. You customize the last mile.

import { pretrained_model } from 'huge-ai-lab';  // billions of images seen
const my_model = finetune(pretrained_model, my_small_dataset);  // 100 images
```

## Why It Works

A model trained on ImageNet (14 million images, 1000 classes) has learned:

```
Layer 1-3:   Edges, textures, colors        ← universal (works for ANY image)
Layer 4-6:   Shapes, patterns               ← mostly universal
Layer 7-10:  Object parts (ears, wheels)     ← somewhat specific
Layer 11-12: Full objects (dogs, cars)       ← specific to ImageNet classes
```

**Key insight:** The early/middle layers are **reusable**. Edges and shapes are edges and shapes whether you're classifying dogs or X-rays.

You keep the universal layers and only retrain the final layers for your task.

## Three Strategies

### Strategy 1: Feature Extraction (Fastest, Least Data)

Freeze the entire pretrained model. Only train a new output layer.

```
[Pretrained CNN — ALL FROZEN]  →  [New classifier layer — TRAINABLE]

Frozen: learns nothing new, just extracts features
New layer: learns to map those features to YOUR classes
```

**When to use:** You have very little data (< 1000 samples) and your task is similar to the original.

```python
import torchvision.models as models

# Load pretrained ResNet (trained on ImageNet)
model = models.resnet50(pretrained=True)

# Freeze ALL layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer for your task
model.fc = nn.Linear(2048, num_your_classes)  # only this trains
```

### Strategy 2: Fine-Tuning (Most Common)

Unfreeze some later layers and train them with a **very small learning rate**.

```
[Early layers — FROZEN]  [Later layers — TRAINABLE (tiny LR)]  [New head — TRAINABLE]

Early layers: keep universal features (edges, shapes)
Later layers: adapt to your domain
New head: learn your classes
```

**When to use:** You have moderate data (1000-50000 samples). This is the most common approach.

```python
# Load pretrained model
model = models.resnet50(pretrained=True)

# Freeze early layers (first 6 out of 10 blocks)
for name, param in model.named_parameters():
    if 'layer1' in name or 'layer2' in name or 'layer3' in name:
        param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(2048, num_your_classes)

# Use DIFFERENT learning rates
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},   # fine-tune slowly
    {'params': model.fc.parameters(), 'lr': 1e-3},        # new layer learns fast
])
```

### Strategy 3: Full Fine-Tuning (Most Data Needed)

Unfreeze everything and retrain the whole model with a small learning rate.

```
[ALL layers — TRAINABLE (tiny LR)]

The whole model adapts to your domain
Risk: can overfit if you don't have enough data
```

**When to use:** You have a lot of data (50000+) and your task is very different from the original.

## Transfer Learning for NLP (Hugging Face)

This is where you'll spend most of your time as an AI Engineer:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load pretrained BERT (trained on massive text corpus)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,  # your number of classes
)

# Tokenize your data
inputs = tokenizer("This movie was amazing!", return_tensors="pt")
outputs = model(**inputs)
# → class probabilities for your task

# Fine-tune on YOUR data with the Trainer API
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    learning_rate=2e-5,        # very small LR for fine-tuning!
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

## Common Pretrained Models

### For Images

| Model | Params | Use Case |
|-------|--------|----------|
| **ResNet-50** | 25M | General image classification (good starting point) |
| **EfficientNet** | 5-66M | Best accuracy/size tradeoff |
| **ViT (Vision Transformer)** | 86-632M | State-of-the-art when you have enough data |

### For Text/NLP

| Model | Params | Use Case |
|-------|--------|----------|
| **BERT** | 110M | Text understanding (classification, NER, QA) |
| **RoBERTa** | 125M | Better-trained BERT |
| **GPT-2** | 124M-1.5B | Text generation |
| **Llama** | 7B-70B | Open-source text generation |

## The Fine-Tuning Recipe

```
1. Pick a pretrained model close to your task
2. Replace the output layer for your number of classes
3. Freeze early layers (optional, depends on data size)
4. Use a SMALL learning rate (1e-5 to 5e-5 for NLP, 1e-4 for vision)
5. Train for just 2-5 epochs (not 100! the model is already good)
6. Monitor validation loss — stop if it increases (overfitting)
```

## How Much Data Do You Need?

| Strategy | Data Needed | Example |
|----------|-------------|---------|
| Feature extraction | 50-500 per class | Classify 5 types of skin lesions |
| Fine-tuning (partial) | 500-5000 per class | Sentiment analysis on reviews |
| Fine-tuning (full) | 5000+ per class | Custom domain NLP |
| Training from scratch | 100,000+ per class | Only when no pretrained model fits |

Compare: training ResNet from scratch on ImageNet took **weeks on 8 GPUs**. Fine-tuning it for your task takes **minutes on 1 GPU**.

## Common Mistakes

| Mistake | Why It's Bad | Fix |
|---------|-------------|-----|
| Learning rate too high | Destroys pretrained features | Use 1e-5 to 5e-5 for fine-tuning |
| Training too many epochs | Overfits to small dataset | 2-5 epochs, use early stopping |
| Not freezing enough layers | Overfits with small data | Freeze more layers, only train the head |
| Forgetting to preprocess correctly | Model expects specific input format | Use the model's own preprocessor/tokenizer |
| Training from scratch | Wasted effort when pretrained exists | Always check for pretrained models first |

## Key Takeaway

Transfer learning is the **most practical skill in deep learning**. You'll almost never train from scratch. The workflow: pick a pretrained model (Hugging Face has thousands), replace the output layer, fine-tune with a small learning rate on your data, done. This works because early layers learn **universal features** that transfer to any task. For NLP, Hugging Face's `transformers` library is your best friend. For images, `torchvision.models` has you covered.
