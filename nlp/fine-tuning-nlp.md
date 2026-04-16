# Fine-Tuning for NLP

## What Is It?

Fine-tuning takes a **pretrained language model** (BERT, GPT, Llama) and trains it further on **your specific data** so it becomes an expert at your task. The model already understands language — you're just teaching it your domain.

```
Pretrained BERT:     Understands English (general)
     + fine-tune on medical papers
Fine-tuned BERT:     Understands medical terminology and diagnoses
```

## Frontend Analogy

```javascript
// Pretrained model = a component library (Material UI)
// Fine-tuning = customizing the theme and overriding specific components

// You don't rebuild every component from scratch.
// You take the working library and adapt it to YOUR brand.

// import { ThemeProvider, createTheme } from '@mui/material';
// const myTheme = createTheme({
//   palette: { primary: { main: '#your-brand-color' } },
//   components: { MuiButton: { /* your overrides */ } }
// });
// <ThemeProvider theme={myTheme}>  ← that's fine-tuning!
```

## When to Fine-Tune vs When NOT To

| Approach | When to Use | Cost |
|----------|-------------|------|
| **Zero-shot (just prompt it)** | Task is simple, LLM handles it well | Free / API cost per call |
| **Few-shot (examples in prompt)** | Need slight guidance, works with 3-10 examples | Same as zero-shot |
| **Fine-tune** | Need high accuracy on specific domain, consistent format, lower latency | GPU time + labeled data |
| **Train from scratch** | Almost never. Only if no pretrained model fits | Massive cost |

### Fine-Tune When:

- Zero/few-shot accuracy isn't good enough
- You need **consistent output format** (always returns JSON, always follows a schema)
- You have **domain-specific data** (medical, legal, code in your language)
- You want a **smaller, faster, cheaper model** that matches a big model's quality on your specific task
- You need to run **locally** without API calls

### DON'T Fine-Tune When:

- Prompting already works well enough
- You don't have labeled data (at least 100-1000 examples)
- The task changes frequently (prompting is more flexible)
- You're still prototyping (fine-tuning is premature optimization)

## The Fine-Tuning Workflow

```
1. CHOOSE BASE MODEL
   └── BERT/DistilBERT for classification/NER
   └── Llama/Mistral for generation
   └── T5 for text-to-text tasks

2. PREPARE YOUR DATA
   └── Labeled examples in the right format
   └── Split: 80% train, 10% validation, 10% test

3. SET HYPERPARAMETERS
   └── Learning rate: 1e-5 to 5e-5 (SMALL — don't destroy pretrained knowledge)
   └── Epochs: 2-5 (not 100!)
   └── Batch size: 8-32

4. TRAIN
   └── Use Hugging Face Trainer or your own loop
   └── Monitor validation loss — stop if it increases

5. EVALUATE
   └── Test on held-out data
   └── Compare against baseline (zero-shot, few-shot)

6. DEPLOY
   └── Save model → load in production → serve predictions
```

## Full Example: Sentiment Classification

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import load_dataset
import numpy as np
import evaluate

# 1. Load dataset
dataset = load_dataset("imdb")
# Reduce for faster demo (use full data in production)
train_data = dataset["train"].shuffle(seed=42).select(range(2000))
test_data = dataset["test"].shuffle(seed=42).select(range(500))

# 2. Load pretrained model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Tokenize
def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=256)

train_data = train_data.map(preprocess, batched=True)
test_data = test_data.map(preprocess, batched=True)

# 4. Define metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# 5. Training config
training_args = TrainingArguments(
    output_dir="./sentiment-model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,              # small LR!
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50,
)

# 6. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
)
trainer.train()

# 7. Evaluate
results = trainer.evaluate()
print(f"Accuracy: {results['eval_accuracy']:.2%}")

# 8. Save and use
trainer.save_model("./sentiment-model")
tokenizer.save_pretrained("./sentiment-model")

# Load and use later:
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="./sentiment-model")
classifier("This movie was amazing!")  # → POSITIVE 0.99
```

## LoRA & PEFT — Fine-Tuning Large Models Cheaply

Full fine-tuning updates **all parameters**. For a 7B model, that needs ~28GB of GPU memory just for the model, plus more for gradients and optimizer state.

**LoRA (Low-Rank Adaptation)** only trains a **tiny adapter** (0.1-1% of parameters):

```
Full fine-tuning:
  7B parameters × 4 bytes = 28GB + optimizer = 80GB+ GPU needed

LoRA fine-tuning:
  7B parameters FROZEN
  + 10M adapter parameters TRAINABLE (0.1%)
  = fits in 8-16GB GPU!
```

### How LoRA Works (Simplified)

```
Original weight matrix W (huge):     [4096 × 4096] = 16M parameters

LoRA adds two small matrices:
  A: [4096 × 8]   = 32K parameters   (down-project)
  B: [8 × 4096]   = 32K parameters   (up-project)

During fine-tuning: only train A and B (64K params instead of 16M)
During inference:   W_new = W + A × B (merge back)
```

The rank (8 in this example) controls the tradeoff:
- **Higher rank** (16, 32) → more capacity, more memory
- **Lower rank** (4, 8) → less memory, slightly less accuracy

### LoRA with Hugging Face PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load base model
model_name = "meta-llama/Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # rank (smaller = less memory)
    lora_alpha=32,          # scaling factor
    lora_dropout=0.1,       # regularization
    target_modules=["q_proj", "v_proj"],  # which layers to adapt
)

# 3. Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 8,030,261,248
# trainable%: 0.0522%    ← only 0.05% of parameters!

# 4. Train with Trainer (same as before)
# 5. Save adapter (tiny file, ~20MB instead of 16GB)
model.save_pretrained("./my-lora-adapter")

# 6. Load for inference
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "./my-lora-adapter")
```

## QLoRA — Even Cheaper

QLoRA = LoRA + **4-bit quantization** (compress model weights from 16-bit to 4-bit):

```
Full fine-tuning 7B model:   ~80GB GPU
LoRA:                        ~16GB GPU
QLoRA:                       ~6GB GPU   ← fits on consumer GPUs!
```

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4-bit quantization
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)
# Then apply LoRA on top of this quantized model
```

## Fine-Tuning Comparison

| Method | Memory | Speed | Quality | When to Use |
|--------|--------|-------|---------|-------------|
| **Full fine-tune** | Very high | Slow | Best | Small models (BERT) or big GPU budget |
| **LoRA** | Low | Fast | Near-best | Default for large models (7B+) |
| **QLoRA** | Very low | Moderate | Good | Consumer GPUs, prototyping |
| **Prompt tuning** | Minimal | Fast | Good enough | When you can't modify model at all |

## Common Mistakes

| Mistake | Why It Hurts | Fix |
|---------|-------------|-----|
| Learning rate too high | Destroys pretrained knowledge ("catastrophic forgetting") | Use 1e-5 to 5e-5 |
| Too many epochs | Overfits to training data | 2-5 epochs, early stopping |
| Wrong tokenizer | Tokens don't match the model | Always load tokenizer WITH the model |
| Not shuffling data | Model learns data order, not patterns | `shuffle=True` in DataLoader |
| Too little data | Model overfits immediately | Need at least 100+ examples per class |
| Skipping evaluation | Don't know if fine-tuning actually helped | Always compare to baseline |

## Key Takeaway

Fine-tuning adapts a pretrained model to **your specific task** — like customizing a UI library for your brand. Use **full fine-tuning for small models** (BERT, DistilBERT) and **LoRA/QLoRA for large models** (Llama, Mistral). The recipe: small learning rate (2e-5), few epochs (3), monitor validation loss. Always compare your fine-tuned model against a zero-shot/few-shot baseline to confirm it's actually worth the effort. LoRA makes fine-tuning accessible on consumer hardware — you can adapt a 7B model on a single GPU.
