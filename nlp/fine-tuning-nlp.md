# Fine-Tuning for NLP

## 1. TL;DR

Fine-tuning takes a pretrained model (BERT, Llama, T5) and trains it further on **your specific data** to become an expert at your task. The model already understands language — you're just redirecting that knowledge. Use **full fine-tuning for small models** (BERT, DistilBERT), **LoRA for large models** (Llama 7B+), and **QLoRA** when GPU memory is tight. Never fine-tune until you've confirmed that zero-shot or few-shot prompting doesn't already work well enough.

---

## 2. The Mental Model

> 💡 **Think of it like this:** Fine-tuning is like **hiring a brilliant generalist and giving them specialist training**.

A newly hired consultant (pretrained model) already knows business, communication, and problem-solving (general language understanding). You don't retrain them from kindergarten — you send them to a 2-week domain bootcamp (fine-tuning). They come out understanding your industry's jargon, your company's style, and your specific task requirements.

| Real world | Technical concept |
|---|---|
| Generalist consultant | Pretrained language model |
| Domain bootcamp (2 weeks, not 4 years) | Fine-tuning (hours, not months) |
| Learning your company's style guide | Adapting weights to your data distribution |
| Still knows general skills underneath | Pretrained weights preserved (low learning rate) |
| Specialist consultant with general foundation | Fine-tuned model |

---

## 3. Why It Exists

**The problem:** A pretrained model knows general language but doesn't know your domain. A general BERT trained on Wikipedia doesn't know that "LTV" means "lifetime value" in your SaaS app, or that "effusion" in a medical report is a symptom, not a grammar concept.

**What came before:** Training models from scratch on every task. Needed millions of examples, weeks of compute, and ML PhDs. Most companies couldn't afford it.

**What changed:** Transfer learning (2018+) showed that pretrained weights are almost universally useful as starting points. Fine-tuning on 1,000 labeled examples beats training from scratch on 100,000. The pretrained model already learned grammar, facts, and reasoning — you're just the last mile.

**What changed further:** LoRA (2021) showed you don't even need to update all parameters — train tiny adapter matrices (0.1% of weights) and get near-full fine-tuning quality. This brought fine-tuning from "need a data center" to "fits on your laptop GPU."

---

## 4. Core Concepts

### Transfer Learning

**One-line definition:** Use knowledge learned from one task/dataset as the starting point for a different task/dataset.

**Analogy:** A Spanish speaker learning Portuguese. They don't start from zero — they transfer knowledge of Latin roots, sentence structure, and vocabulary patterns. Fine-tuning is the same: a model trained on English Wikipedia transfers that language knowledge to sentiment analysis on product reviews.

```
Pre-training (learned from internet-scale data):
  Grammar, facts, reasoning, language patterns, world knowledge

Fine-tuning adds on top:
  Your domain vocabulary, your task format, your label space
```

**Common misconception:** ❌ "I need millions of examples to fine-tune" → ✅ 100-1,000 labeled examples often beat a baseline trained from scratch on 100,000. The pretrained model already did the heavy lifting. You're providing task-specific signal, not teaching language from scratch.

---

### Catastrophic Forgetting

**One-line definition:** When training on new data with too high a learning rate, the model "forgets" what it learned during pretraining.

**Analogy:** Imagine the consultant from the analogy came back from bootcamp and forgot how to write professional emails. That's catastrophic forgetting — they over-specialized and lost general skills.

```
Learning rate too HIGH:
  New task gradients overwrite pretrained weights
  Model becomes great at your task but terrible at everything else
  Often shows as: training loss drops but validation loss stays high

Learning rate just right (2e-5 to 5e-5):
  Model adapts to your task while retaining pretrained knowledge
  Both training and validation loss decrease together
```

**Common misconception:** ❌ "Larger learning rate trains faster so it's better" → ✅ Fine-tuning is not training from scratch. A learning rate of 2e-5 is 100x smaller than a typical from-scratch learning rate. This small rate makes tiny, careful adjustments to existing knowledge.

---

### When to Fine-Tune vs. When NOT To

**One-line definition:** Fine-tune only when simpler approaches (prompting, zero-shot) don't meet your accuracy requirements.

**Decision flow:**

```
Can prompting solve it?
  ↓ yes → use LLM API (Claude, GPT-4). Cheapest, fastest, most flexible.
  ↓ no ↓
Do you have labeled data (100+ examples)?
  ↓ no → collect data or use few-shot prompting
  ↓ yes ↓
Is it a classification/extraction task on a small model?
  ↓ yes → full fine-tune BERT/DistilBERT
Is it a generation task or requires a large model (7B+)?
  ↓ yes → LoRA or QLoRA fine-tuning
```

**Fine-tune WHEN:**
- Zero/few-shot accuracy isn't good enough
- Need consistent output format (always JSON, specific schema)
- High volume where API costs matter (1M+ predictions/month)
- Domain-specific vocabulary LLMs don't know
- Need offline/local inference

**DON'T fine-tune WHEN:**
- Prompting already works (don't fix what's not broken)
- Fewer than ~100 examples per class
- Task changes frequently (prompts are easier to update than models)
- Still prototyping (premature optimization)

---

### LoRA (Low-Rank Adaptation)

**One-line definition:** Instead of updating all model weights, train tiny adapter matrices inserted into each layer — 0.1% the parameters, near-full performance.

**Analogy:** Instead of repainting the entire house (full fine-tuning), you apply wall decals (LoRA adapters) — fast, cheap, reversible, and you can swap them out for different styles.

```
Full fine-tuning:
  7B model × 4 bytes/param = 28GB for weights
  + optimizer states = 80GB+ total
  Result: expensive, needs big GPU

LoRA fine-tuning:
  7B parameters FROZEN (no gradients needed)
  + add tiny matrices A and B at each attention layer:
    Original W: [4096 × 4096] = 16M params
    LoRA A:     [4096 × 8]    = 32K params  (down-project to rank r)
    LoRA B:     [8 × 4096]    = 32K params  (up-project back)
  Train only A and B: 64K params vs 16M = 0.4% of original
  Total trainable params: ~4M out of 7B = 0.05%
  GPU needed: 8-16GB instead of 80GB+
```

**How it works at inference:**
```
W_effective = W_frozen + (B × A) × scaling_factor
Merge adapters into weights → zero inference overhead
Or keep separate → swap adapters without reloading the base model
```

**Common misconception:** ❌ "LoRA is just a workaround for small GPUs" → ✅ LoRA often matches or exceeds full fine-tuning quality on many tasks, while also being faster to train and easier to swap. It's now the standard approach even when memory isn't a constraint.

---

### QLoRA

**One-line definition:** LoRA + 4-bit quantization — compress model weights from 16-bit to 4-bit floats, reducing memory by 75%, then apply LoRA on top.

**Analogy:** LoRA is wall decals on a normal house. QLoRA is wall decals on a pre-fabricated modular house — the house itself is more compact (quantized), and you still only paint the decals (LoRA adapters), not the walls.

```
Memory comparison for fine-tuning Llama 7B:
  Full fine-tuning:  ~80GB  (A100 territory)
  LoRA:              ~16GB  (consumer workstation)
  QLoRA:             ~6GB   (RTX 3090 / Google Colab T4)

Quality: Full ≥ LoRA ≈ QLoRA (near-indistinguishable for most tasks)
```

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # normal float 4 quantization
    bnb_4bit_compute_dtype="float16",   # compute in fp16, store in 4-bit
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
# Then apply LoRA on top of this quantized model
```

---

### Hyperparameters for Fine-Tuning

**One-line definition:** The settings you configure before training that control how learning happens.

**Analogy:** Like cooking temperature and time — too hot (high learning rate) burns the dish (catastrophic forgetting); too cool (too low learning rate) leaves it undercooked (underfitting); too long (too many epochs) and it dries out (overfitting).

```
Learning rate:  2e-5 to 5e-5   ← start here, always
                (NOT 1e-3 — that's for training from scratch)

Epochs:         2-5            ← fine-tuning is fast; more epochs = overfitting
                (NOT 100 — pretrained models don't need that many updates)

Batch size:     8-32           ← depends on GPU memory
                (use gradient_accumulation_steps if limited VRAM)

Weight decay:   0.01           ← L2 regularization to prevent overfitting
Warmup steps:   ~100           ← slowly ramp up learning rate at start
```

---

## 5. How It Actually Works (Step-by-Step)

Let's trace full fine-tuning of DistilBERT for customer support ticket classification:

```
GOAL: Classify tickets into [billing, technical, returns, other]

Step 1: Load pretrained model + tokenizer
  model = AutoModelForSequenceClassification.from_pretrained(
      "distilbert-base-uncased",
      num_labels=4        ← add 4-class classification head
  )
  Pretrained layers: FROZEN initially, then small updates
  New classification head: randomly initialized (trained from scratch)

Step 2: Tokenize your labeled data
  "My payment didn't go through" → [101, 2026, 7834, 2134, ...] + labels=[0]  (billing)
  "App keeps crashing on login"  → [101, 4906, 4273, 7482, ...] + labels=[1]  (technical)
  (need ~200-500 examples per class)

Step 3: Forward pass — compute predictions
  Input tokens → DistilBERT layers (6 layers, pretrained weights)
  [CLS] output → linear head → [4.2, -1.3, 0.5, -2.1] (logits for 4 classes)

Step 4: Compute loss
  True label: billing (class 0)
  Prediction: [0.96, 0.01, 0.02, 0.01] after softmax
  Cross-entropy loss = -log(0.96) = 0.04   ← low loss, correct prediction

  OR:
  True label: technical (class 1)
  Prediction: [0.60, 0.20, 0.15, 0.05] after softmax
  Cross-entropy loss = -log(0.20) = 1.61   ← high loss, wrong prediction

Step 5: Backpropagation + weight update
  Gradients flow back through classification head and BERT layers
  Optimizer (AdamW) updates weights by: W ← W - lr × gradient
  Learning rate 2e-5: tiny nudges, preserving pretrained knowledge

Step 6: Repeat for 3 epochs over your training set
  Epoch 1: accuracy ~65%  (head learns, BERT adapts slowly)
  Epoch 2: accuracy ~82%  (BERT fine-tunes to your domain)
  Epoch 3: accuracy ~89%  (convergence)

Step 7: Evaluate on held-out test set
  Final accuracy: 88%, F1: 0.87
  → Compare to baseline (zero-shot): 71% → fine-tuning helped!

Step 8: Save + deploy
  trainer.save_model("./ticket-classifier")
  pipeline("text-classification", model="./ticket-classifier")
```

> 💡 **Key Insight:** The pretrained weights are not discarded — they're the reason you only need 500 examples instead of 500,000. Fine-tuning is a dialogue between old knowledge and new task, not a replacement.

---

## 6. Code in Practice

### Minimal: Fine-tune DistilBERT for classification

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import load_dataset
import numpy as np
import evaluate

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

dataset = load_dataset("imdb")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=256)

dataset = dataset.map(tokenize, batched=True)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,        # ← always small for fine-tuning
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
print(trainer.evaluate())  # {'eval_accuracy': 0.924}
```

### Practical: LoRA fine-tuning a large model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Trainer, TrainingArguments

model_name = "meta-llama/Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                              # rank — tradeoff: higher = more capacity
    lora_alpha=32,                    # scaling factor (usually 4×r)
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # which attention matrices to adapt
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 8,030,261,248 || trainable%: 0.0522

# Train same as any HF model
trainer = Trainer(model=model, args=TrainingArguments(...), ...)
trainer.train()

# Save only the adapter (tiny: ~20MB vs 16GB for full model)
model.save_pretrained("./my-lora-adapter")
```

### Real-world pattern: QLoRA for consumer GPU

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,   # extra compression
)

# Load 8B model in ~6GB instead of ~16GB
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# Apply LoRA on top of quantized model
lora_config = LoraConfig(r=16, lora_alpha=64, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# Fine-tune on your instruction dataset (chat format)
# Format: {"input": "user message", "output": "expected response"}
# ... same Trainer setup as above
```

---

## 7. Gotchas & Pitfalls

❌ **Learning rate too high (e.g., 1e-3)** → ✅ Catastrophic forgetting. Use 1e-5 to 5e-5 for fine-tuning. The model took weeks to pretrain — don't undo it with large gradient updates.

❌ **Too many epochs (> 5)** → ✅ Overfitting. The model memorizes your training data and loses generalization. Monitor validation loss — if it stops decreasing and starts rising, stop training.

❌ **Not comparing to a baseline** → ✅ Always measure: (1) zero-shot prompt → (2) few-shot prompt → (3) fine-tuned model. Fine-tuning is expensive; confirm it actually helps before committing.

❌ **Wrong tokenizer at inference** → ✅ The fine-tuned model's input_ids depend on the exact tokenizer used during training. If you switch tokenizers at inference, the model sees a completely different sequence and outputs garbage.

❌ **Not shuffling data** → ✅ Ordered data (all class 0 then all class 1) causes the model to see only one label for many batches, creating unstable gradients. Always shuffle before creating DataLoaders.

❌ **Evaluating on training data** → ✅ Training accuracy is always higher than test accuracy. A model that achieves 99% training accuracy and 60% test accuracy is overfit. Always hold out a test set before fine-tuning.

❌ **Skipping LoRA for large models** → ✅ Trying to full fine-tune a 7B+ model without LoRA requires 80GB+ of GPU memory. Use LoRA — it fits in 8-16GB with near-identical quality.

---

## 8. When to Use / When NOT to Use

### Use full fine-tuning when:
- **Small model** (BERT, DistilBERT, T5-small) with enough GPU memory
- **Classification/NER/QA tasks** where you have 500+ labeled examples per class
- **Maximum accuracy** on a fixed, well-defined task
- **Production at scale** where API cost is prohibitive

### Use LoRA when:
- **Large model** (7B+ parameters) and limited GPU memory
- **Multiple domain variants** — keep one base model, swap adapter files
- **Instruction fine-tuning** — teaching a model to follow your response format
- **Rapid experimentation** — train small adapters, compare, pick the best

### Use QLoRA when:
- **Consumer GPU** (RTX 3080/3090, 8-16GB VRAM)
- **Colab / single GPU environment**
- **Prototyping** before committing to a larger training run

### Don't fine-tune when:
- **Zero-shot/few-shot prompting already achieves acceptable accuracy**
- **Fewer than ~100 examples per class** — model will overfit immediately
- **Task definition is still changing** — retraining is expensive; use prompts while requirements evolve
- **No evaluation set** — you need a held-out set to know if fine-tuning actually helped

---

## 9. Related Concepts (The Map)

- **Pretrained Models** — the starting point for fine-tuning. BERT, DistilBERT, Llama, T5 are all pretrained on large corpora. Without pretraining, fine-tuning would require millions of examples.
- **Hugging Face Trainer** — the standard training loop for fine-tuning. Handles gradient updates, checkpointing, evaluation, and logging out of the box.
- **PEFT library** — Hugging Face's library for parameter-efficient fine-tuning. LoRA, QLoRA, prefix tuning, and prompt tuning all live here. Install with `pip install peft`.
- **RAG (alternative to fine-tuning)** — for adding knowledge rather than changing behavior, RAG is often better than fine-tuning. Fine-tuning teaches *how to respond*; RAG teaches *what facts to use*.
- **Evaluation metrics** — fine-tuning without evaluation is blind. Always track accuracy, F1, or task-specific metrics on a held-out set across epochs.

---

## 10. Cheat Sheet

| Method | Memory | Quality | When to Use |
|---|---|---|---|
| **Zero-shot prompting** | None (API) | Good | First try, always |
| **Few-shot prompting** | None (API) | Better | 5-20 examples available |
| **Full fine-tune** | High (all params) | Best | Small models, enough data |
| **LoRA** | Low (0.1% params) | Near-best | Large models, limited GPU |
| **QLoRA** | Very low (4-bit + LoRA) | Good | Consumer GPU |

**Hyperparameter defaults:**
```
learning_rate             = 2e-5      ← never higher than 5e-5
num_train_epochs          = 3         ← start here, add if val loss still falling
per_device_train_batch_size = 16      ← reduce if OOM
weight_decay              = 0.01      ← L2 regularization
evaluation_strategy       = "epoch"   ← always monitor val performance
load_best_model_at_end    = True      ← don't use the last checkpoint
```

**LoRA defaults:**
```python
LoraConfig(
    r=8,              # rank: 4-64 (start at 8)
    lora_alpha=32,    # scaling: usually 4×r
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # attention matrices
)
```

**Remember this:**
1. Try prompting first — fine-tune only when it's not good enough
2. Learning rate 2e-5: small enough to preserve pretrained knowledge
3. LoRA is the standard for 7B+ models — same quality, fraction of the memory

---

## 11. Self-Check Questions

1. Why do you use a much smaller learning rate for fine-tuning than for training from scratch?
2. Your fine-tuned model has 97% training accuracy but 62% validation accuracy. What's wrong, and how do you fix it?
3. What's the practical difference between LoRA and QLoRA, and when would you choose each?
4. You have a new NLP task at work with 50 labeled examples. Should you fine-tune? What instead?
5. Why is LoRA's quality "near-full fine-tuning quality" even though it only trains 0.05% of parameters?

<details>
<summary>Answers</summary>

1. The pretrained model spent weeks learning general language representations from billions of examples. A large learning rate would overwrite those representations with large gradient updates from your small dataset — "catastrophic forgetting." A small learning rate (2e-5) makes tiny, careful adjustments that direct the model toward your task while preserving the valuable pretrained knowledge underneath.

2. This is **overfitting** — the model memorized your training set instead of learning generalizable patterns. Fixes: (1) reduce training epochs (stop at epoch 2 based on val loss), (2) add `weight_decay=0.01` for regularization, (3) collect more training data, (4) use a smaller model (less prone to memorization), (5) add data augmentation (paraphrase training examples).

3. **LoRA** trains tiny adapter matrices on top of the full-precision (16-bit) frozen base model. GPU memory requirement: ~16GB for a 7B model. **QLoRA** first quantizes the base model to 4-bit (reducing 16GB to ~4GB), then applies LoRA adapters on top. GPU memory requirement: ~6GB for a 7B model. Choose LoRA when you have a 16GB GPU workstation; choose QLoRA when you're on a consumer GPU (RTX 3080, Google Colab T4) or want to fit larger models in the same memory.

4. Don't fine-tune with 50 examples — you'll overfit immediately. Better options: (1) try zero-shot prompting with Claude/GPT-4 — it may work well enough, (2) try few-shot prompting (include 5-10 examples in the prompt), (3) collect more labeled data until you have 200-500 examples per class, then fine-tune. With 50 examples, your evaluation set would be tiny and results unreliable anyway.

5. The hypothesis behind LoRA is that the changes needed during fine-tuning have low "intrinsic dimensionality" — you don't need to change all 7 billion parameters to adapt the model. Most of the useful adaptation happens in the attention layers, and within those layers, in a low-rank subspace. The small adapter matrices (A × B) learn to capture this low-rank adaptation. Empirically, this works remarkably well — the full fine-tuning solution appears to lie in a low-dimensional manifold that LoRA can reach with far fewer parameters.

</details>

---

## 12. Go Deeper

- **["LoRA: Low-Rank Adaptation of Large Language Models" (Hu 2021)](https://arxiv.org/abs/2106.09685)** — the original LoRA paper. Clear intuition in Section 2, empirical results in Section 5. Read this to understand *why* it works, not just how to use it.
- **["QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers 2023)](https://arxiv.org/abs/2305.14314)** — the QLoRA paper showing 65B models can be fine-tuned on a single 48GB GPU. Explains 4-bit NormalFloat quantization. The technique that democratized LLM fine-tuning.
- **[Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/)** — complete guide to LoRA, QLoRA, prefix tuning, and prompt tuning. The official reference for parameter-efficient fine-tuning in practice.
- **[Axolotl fine-tuning framework](https://github.com/OpenAccess-AI-Collective/axolotl)** — production-grade fine-tuning tool used by researchers and companies. Supports QLoRA, datasets, FSDP. Better than raw Trainer for serious fine-tuning projects.
- **["Fine-Tuning vs. RAG for LLMs" — Anyscale blog](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts)** — clear breakdown of when to fine-tune vs. when to use RAG. Key insight: fine-tuning changes *how* the model responds; RAG changes *what information* it has access to.
