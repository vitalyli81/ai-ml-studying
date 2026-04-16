# Fine-tuning LLMs (LoRA, QLoRA, PEFT)

## What Is It?

Fine-tuning takes a pre-trained LLM and **trains it further on your specific data** so it learns new behaviors, knowledge, or styles. Instead of building an AI model from scratch (which costs millions), you adapt an existing one to your needs for a fraction of the cost.

```
Pre-trained LLM (general purpose)
        │
        ▼  + Your specific data
┌─────────────────┐
│  Fine-tuning     │  "Learn to write like our brand"
│  (hours, not     │  "Learn our internal API docs"
│   months)        │  "Learn medical terminology"
└────────┬────────┘
         │
         ▼
  Your Custom Model (specialized)
```

## Frontend Analogy

```javascript
// Fine-tuning is like theming a component library

// Pre-trained model = Material UI (generic, works for everyone)
// Fine-tuned model = Material UI + your company theme + custom components

// You don't rewrite Material UI from scratch.
// You take it as-is and customize the parts you need:

// import { ThemeProvider } from '@mui/material';
// const companyTheme = createTheme({
//   palette: { primary: '#1a73e8' },
//   typography: { fontFamily: 'Inter' },
//   // Your specific customizations
// });
//
// <ThemeProvider theme={companyTheme}>
//   <App />  {/* Everything inside now uses your style */}
// </ThemeProvider>

// Fine-tuning = adding your "theme" to a base model
// The model's core capabilities stay, but behavior changes
```

## When to Fine-tune vs When NOT to

```
✅ FINE-TUNE WHEN:
─────────────────
- You need a specific writing style/tone consistently
- You need the model to follow a complex format every time
- Prompt engineering alone can't get reliable results
- You have domain-specific jargon (legal, medical, finance)
- You need faster/cheaper inference (smaller fine-tuned model
  can match bigger base model on your specific task)

❌ DON'T FINE-TUNE WHEN:
──────────────────────
- You just need the model to know specific facts → use RAG
- A good system prompt already solves it → use prompt engineering
- You need real-time/updating data → use RAG
- You have less than ~100 high-quality examples → too few
- The base model already does the task well → don't fix what isn't broken
```

```
Decision Tree:
                                    
  Does the base model do it well?
        │
        ├── YES → Don't fine-tune. You're done.
        │
        └── NO → Can prompt engineering fix it?
                    │
                    ├── YES → Don't fine-tune. Better prompts are cheaper.
                    │
                    └── NO → Is it about KNOWING facts or DOING a behavior?
                                │
                                ├── KNOWING → Use RAG (cheaper, updatable)
                                │
                                └── DOING → Fine-tune! ✅
                                           (style, format, domain behavior)
```

## Full Fine-tuning vs Parameter-Efficient Methods

### Full Fine-tuning

Update **all** the model's parameters. Expensive but thorough.

```
Llama 3 70B has 70 billion parameters
Full fine-tuning: update all 70B parameters
Requires: 4-8 × A100 GPUs (80GB each), ~$5,000-$50,000

You need:
  - 140GB+ GPU memory (the model alone is 140GB in FP16)
  - Training data × batch size × gradient memory
  - Total: ~500-700GB GPU memory
  
This is impractical for most teams.
```

### PEFT (Parameter-Efficient Fine-Tuning)

Only update a **small fraction** of parameters. Same results, 100x cheaper.

```
PEFT Methods:
┌────────────────────────────────────────────────────────────┐
│  Method     │ Params Updated │ Memory Needed │ Quality     │
├────────────────────────────────────────────────────────────┤
│  Full FT    │ 100%           │ 500-700GB     │ ⭐⭐⭐⭐⭐    │
│  LoRA       │ 0.1-1%         │ 16-48GB       │ ⭐⭐⭐⭐      │
│  QLoRA      │ 0.1-1%         │ 8-24GB        │ ⭐⭐⭐⭐      │
│  Prefix     │ <0.1%          │ 8-16GB        │ ⭐⭐⭐       │
│  Adapters   │ <1%            │ 16-32GB       │ ⭐⭐⭐⭐      │
└────────────────────────────────────────────────────────────┘

LoRA and QLoRA are by far the most popular. Learn these first.
```

## LoRA (Low-Rank Adaptation)

### The Core Idea

Instead of updating the model's huge weight matrices, LoRA adds **small trainable matrices** alongside them. The original weights stay frozen.

```
Original weight matrix W: 4096 × 4096 = 16.7M parameters
                          (frozen, never changes)

LoRA adds two small matrices:
  A: 4096 × 16 = 65,536 parameters  (random init)
  B: 16 × 4096 = 65,536 parameters  (zero init)

Total LoRA params: 131,072  (0.8% of original!)

During inference:
  output = W·x + (B·A)·x
          ↑         ↑
     original    LoRA adjustment
     (frozen)    (trained)
```

### Visual Explanation

```
Standard Fine-tuning:
┌──────────────────────────┐
│  W (4096 × 4096)         │  ← Update ALL 16.7M params
│  ████████████████████████ │
│  ████████████████████████ │
│  ████████████████████████ │
└──────────────────────────┘

LoRA Fine-tuning:
┌──────────────────────────┐     ┌──────┐   ┌──────────────────────────┐
│  W (4096 × 4096)         │  +  │A(4096│ × │B (16 × 4096)             │
│  ░░░░░░░░░░░░░░░░░░░░░░░ │     │× 16) │   │████████████████████████  │
│  ░░░░░░░░░░░░░░░░░░░░░░░ │     │██████│   └──────────────────────────┘
│  ░░░░░░░░░░░░░░░░░░░░░░░ │     │██████│     Only train the ████ parts
└──────────────────────────┘     └──────┘     (0.8% of total params)
         FROZEN                   TRAINED
```

### Why It Works

```
Key insight: The changes needed for fine-tuning live in a LOW-RANK space.

You don't need to change all 16.7M values in W.
The adjustment can be expressed as two small matrices multiplied together.

Think of it like this:
  - Full fine-tuning: moving every pixel in a 4K image
  - LoRA: moving a few control points that warp the image

Both achieve the same visual result, but LoRA is way more efficient.
```

```javascript
// Frontend analogy:

// Full fine-tuning = forking the entire React codebase and modifying it
// LoRA = creating a small plugin that modifies React's behavior at runtime

// The React source code stays unchanged (frozen weights)
// Your plugin adds small interceptors (LoRA matrices)
// The result behaves differently, but React itself wasn't touched

// Another way: CSS overrides
// Full FT = rewriting the entire stylesheet
// LoRA = adding a small override.css that changes just what you need
//        .button { color: blue; }  ← small change, big visual impact
```

### The "Rank" in LoRA (the r parameter)

```
r = the inner dimension of the LoRA matrices (A and B)

r = 8:   A is 4096×8,  B is 8×4096   → 65K params  (very small)
r = 16:  A is 4096×16, B is 16×4096  → 131K params (small)
r = 64:  A is 4096×64, B is 64×4096  → 524K params (medium)
r = 256: A is 4096×256, B is 256×4096 → 2M params  (getting large)

Higher r = more capacity to learn = better but more expensive
Start with r=16 for most tasks. Increase if quality is insufficient.
```

## QLoRA (Quantized LoRA)

QLoRA = LoRA but the frozen base model is **compressed (quantized)** to use less memory.

```
Normal model (FP16):     Each parameter = 16 bits → 70B model = 140GB
4-bit quantized model:   Each parameter = 4 bits  → 70B model = 35GB

QLoRA:
  1. Quantize base model to 4-bit (frozen, saves memory)
  2. Add LoRA adapters in full precision (trainable)
  3. Train only the LoRA adapters

Result: Fine-tune a 70B model on a single 24GB GPU!
```

```
Memory comparison for Llama 3 70B:

Full fine-tuning:    ~600GB GPU memory  (8× A100 80GB)     💰💰💰💰💰
LoRA (FP16):         ~160GB GPU memory  (2× A100 80GB)     💰💰💰
QLoRA (4-bit):       ~24GB GPU memory   (1× RTX 4090)      💰
                                         ^^^^^^^^^
                                     A consumer GPU!
```

```javascript
// Frontend analogy:

// Full model in FP16 = uncompressed PNG image (huge but perfect quality)
// Quantized model = WebP image (much smaller, nearly identical quality)

// You can barely tell the difference visually (in model quality),
// but the file size (memory usage) is 4× smaller.

// QLoRA = editing a compressed image with a high-quality overlay
// Base: compressed (4-bit), Edits: full quality (FP16 LoRA)
```

## Practical Fine-tuning with Python

### Step 1: Prepare Your Data

Fine-tuning data is a set of (input, output) pairs:

```python
# training_data.jsonl — one example per line
{"messages": [
  {"role": "system", "content": "You are a helpful customer support agent for TechCorp."},
  {"role": "user", "content": "My laptop won't turn on"},
  {"role": "assistant", "content": "I'm sorry to hear that! Let's troubleshoot. First, is the charging LED lit when you plug in the power adapter? If not, try a different outlet. If the LED is on but the laptop won't start, try holding the power button for 15 seconds to force reset."}
]}

{"messages": [
  {"role": "system", "content": "You are a helpful customer support agent for TechCorp."},
  {"role": "user", "content": "How do I return a product?"},
  {"role": "assistant", "content": "You can return any product within 30 days of purchase. Go to techcorp.com/returns, enter your order number, and we'll email you a prepaid shipping label. Refunds process within 5-7 business days."}
]}
```

### Data Quality Guidelines

```
Quality > Quantity. 100 excellent examples beat 10,000 mediocre ones.

✅ Good training data:
  - Diverse examples covering different scenarios
  - Consistent style/format in responses
  - Correct, factual information
  - Natural conversation flow
  - Edge cases included

❌ Bad training data:
  - Copy-pasted or templated responses
  - Inconsistent formatting
  - Factual errors (the model will learn those too!)
  - Only simple/happy-path examples
  - Too short or too long responses

Minimum: ~100-500 high-quality examples for noticeable effect
Sweet spot: 1,000-5,000 examples
Diminishing returns: >10,000 examples
```

### Step 2: Fine-tune with Hugging Face + PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 1. Load base model
model_name = "meta-llama/Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,          # ← QLoRA: load in 4-bit
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Configure LoRA
lora_config = LoraConfig(
    r=16,                        # Rank — start with 16
    lora_alpha=32,               # Scaling factor (usually 2×r)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Which layers to adapt
    lora_dropout=0.05,           # Regularization
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: "trainable params: 6,553,600 || all params: 8,030,261,248 || trainable%: 0.0816"
# Only 0.08% of parameters are being trained!

# 4. Training configuration
training_args = TrainingArguments(
    output_dir="./my-fine-tuned-model",
    num_train_epochs=3,           # 3 passes through the data
    per_device_train_batch_size=4,
    learning_rate=2e-4,           # Standard for LoRA
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,                    # Mixed precision training
)

# 5. Train!
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,        # Your prepared dataset
    tokenizer=tokenizer,
    max_seq_length=2048,
)

trainer.train()

# 6. Save the LoRA adapter (small file!)
model.save_pretrained("./my-lora-adapter")
# This saves only the LoRA weights — typically 10-100MB
# Not the full model (which is 16GB+)
```

### Step 3: Use Your Fine-tuned Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "./my-lora-adapter")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

# Use it like any other model
inputs = tokenizer("How do I return a product?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
# Response now follows your fine-tuned style!
```

## Fine-tuning via APIs (No GPUs Needed)

If you don't want to manage GPUs, you can fine-tune through provider APIs:

### OpenAI Fine-tuning

```python
from openai import OpenAI
client = OpenAI()

# 1. Upload training data
file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# 2. Start fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",  # Base model to fine-tune
    hyperparameters={
        "n_epochs": 3,
    }
)

# 3. Use your fine-tuned model
response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:my-org::abc123",  # Your fine-tuned model ID
    messages=[{"role": "user", "content": "How do I return a product?"}]
)
```

## Evaluating Fine-tuned Models

```python
# Always compare: base model vs fine-tuned model on the same test set

test_prompts = [
    "How do I return a product?",
    "My laptop screen is cracked",
    "Can I get a discount?",
    # ... 50-100 test cases
]

# Evaluate both models
for prompt in test_prompts:
    base_response = base_model.generate(prompt)
    finetuned_response = finetuned_model.generate(prompt)
    
    # Manual evaluation (gold standard)
    # Or use an LLM-as-judge:
    judge_prompt = f"""
    Rate which response is better for a TechCorp customer support agent.
    
    User query: {prompt}
    Response A: {base_response}
    Response B: {finetuned_response}
    
    Rate each response 1-5 on: helpfulness, accuracy, brand voice.
    """
    evaluation = judge_llm.generate(judge_prompt)
```

### Key Metrics

```
1. Task-specific accuracy:  Does it do what you trained it to do?
2. Format compliance:       Does it follow the expected format?
3. Brand voice consistency: Does it sound right?
4. Hallucination rate:      Does it make stuff up more or less?
5. Regression testing:      Is it still good at general tasks?

Common pitfall: "catastrophic forgetting"
  → Model becomes great at your specific task
  → But loses general abilities it had before
  → Solution: mix in some general training data (5-10%)
```

## LoRA Hyperparameter Guide

```
Parameter        What It Does                   Start With
─────────────────────────────────────────────────────────────
r (rank)         Capacity of adaptation          16
lora_alpha       Scaling factor                  32 (2 × r)
lora_dropout     Regularization                  0.05
target_modules   Which layers to adapt           q_proj, v_proj
learning_rate    Step size for updates            2e-4
epochs           Passes through data              3
batch_size       Examples per step                4-8

When to increase r:
  - Complex task (e.g., learning a new language) → r=32 or r=64
  - Simple task (e.g., changing tone) → r=8 is enough

When to increase epochs:
  - Small dataset (<500 examples) → 5-10 epochs
  - Large dataset (>5000 examples) → 1-3 epochs

Signs of overfitting:
  - Training loss keeps dropping but validation loss goes up
  - Model starts memorizing training examples word-for-word
  - Fix: reduce epochs, increase dropout, add more diverse data
```

## Summary Comparison

```
┌──────────────────────────────────────────────────────────────────┐
│  Method          │ Cost     │ Time    │ GPU Needed │ Quality     │
├──────────────────────────────────────────────────────────────────┤
│  Prompt Eng.     │ Free     │ Minutes │ None       │ Good        │
│  RAG             │ $        │ Hours   │ None       │ Great       │
│  API Fine-tune   │ $$       │ Hours   │ None       │ Great       │
│  QLoRA (local)   │ $$       │ Hours   │ 1× 24GB   │ Great       │
│  LoRA (local)    │ $$$      │ Hours   │ 1-2× 80GB │ Excellent   │
│  Full Fine-tune  │ $$$$     │ Days    │ 4-8× 80GB │ Excellent   │
│  Pre-train       │ $$$$$$$  │ Months  │ Thousands  │ N/A         │
└──────────────────────────────────────────────────────────────────┘

Start from the top. Move down only when the simpler method isn't enough.
```

## Key Takeaways

| Concept | What to Remember |
|---------|-----------------|
| Fine-tuning | Adapt a pre-trained model to your specific task |
| LoRA | Train tiny adapter matrices, freeze the rest (0.1% params) |
| QLoRA | LoRA + 4-bit quantization = fine-tune 70B on one GPU |
| PEFT | Family of efficient fine-tuning methods (LoRA is most popular) |
| Data quality | 100 great examples > 10,000 mediocre ones |
| When to use | Style/behavior changes, not factual knowledge (use RAG for that) |
| Evaluation | Always compare base vs fine-tuned on held-out test set |

## What's Next?

You now have the complete LLM toolkit:
- **Fundamentals** — how LLMs work
- **Prompt Engineering** — how to talk to them
- **RAG** — how to give them knowledge
- **APIs & SDKs** — how to build apps with them
- **Agents** — how to let them take actions
- **Fine-tuning** — how to customize them

Next phase: [Phase 6: MLOps & Production](../README.md) — taking everything to production!
