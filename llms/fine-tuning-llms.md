# Fine-tuning LLMs (LoRA, QLoRA, PEFT)

## TL;DR

Fine-tuning takes a pre-trained LLM and trains it further on your specific data so it learns new behaviors, styles, or domain-specific patterns. You're not building from scratch — you're customizing an existing model for your needs. LoRA (Low-Rank Adaptation) makes this practical by training only 0.1% of parameters; QLoRA compresses the model further so a 70B model fits on a single consumer GPU. Fine-tune for *behavior* (style, format, following domain conventions) — not for *facts* (use RAG for that).

> 💡 **Key Insight:** Fine-tuning changes HOW the model responds. RAG changes WHAT the model knows. Don't confuse them — this mistake wastes weeks of effort.

---

## The Mental Model

**Think of fine-tuning like teaching a foreign exchange student your company's specific way of doing things.**

The student already speaks English fluently and knows general programming (that's the pre-trained model). You're not re-teaching them English. You're teaching them your team's code review standards, your customer support tone, your internal jargon. A few weeks of focused practice, and they respond like a senior member of your team.

| Real world | Technical concept |
|------------|------------------|
| Student already knows English | Pre-trained model has general knowledge |
| Teaching team-specific practices | Fine-tuning on your domain data |
| Practice exercises with feedback | Training on (input, ideal output) pairs |
| Only learning company-specific parts | Parameter-efficient fine-tuning (PEFT) |
| Student adapts without losing English | Catastrophic forgetting risk |
| Different trainees, same knowledge | Multiple LoRA adapters on one base model |

---

## Why It Exists (Problem → Solution)

**The problem:** General-purpose LLMs don't know your domain. They don't write in your brand voice, don't follow your exact JSON schema, don't understand your internal product terminology, and aren't consistently reliable for your specific task even with perfect prompts.

**What came before:** Full retraining from scratch — impossibly expensive for most teams ($1M+). Or endlessly tweaking prompts and accepting inconsistency.

**What changed:** PEFT methods (2021–2022) — especially LoRA — made fine-tuning possible without expensive GPU clusters. You can adapt a powerful base model for your specific needs in hours, on a single GPU, for less than $100. This democratized model customization.

---

## Core Concepts

### 1. When to Fine-tune vs. When NOT to

**Plain English:** Fine-tuning is expensive and slow compared to prompting. Only reach for it when prompting reliably fails. It's a scalpel, not a hammer.

**Analogy:** Training a new employee vs. writing better instructions. If a well-written memo solves the problem, don't spend 3 months in training. Fine-tuning is the training program — use it when the memo isn't enough.

```
Decision tree:
                                    
Does the base model do it well with the right prompt?
      │
      ├── YES → Don't fine-tune. You're done.
      │
      └── NO → Can better prompt engineering fix it?
                  │
                  ├── YES → Don't fine-tune. Write better prompts.
                  │
                  └── NO → Is it about KNOWING facts or DOING a behavior?
                              │
                              ├── KNOWING → Use RAG. (cheaper, updatable)
                              │
                              └── DOING → Fine-tune ✅
                                         (style, format, domain behavior)
```

```
✅ Good reasons to fine-tune:
  - Need specific writing style consistently (brand voice, legal tone)
  - Need strict output format the model keeps getting wrong
  - Domain jargon the model misinterprets (medical, legal, finance)
  - Task requires specialized reasoning patterns
  - Need faster/cheaper inference (small fine-tuned model ≈ large base model)

❌ Bad reasons to fine-tune:
  - "I want the model to know our product docs" → use RAG
  - "The base model is almost right" → fix the prompt first
  - "We have 50 examples" → too few, use few-shot prompting
  - "We need real-time data" → use RAG
```

**Common misconception:** Fine-tuning teaches the model new facts. It teaches *behaviors*, not facts. If you fine-tune on your product docs, the model will write responses *in the style* of your docs but will still hallucinate facts. For knowledge, use RAG.

---

### 2. Full Fine-tuning vs. PEFT

**Plain English:** Full fine-tuning updates all model parameters — impossibly expensive for large models. PEFT methods update only a tiny fraction — same quality, 100× cheaper.

**Analogy:** Full fine-tuning is renovating every room in a house. LoRA is adding a few key pieces of furniture — enough to make it feel completely different without touching the structure.

```
Method          Params Updated   GPU Memory Needed   Cost
──────────────────────────────────────────────────────────
Full FT         100%             500–700 GB          $$$$$$
LoRA            0.1–1%           16–48 GB            $$
QLoRA           0.1–1%           8–24 GB             $
Prefix Tuning   <0.1%            8–16 GB             $
Adapters        <1%              16–32 GB            $$

→ LoRA and QLoRA dominate in practice. Learn these.
```

**Common misconception:** PEFT methods compromise quality. Well-tuned LoRA matches full fine-tuning quality on most tasks, especially when the task doesn't require fundamentally changing the model's knowledge.

---

### 3. LoRA (Low-Rank Adaptation)

**Plain English:** Instead of updating the model's huge weight matrices, LoRA adds two small trainable matrices alongside each weight layer. The original weights stay frozen — only the tiny LoRA matrices train.

**Analogy:** Think of CSS `!important` overrides. You don't rewrite the entire stylesheet — you add a small override file that changes only what you need, while the original stylesheet stays intact.

```
Standard weight matrix W: 4096 × 4096 = 16.7M parameters
                          (frozen, never changes during fine-tuning)

LoRA adds two small matrices:
  A: 4096 × 16   = 65,536 parameters  (random init)
  B: 16   × 4096 = 65,536 parameters  (zero init)
  
Total LoRA params: 131,072  (0.8% of original!)

During forward pass:
  output = W·x  +  scale × (B·A)·x
           ↑              ↑
      original        LoRA adjustment
      (frozen)        (trained on your data)
```

```
Visual comparison:

Standard Fine-tuning:              LoRA Fine-tuning:
┌──────────────────┐               ┌──────────────────┐   ┌────┐   ┌──────────────────┐
│  W (4096×4096)   │               │  W (4096×4096)   │ + │ A  │ × │  B (16×4096)     │
│ ████████████████ │  ← update all │ ░░░░░░░░░░░░░░░░ │   │4096│   │ ████████████████ │
│ ████████████████ │  16.7M params │ ░░░░░░░░░░░░░░░░ │   │× 16│   │                  │
│ ████████████████ │               │ ░░ FROZEN ░░░░░░ │   └────┘   └──────────────────┘
└──────────────────┘               └──────────────────┘   train only the ████ parts
```

**The `r` (rank) parameter:**
```
r = inner dimension of LoRA matrices

r = 8:   → 65K additional params   (simple tasks — tone, format)
r = 16:  → 131K additional params  (most tasks — start here)
r = 64:  → 524K additional params  (complex tasks — new capabilities)
r = 256: → 2M additional params    (diminishing returns beyond this)

Higher r = more capacity = better quality = slower training
Start with r=16. Increase only if quality is insufficient.
```

**Common misconception:** Higher rank always helps. For simple style changes (different output format, different tone), r=8 is often identical to r=64 and trains 8× faster.

---

### 4. QLoRA (Quantized LoRA)

**Plain English:** QLoRA = LoRA + compressed base model. The frozen base model is quantized to 4-bit (from 16-bit), making it ~4× smaller in memory. The LoRA adapters stay in higher precision. Result: fine-tune large models on a single GPU that wouldn't otherwise fit.

**Analogy:** A high-quality JPEG instead of a RAW photo. The image looks almost identical, but the file is much smaller. You edit the JPEG with full-resolution brushes (LoRA adapters in BF16) — the compressed base is the canvas, your edits are precise.

```
Rough memory budgets (weights + activations + LoRA grads + optimizer state):

Model size │ Full FT (BF16) │ LoRA (BF16)    │ QLoRA (4-bit base)
───────────┼────────────────┼────────────────┼────────────────────
  8B       │  ~160 GB       │  ~24 GB        │  ~10–12 GB  (1× 16GB)
 13B       │  ~260 GB       │  ~40 GB        │  ~14–18 GB  (1× 24GB)
 70B       │ ~1,400 GB      │  ~160 GB       │  ~46–48 GB  (1× 48GB)

  → 70B QLoRA realistically needs an A6000/A100 (48GB), not a 24GB RTX 4090.
  → The original QLoRA paper fine-tuned 65B on a single 48GB GPU.
  → 24GB is enough for 7B–13B QLoRA, which is already huge.
```

**Cost comparison (renting GPUs for 4 hours to tune a 13B model):**
```
Full FT:  4× A100 80GB ≈ $40/hr → $160
QLoRA:    1× RTX 4090 24GB ≈ $0.50/hr → $2
```

```javascript
// Frontend analogy:

// Full fine-tuning = storing 4K PNG for every image
// QLoRA base model = storing WebP (4× smaller, nearly same quality)
// LoRA adapters = your edits in full resolution on top

// The compression is nearly lossless for inference purposes.
// You can barely tell the quality difference, but the size difference is massive.
```

**Common misconception:** Quantization significantly degrades model quality. Research shows 4-bit QLoRA models perform within 1–2% of their full-precision equivalents on most benchmarks. For practical tasks, the quality trade-off is negligible.

---

## How Fine-tuning Works (Step-by-Step)

```
Step 1: Prepare training data
        (instruction, ideal response) pairs in JSONL format
        ↓
Step 2: Load pre-trained base model
        (quantize to 4-bit for QLoRA)
        ↓
Step 3: Inject LoRA matrices
        (A and B matrices added to attention layers)
        ↓
Step 4: Training loop (frozen base + trainable LoRA)
        → Forward pass: input × (W + B·A) → output
        → Compute loss: how different from ideal response?
        → Backward pass: only LoRA gradients computed
        → Update: only A and B matrices change
        ↓
Step 5: Save LoRA adapter (small file — 10–100 MB)
        ↓
Step 6: Deploy: base model + adapter = your custom model
```

```
Training data quality determines everything:
                                    
  100 excellent examples  ──────────────►  Great results
  10,000 mediocre examples ─────────────►  Mediocre results
  
  Quality > Quantity. Always.
```

---

## Code in Practice

### Preparing Training Data

```python
# training_data.jsonl — one JSON object per line
{"messages": [
  {"role": "system", "content": "You are a customer support agent for TechCorp."},
  {"role": "user", "content": "My laptop won't turn on"},
  {"role": "assistant", "content": "I'm sorry to hear that! Let's troubleshoot step by step. First, is the charging LED lit when you plug in the power adapter? If yes, try holding the power button for 15 seconds to force reset. If the LED is off, try a different outlet and charging cable."}
]}

{"messages": [
  {"role": "system", "content": "You are a customer support agent for TechCorp."},
  {"role": "user", "content": "How do I return a product?"},
  {"role": "assistant", "content": "You can return any product within 30 days of purchase. Visit techcorp.com/returns, enter your order number, and we'll email a prepaid shipping label. Refunds process in 5–7 business days."}
]}
```

**Data quality checklist:**
```
✅ Diverse examples — cover different scenarios, not just easy cases
✅ Consistent response style — same tone and format throughout
✅ Correct information — the model learns errors too!
✅ Natural conversation flow — sounds human, not templated
✅ Edge cases included — what to do when things go wrong
✅ 100–500 high-quality examples minimum
✅ 1,000–5,000 is the sweet spot

❌ Templated/copy-paste responses — model learns repetition
❌ Factual errors — it learns those too
❌ Only happy path — no edge case handling
❌ Fewer than 50 examples — too few to see effect
```

### Fine-tuning with Python (QLoRA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

# 1. Load base model (in 4-bit for QLoRA)
model_name = "meta-llama/Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,       # ← QLoRA quantization
    device_map="auto",        # auto-detect GPU
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Configure LoRA
lora_config = LoraConfig(
    r=16,                     # rank — start here
    lora_alpha=32,            # scaling (usually 2 × r)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # attention layers
    lora_dropout=0.05,        # regularization
    task_type="CAUSAL_LM",
)

# 3. Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → "trainable params: 6,553,600 || all params: 8,030,261,248 || trainable%: 0.0816"
# Only 0.08% of parameters train!

# 4. Training config
training_args = TrainingArguments(
    output_dir="./my-fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,          # standard for LoRA
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# 5. Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
)
trainer.train()

# 6. Save only the LoRA adapter — typically 10–100 MB
model.save_pretrained("./my-lora-adapter")
# The base model (16GB+) is unchanged and not saved again
```

### Using Your Fine-tuned Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base + adapter
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-Instruct")
model = PeftModel.from_pretrained(base, "./my-lora-adapter")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

# Use like any model — adapter is applied automatically
inputs = tokenizer("How do I return a product?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
# Responds in your fine-tuned style!
```

### Fine-tuning via API (No GPU needed)

```python
from openai import OpenAI
client = OpenAI()

# 1. Upload training file
file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# 2. Start job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={"n_epochs": 3}
)

# 3. Use the fine-tuned model (same API, different model ID)
response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:my-org::abc123",  # your fine-tuned model ID
    messages=[{"role": "user", "content": "How do I return a product?"}]
)
```

---

## Evaluating Fine-tuned Models

Always compare base model vs. fine-tuned model on the same held-out test set.

```python
test_prompts = [
    "How do I return a product?",
    "My laptop screen is cracked",
    "Can I get a student discount?",
    # 50–100 test cases you didn't train on
]

results = []
for prompt in test_prompts:
    base_response = base_model.generate(prompt)
    ft_response = finetuned_model.generate(prompt)
    
    # LLM-as-judge (faster than human eval at scale)
    score = judge_llm(f"""
    Rate which response better fits our customer support style:
    
    Query: {prompt}
    Response A (base): {base_response}
    Response B (fine-tuned): {ft_response}
    
    Score each 1–5 on: accuracy, brand voice, format compliance.
    """)
    results.append(score)
```

**Key metrics to track:**
```
1. Task accuracy      — does it do what you trained it to do?
2. Format compliance  — does it follow the expected format?
3. Brand voice        — does it sound like your company?
4. Hallucination rate — is it making things up more than before?
5. General benchmark  — hasn't it broken things it used to do well?
```

**Catastrophic forgetting** — the main risk:
```
Symptom: Model is great at your specific task but lost general abilities.
Example: Fine-tuned on customer support, now can't write code.

Fix: Mix 5–10% general instruction data into your training set.
     This preserves general capabilities while learning your specific task.
```

---

## LoRA Hyperparameter Guide

```
Parameter       What It Does                    Start With    Adjust When
──────────────────────────────────────────────────────────────────────────
r (rank)        LoRA matrix size (capacity)     16            Complex task → 32–64
lora_alpha      Scaling factor                  32 (2 × r)    Usually leave at 2 × r
lora_dropout    Regularization (overfitting)    0.05          Overfitting → 0.1
learning_rate   Step size for updates            2e-4          Unstable training → 1e-4
num_epochs      Full passes through data         3             Small dataset → 5–10
batch_size      Examples per gradient step       4–8           OOM → reduce; slow → increase

Signs of overfitting:
  - Training loss keeps falling ✅ but validation loss goes up ❌
  - Model memorizes training examples word-for-word
  Fix: reduce epochs, increase dropout, add diverse data

Signs of underfitting:
  - Both training and validation loss plateau high
  Fix: increase r, more epochs, more training data
```

---

## Comparison: All the Options

```
┌──────────────────────────────────────────────────────────────────┐
│ Method         │ Cost     │ Time    │ GPU Needed │ Quality       │
├──────────────────────────────────────────────────────────────────┤
│ Prompt Eng.    │ Free     │ Minutes │ None       │ Good          │
│ RAG            │ $        │ Hours   │ None       │ Great (facts) │
│ API Fine-tune  │ $$       │ Hours   │ None       │ Great         │
│ QLoRA          │ $$       │ Hours   │ 1× 24 GB   │ Great         │
│ LoRA (FP16)    │ $$$      │ Hours   │ 1–2× 80 GB │ Excellent     │
│ Full Fine-tune │ $$$$     │ Days    │ 4–8× 80 GB │ Excellent     │
│ Pre-training   │ $$$$$$$  │ Months  │ Thousands  │ N/A           │
└──────────────────────────────────────────────────────────────────┘

Always start from the top. Move down only when simpler methods fail.
```

---

## Gotchas & Pitfalls

```
❌ Fine-tuning for factual knowledge → ✅ Use RAG for facts
   Fine-tuning teaches behavior, not facts. The model will still hallucinate.

❌ Training on too little data → ✅ Minimum ~100 high-quality examples
   Fewer than 50 examples rarely produces noticeable improvement.

❌ Training on inconsistent data → ✅ Audit quality before training
   If your training responses vary in tone/format, the model learns chaos.

❌ Skipping catastrophic forgetting check → ✅ Always run general benchmarks
   A support bot that can't handle common questions is worse than the base model.

❌ Not comparing base vs. fine-tuned → ✅ Always A/B test on a held-out test set
   "The fine-tuned model feels better" is not evaluation. Measure it.

❌ Setting r too high → ✅ Start with r=16, increase only if needed
   High rank = more parameters = overfitting risk on small datasets.

❌ Fine-tuning when prompting would work → ✅ Exhaust prompting first
   Fine-tuning takes hours and $. A 30-minute prompt iteration might solve it.
```

---

## When to Use / When NOT to Use Fine-tuning

**Use fine-tuning when:**
- You need the model to consistently output a specific complex format
- Domain-specific jargon is being misinterpreted by the base model
- You need a specific writing style maintained across thousands of generations
- Prompting alone can't achieve reliable results after multiple iterations
- You need to run a smaller model that matches a larger one's task performance

**Don't use fine-tuning when:**
- You need the model to know facts or access recent information (→ RAG)
- Prompt engineering hasn't been fully exhausted (→ try prompts first)
- You have fewer than ~100 training examples (→ use few-shot prompting)
- The base model already does the task adequately (→ don't fix what isn't broken)
- You need real-time data updates (→ RAG, updated in seconds vs. re-training in hours)

---

## Related Concepts (The Map)

| If you know... | Fine-tuning concept is like... |
|----------------|-------------------------------|
| CSS theming / overrides | LoRA = a small override stylesheet on top of base styles |
| React forking vs. wrapping | Full FT = fork; LoRA = HOC that wraps and extends |
| Browser extensions | LoRA adapters = browser extension that modifies behavior without touching source |
| Image quantization (JPEG) | QLoRA base model = compressed image, nearly same quality |
| Plugin architecture | PEFT adapters = plugins — swap without changing the base |

**Connected topics:**
- **LLM Fundamentals** → pre-training vs. instruction tuning vs. RLHF (fine-tuning continues this chain)
- **RAG** → the alternative when you need knowledge, not behavior change
- **Prompt Engineering** → exhaust this before fine-tuning
- **LLM APIs & SDKs** → OpenAI/Anthropic offer fine-tuning via API (no GPU needed)

---

## Cheat Sheet

| Term | One-line definition |
|------|---------------------|
| Fine-tuning | Continuing training a pre-trained model on your specific data |
| PEFT | Family of methods that update only a fraction of parameters |
| LoRA | Adds two small trainable matrices alongside frozen weights (~0.1% params) |
| QLoRA | LoRA + 4-bit quantized base model = fine-tune 70B on 24GB GPU |
| r (rank) | LoRA matrix capacity — start at 16, increase for complex tasks |
| lora_alpha | Scaling factor for LoRA updates — usually 2× r |
| Catastrophic forgetting | Model loses general ability after fine-tuning on narrow data |
| Adapter | LoRA weight file saved separately from base model (10–100 MB) |

**LoRA quick-start values:**
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
TrainingArguments(
    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
)
```

**Remember these 3 things:**
1. Fine-tune for behavior/style. Use RAG for facts.
2. Quality of training data > quantity of training data
3. Always compare base vs. fine-tuned on a held-out test set

---

## Self-Check Questions

1. **A teammate says "let's fine-tune the model on our product documentation so it knows our products." What do you say?**

<details>
<summary>Answer</summary>
Redirect to RAG instead. Fine-tuning teaches the model how to respond in style, but it will still hallucinate product facts even after training. Worse, facts in training data become stale when products change — you'd need to retrain. RAG gives you up-to-date, accurate answers with source citations, and you can update the knowledge base in seconds without any GPU time.
</details>

2. **What happens if you train LoRA with r=256 on 50 examples?**

<details>
<summary>Answer</summary>
The model will overfit badly. High rank (r=256) gives the LoRA matrices enormous capacity to memorize. With only 50 examples, it will memorize the training set exactly instead of generalizing. Signs: training loss hits near-zero, but the model on new inputs just copies training responses word-for-word. Fix: reduce rank to r=8–16, add more training data, increase dropout.
</details>

3. **You fine-tuned a model for customer support. Now users report it can't write code anymore. What happened and how do you fix it?**

<details>
<summary>Answer</summary>
Catastrophic forgetting: fine-tuning on a narrow domain caused the model to lose general capabilities. Fix: retrain with a mix of your domain data (90%) and general instruction data (10%). This preserves general capabilities while learning your specific task. Tools like Alpaca dataset or the base model's original instruction data work well for the general portion.
</details>

4. **How does QLoRA dramatically cut the GPU memory needed to fine-tune large models?**

<details>
<summary>Answer</summary>
Three key tricks from the QLoRA paper: (1) Quantize the frozen base model to 4-bit using **NF4** (a data-type optimized for normally-distributed weights) — a 70B model drops from ~140GB (BF16) to ~35GB. (2) **Double quantization** — quantize the quantization constants themselves, saving a bit more. (3) **Paged optimizers** — spill optimizer state to CPU when it would OOM the GPU. Because gradients only flow through the tiny LoRA adapters, you never store full-model gradients or optimizer state. Net effect: 70B fits on a single 48GB GPU; 7–13B fits on a 24GB consumer card.
</details>

5. **Your fine-tuned model performs worse than the base model on your task. What do you debug first?**

<details>
<summary>Answer</summary>
In order: (1) Data quality — review 20 random training examples. Are they consistent? Correct? Representative? Bad data is the #1 cause. (2) Overfitting — check if training loss is much lower than validation loss. If so, reduce epochs or increase dropout. (3) Underfitting — if both losses plateau high, try higher r or more data. (4) Evaluation method — are you comparing fairly? Same temperature, same system prompt, same test prompts? Bad comparison methodology is surprisingly common.
</details>

---

## Go Deeper

1. **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** — The original LoRA paper by Hu et al. Read sections 1–3 and the results table. Understanding the math (even at a high level) makes you a much better fine-tuner. (30 min)

2. **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)** — The paper that made 70B fine-tuning accessible. The introduction and Table 1 tell you everything you need to know practically. (20 min)

3. **[Hugging Face PEFT Library](https://huggingface.co/docs/peft/index)** — The go-to library for all PEFT methods. The docs have complete working examples for LoRA, QLoRA, and adapter-based fine-tuning. (2 hours)

4. **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — Production-ready fine-tuning framework that handles all the boilerplate. If you want to fine-tune without writing training loops from scratch, start here. (1 hour setup)

5. **[LLM Fine-tuning Guide by Mistral](https://docs.mistral.ai/guides/finetuning/)** — Excellent practical guide with dataset format examples, hyperparameter recommendations, and evaluation strategies. Provider-agnostic advice despite the source. (1 hour)

---

**What's next?** You now have the complete LLM toolkit:
- [LLM Fundamentals](llm-fundamentals.md) — how LLMs work
- [Prompt Engineering](prompt-engineering.md) — how to talk to them  
- [RAG](rag.md) — how to give them knowledge
- [APIs & SDKs](llm-apis-sdks.md) — how to build apps with them
- [Agents](agents-tool-use.md) — how to let them take actions
- **Fine-tuning** — how to customize them (you're here)

Next phase: MLOps & Production — taking everything to production!
