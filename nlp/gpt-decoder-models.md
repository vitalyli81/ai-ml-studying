# GPT & Decoder Models

## What Is It?

GPT (Generative Pre-trained Transformer) is a **decoder-only Transformer** that generates text by predicting **one token at a time**, left to right. It's the architecture behind ChatGPT, Claude, Llama, and every major chatbot/AI assistant.

```
"The cat sat on the" → predict "mat"
"The cat sat on the mat" → predict "."
"The cat sat on the mat." → predict [end]

Each step: see everything before → predict next token → add it → repeat
```

## Frontend Analogy

```javascript
// GPT is like server-sent events (SSE) / streaming response:
// Tokens arrive one at a time, each depending on what came before

const eventSource = new EventSource('/api/chat');
eventSource.onmessage = (event) => {
  // Each token arrives sequentially
  // "The" → "cat" → "sat" → "on" → "the" → "mat"
  appendToOutput(event.data);
};

// You can't skip ahead — each token depends on all previous tokens
// That's exactly how GPT generates text
```

## How GPT Generates Text

### Autoregressive Generation

"Autoregressive" = each output becomes input for the next step:

```
Step 1: Input  "Tell me about"        → Predict "cats"
Step 2: Input  "Tell me about cats"    → Predict "."
Step 3: Input  "Tell me about cats."   → Predict "Cats"
Step 4: Input  "Tell me about cats. Cats" → Predict "are"
...continues until [end] token or max length
```

### Masked Self-Attention (Causal Attention)

Unlike BERT (sees all tokens), GPT **can only look backwards**:

```
Generating: "The cat sat on the mat"

When processing "sat":
  Can see:    "The", "cat", "sat"     ✓
  Cannot see: "on", "the", "mat"      ✗ (future tokens masked)

This is enforced by an attention mask:
Token:    The  cat  sat  on   the  mat
The:      ✓    ✗    ✗    ✗    ✗    ✗
cat:      ✓    ✓    ✗    ✗    ✗    ✗
sat:      ✓    ✓    ✓    ✗    ✗    ✗
on:       ✓    ✓    ✓    ✓    ✗    ✗
the:      ✓    ✓    ✓    ✓    ✓    ✗
mat:      ✓    ✓    ✓    ✓    ✓    ✓
```

## The GPT Evolution

```
GPT-1 (2018):   117M params    — showed language models can be pretrained
GPT-2 (2019):   1.5B params   — "too dangerous to release" — surprisingly coherent text
GPT-3 (2020):   175B params   — few-shot learning, no fine-tuning needed
ChatGPT (2022): GPT-3.5 + RLHF — conversation, instruction following
GPT-4 (2023):   ~1.8T params  — multimodal (text + images), strong reasoning
GPT-4o (2024):  Optimized      — faster, cheaper, same quality
```

### The Key Insight: Scale → Emergent Abilities

```
Small model:   Can complete sentences
Medium model:  Can answer questions
Large model:   Can reason, write code, follow complex instructions
Huge model:    Can do math, plan, use tools, think step-by-step

Nobody programmed these abilities — they EMERGED from scale.
```

## How GPT Learns: Pre-training + Alignment

### Step 1: Pre-training (Unsupervised)

Train on the entire internet to predict next tokens:

```
Training data: "The capital of France is ___"
Model learns:  "Paris" (from millions of web pages)

Training data: "def fibonacci(n):\n    ___"
Model learns:  "if n <= 1: return n" (from millions of code files)
```

Result: a model that knows facts, grammar, code, reasoning patterns — but it's not helpful yet. It just completes text.

### Step 2: Instruction Fine-tuning (Supervised)

Train on human-written (prompt, response) pairs:

```
Prompt:   "Explain photosynthesis in simple terms"
Response: "Plants use sunlight, water, and CO2 to make food and oxygen..."

Prompt:   "Write a Python function to sort a list"
Response: "def sort_list(lst):\n    return sorted(lst)"
```

Result: the model follows instructions instead of just completing text.

### Step 3: RLHF (Reinforcement Learning from Human Feedback)

Humans rank multiple responses. The model learns which responses humans prefer:

```
Prompt: "Is it safe to mix bleach and ammonia?"

Response A: "Yes, mixing bleach and ammonia creates a powerful cleaner."  ← BAD
Response B: "No! This creates toxic chloramine gas. Never mix them."      ← GOOD

Human ranks B > A → model learns to prioritize safety and accuracy
```

This is what makes ChatGPT/Claude helpful, harmless, and honest.

## Open-Source Decoder Models

| Model | Creator | Params | Strength |
|-------|---------|--------|----------|
| **Llama 3** | Meta | 8B-70B | Best open-source general purpose |
| **Mistral** | Mistral AI | 7B | Excellent for its size |
| **Mixtral** | Mistral AI | 8x7B | Mixture of experts, strong reasoning |
| **Phi-3** | Microsoft | 3.8B | Surprisingly good for tiny size |
| **Gemma** | Google | 2B-7B | Good for mobile/edge deployment |
| **CodeLlama** | Meta | 7B-34B | Specialized for code |

## Using GPT-Style Models

### Text Generation with Hugging Face

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

result = generator(
    "The future of artificial intelligence is",
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True,
)
print(result[0]['generated_text'])
```

### Using LLM APIs (What You'll Do Most as AI Engineer)

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain React hooks in 3 sentences."}
    ]
)
print(message.content[0].text)
```

### Key Generation Parameters

```python
# Temperature: controls randomness
temperature=0.0   # deterministic — always same output (good for factual tasks)
temperature=0.7   # balanced (good default for most tasks)
temperature=1.0   # creative (good for brainstorming, stories)

# Top-p (nucleus sampling): limits the token pool
top_p=0.9         # sample from tokens covering 90% of probability

# Max tokens: how long the response can be
max_tokens=100    # short response
max_tokens=4096   # long response

# Stop sequences: stop generating when you see this string
stop=["\n\n", "END"]
```

## In-Context Learning (The GPT Superpower)

GPT can learn new tasks **from examples in the prompt** — no fine-tuning needed:

### Zero-Shot (No Examples)

```
Classify this review as positive or negative:
"The battery life is incredible" → positive
```

### Few-Shot (A Few Examples)

```
Classify reviews:
"Great product!" → positive
"Terrible quality" → negative
"Works perfectly" → positive
"Broke after a week" → negative

"The battery life is incredible" → ???

GPT: "positive"   (learned the pattern from examples!)
```

### Chain-of-Thought (Reasoning)

```
Q: If a shirt costs $25 and is 20% off, what do you pay?
A: Let me think step by step.
   Original price: $25
   Discount: 20% of $25 = $5
   Final price: $25 - $5 = $20
```

Adding "Let me think step by step" dramatically improves accuracy on reasoning tasks.

## BERT vs GPT — Complete Comparison

| | BERT (Encoder) | GPT (Decoder) |
|---|---|---|
| **Direction** | Bidirectional (sees all) | Left-to-right (causal) |
| **Training** | Masked language model | Next token prediction |
| **Output** | Embeddings/classifications | Generated text |
| **Strength** | Understanding | Generation |
| **Tasks** | Classification, NER, search | Chatbots, writing, code gen |
| **Sizes** | 110M-340M | 7B-1.8T |
| **Fine-tune for** | Specific tasks (cheap) | Instruction following (expensive) |
| **Use via** | Hugging Face locally | API calls (or local for open models) |

## The Decoder → LLM Pipeline (What Comes Next)

```
Decoder models are the foundation of LLMs:

Decoder Architecture
    ↓ scale up
GPT-3 / Llama (pre-trained)
    ↓ instruction fine-tuning
ChatGPT / Claude (helpful assistant)
    ↓ add tools
AI Agents (can browse, code, search)
    ↓ add retrieval
RAG Systems (grounded in your data)

Your next phase (LLMs & AI Engineering) builds directly on this.
```

## Key Takeaway

GPT-style decoder models generate text **one token at a time**, left to right. They're pre-trained on massive text to learn language, then aligned with human preferences (RLHF) to be helpful. The key innovation is **in-context learning** — they can learn new tasks from examples in the prompt without any training. As an AI Engineer, you'll mostly interact with these models via **APIs** (Claude, GPT) and use techniques like prompt engineering, few-shot learning, and chain-of-thought to get the best results. This is the direct bridge to Phase 5 (LLMs & AI Engineering).
