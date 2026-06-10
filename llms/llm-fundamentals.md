# LLM Fundamentals

## TL;DR

LLMs are neural networks trained to predict the next word — and at massive scale, this simple trick produces systems that can reason, code, translate, and more. They don't "understand" like humans, but they've seen so much text that they've compressed patterns of human knowledge into billions of numerical weights. You talk to them through tokens; they respond one token at a time.

> 💡 **Key Insight:** Everything an LLM does — coding, math, storytelling — is an emergent consequence of one task: "predict the next token."

---

## The Mental Model

**Think of an LLM like an impossibly well-read intern.**

They've read the entire internet, every book, every code repo. They can't look things up anymore — it's all in their head. When you ask them something, they don't search; they recall patterns from everything they've read and synthesize an answer.

| Real world | Technical concept |
|------------|------------------|
| Intern reads millions of books | Pre-training on trillions of tokens |
| Intern recalls patterns from reading | Model weights store compressed knowledge |
| Intern writes one sentence at a time | Autoregressive token generation |
| Intern gets feedback and improves | RLHF (Reinforcement Learning from Human Feedback) |
| Intern can only focus on so much at once | Context window limit |

---

## Build the Intuition From Zero

Three things about LLMs constantly trip up engineers and lead to real bugs: **the model has no memory between calls, the "context window" is its only working memory, and "smart" doesn't mean "knows true facts."** Get these and you'll design AI features correctly.

### Idea 1: Each API call is a blank slate — the model has no memory

This is the #1 misconception. An LLM does **not** remember your last message. Every API call, it starts from total amnesia. A chatbot "remembers" the conversation only because **your code resends the entire history every time**:

```
You think:                          What actually happens on call #2:
  Turn 1: "My name is Sam"            you send: [all of turn 1]  +  "What's my name?"
  Turn 2: "What's my name?"           the model re-reads the WHOLE thing fresh, answers "Sam"
  (model "remembers" Sam)             → it never stored anything; YOU replayed the transcript
```

So "memory" is an illusion you build by accumulating and resending text. This is why long chats get expensive (you resend more each turn) and why "it forgot what I said earlier" happens (the history fell out of the window — next idea).

### Idea 2: The context window is the model's entire desk

The **context window** is the maximum tokens the model can look at in one call — system prompt + your history + retrieved docs + its own reply, all of it. Picture a desk of fixed size:

```
┌──────────────── context window (e.g. 200,000 tokens) ─────────────────┐
│ [system prompt] [conversation history] [retrieved RAG docs] [your Q] [room for the answer] │
└────────────────────────────────────────────────────────────────────────┘
   everything the model "knows right now" must fit on this desk.
   Overflow → oldest stuff falls off the edge → "it forgot."
```

Everything the model reasons about must be *on the desk at once*. It can't "go look something up" elsewhere — if a fact isn't in the window (or in its trained weights), it doesn't exist to the model. This is exactly why [RAG](rag.md) exists: it *fetches* the right documents and places them on the desk before asking.

### Idea 3: "Fluent" is not "correct" — why it hallucinates

The model was trained to produce **plausible** next tokens, not **true** ones. When it doesn't know something, it doesn't stop — it generates the most likely-sounding continuation, which can be confidently wrong (a **hallucination**):

```
"The 2019 paper by Dr. Smith on quantum biology found..."
   → the model has no such paper, but "<plausible finding>" is a likely continuation,
     so it invents one, in the same confident tone as a true fact.
```

It's not lying — it has no concept of "I don't know" unless trained/prompted to. This is why production LLM features need grounding ([RAG](rag.md)), guardrails ([../ml-ops/safety-guardrails.md](../ml-ops/safety-guardrails.md)), and evals ([evals.md](evals.md)) — you can't trust fluency as truth.

> 💡 **The engineer's mental model:** an LLM is a stateless function — `text in → text out` — with a fixed-size desk (context) and no built-in truth-checking. Memory, knowledge, and reliability are things *you* engineer around it, not properties it has. (The "predict next token → emergent skill" mechanism is in [../nlp/gpt-decoder-models.md](../nlp/gpt-decoder-models.md).)

---

## Why It Exists (Problem → Solution)

**The problem:** Earlier AI systems needed hand-crafted rules for each task. Want a translator? Engineer linguistic rules. Want a summarizer? Engineer summarization logic. It was brittle, expensive, and didn't generalize.

**What came before:** RNNs and LSTMs could handle sequences but struggled with long-range dependencies — they "forgot" context from 50 words ago.

**The breakthrough:** The Transformer architecture (2017, "Attention is All You Need") introduced self-attention, letting models relate any token to any other token in the sequence, regardless of distance. Then researchers discovered: train this on enough text with enough parameters, and a single model learns to do *everything*.

**What changed:** Instead of building 100 specialized models, you build one large one and prompt it.

---

## Core Concepts

### 1. Tokens — The Language LLMs Speak

**Plain English:** Tokens are the chunks of text an LLM processes. Not quite words, not quite characters — somewhere in between.

**Analogy:** If words are Lego sets, tokens are individual Lego pieces. "Tokenization" breaks text into its smallest meaningful pieces.

**Technical explanation:** Tokenization uses algorithms like BPE (Byte Pair Encoding) to split text into subword units. Common words are single tokens; rare words are split.

```
"Hello, world!"      → ["Hello", ",", " world", "!"]         = 4 tokens
"Tokenization"       → ["Token", "ization"]                  = 2 tokens
"ChatGPT is cool"    → ["Chat", "G", "PT", " is", " cool"]  = 5 tokens
```

```javascript
// As an AI engineer, tokens = cost + speed + limits

const cost = (inputTokens * inputPrice) + (outputTokens * outputPrice);
// Prices change — always check the provider's pricing page.
// Ballpark as of 2026:
//   Claude Haiku:  ~$1/M input,   ~$5/M output
//   Claude Sonnet: ~$3/M input,   ~$15/M output
//   Claude Opus:   ~$15/M input,  ~$75/M output

// Context Window — max tokens the model can "see"
// Claude:  200K tokens ≈ 150,000 words ≈ ~500 pages
// GPT-4o:  128K tokens ≈ 96,000 words ≈ ~300 pages
```

**Common misconception:** People think tokens = words. Actually 1 token ≈ 0.75 words. "Tokenization" is 2 tokens, not 1.

---

### 2. Parameters — Where Knowledge Lives

**Plain English:** Parameters are the millions/billions of numbers that define how the model behaves. They're the model's "memory" of everything it learned during training.

**Analogy:** Think of parameters like the settings on a massive mixing board with 70 billion knobs. Training adjusts every knob until the music (outputs) sounds right.

**Technical explanation:** Parameters are weight values in the neural network's matrices. When you multiply the input (your prompt as numbers) through these matrices, you get the output (the next token probabilities).

```
GPT-3:   175 billion parameters (2020, dense)
GPT-4:   Not disclosed — widely rumored ~1.8T total via mixture-of-experts,
         ~280B active per token. Treat exact numbers as speculation.
Llama 3: 8B, 70B, 405B parameter variants (dense, open weights)
Claude:  Not disclosed
```

> **Sidebar — Dense vs. MoE:** A "dense" model activates all parameters for every token. A **Mixture of Experts (MoE)** model has many "expert" sub-networks and routes each token through only a few of them — total params are huge, but compute per token stays cheap. GPT-4, Mixtral, and DeepSeek-V3 are MoE; Llama 3 is dense.

**Common misconception:** More parameters always = better. In practice, a well-trained smaller model (e.g., Llama 3 8B) often beats a poorly-trained larger one on specific tasks.

---

### 3. Context Window — The Model's Working Memory

**Plain English:** The context window is the total text the model can "see" at once — your prompt plus the conversation history plus its response.

**Analogy:** Like RAM in a computer. You can have 2TB on disk, but you can only process what fits in RAM right now. The context window is the LLM's RAM.

```javascript
// Context window = input tokens + output tokens
// ┌─────────────────────────────────────────────┐
// │  System prompt  │  Chat history  │  Response │
// │  (500 tokens)   │  (50K tokens)  │ (2K max)  │
// └─────────────────────────────────────────────┘
//              Total: must fit within context window

// When chat history exceeds the window, older messages get dropped
```

**Context Window Sizes:**
```
Model               Context Window    ≈ Pages of Text
──────────────────────────────────────────────────────
GPT-4o              128K tokens       ~300 pages
Claude Sonnet 4.x   200K tokens       ~500 pages (1M in beta)
Claude Opus 4.x     200K tokens       ~500 pages
Gemini 1.5 Pro      1M tokens         ~2,500 pages
Llama 3 (8B)        8K tokens         ~20 pages
```

**Common misconception:** Bigger context window = the model uses all of it well. In reality, models often "lose focus" on information buried in the middle of very long contexts (the "lost in the middle" problem).

---

### 4. Temperature — The Creativity Dial

**Plain English:** Temperature controls how random vs. predictable the model's outputs are.

**Analogy:** Think of it like a chef who either follows the recipe exactly (temp=0) or improvises wildly (temp=2). Same ingredients, very different dishes.

```
Temperature = 0.0  → Always picks the most likely token (deterministic)
Temperature = 0.7  → Some randomness (good default for chat)
Temperature = 1.0  → More creative/varied
Temperature = 1.5+ → Wild, often nonsensical

Prompt: "The best programming language is"

temp=0.0: "Python" (always)
temp=0.7: "Python" / "JavaScript" / "it depends on the use case"
temp=1.5: "a rubber duck that speaks Haskell" (too random)
```

```
When to use what:
Temperature 0:       Factual Q&A, code generation, data extraction
Temperature 0.3-0.7: General chat, writing, creative tasks
Temperature 1.0+:    Brainstorming, creative writing, poetry
```

**Common misconception:** Temperature 0 means the model is "certain." It's still probabilistic — it just always picks the single highest-probability token, which can still be wrong.

---

### 5. Emergent Abilities

**Plain English:** Skills that nobody explicitly programmed — they just appeared when the model got big enough.

**Analogy:** Water molecules don't have "wetness." But enough of them together produces wetness — an emergent property. Similarly, individual parameters don't "understand" — but enough of them together produce something that looks like understanding.

**Technical explanation:** Researchers have observed that capabilities like multi-step arithmetic, code debugging, and logical reasoning appear suddenly (not gradually) as model scale increases past certain thresholds.

```
Small model (1B params):  Can autocomplete text
Medium model (7B params): Can answer questions, translate
Large model (70B+ params): Can reason, write code, do math,
                            understand nuance, follow complex instructions
```

**Common misconception:** Emergent abilities prove LLMs are "intelligent." They might just be very sophisticated pattern matching that *looks* like reasoning — the debate is ongoing.

---

> 🧠 **Quick recall — answer out loud before scrolling on** (all answers are above):
> 1. Why does a chatbot "remember" your name — what is actually happening on call #2?
> 2. The context window is the model's ___ — and what happens to facts that aren't on it?
> 3. Why does the model hallucinate instead of saying "I don't know"?
> 4. Roughly: 1 token ≈ how many words, and which costs more — input or output tokens?
> 5. Temperature 0 — deterministic, accurate, both, or neither?

---

## How LLMs Are Trained (Step-by-Step)

Training an LLM happens in 3 stages. Think of it as: teaching someone to read → teaching them to follow instructions → teaching them what's helpful.

```
Stage 1: Pre-training         Stage 2: Instruction Tuning    Stage 3: RLHF
─────────────────────         ──────────────────────────     ─────────────
Internet text                 (instruction, answer) pairs    Human rankings
     │                               │                             │
     ▼                               ▼                             ▼
Predict next token            Learn to follow commands        Learn preferences
(unsupervised)                (supervised)                    (reinforcement)
     │                               │                             │
     ▼                               ▼                             ▼
Base model                    Instruct model                  Chat model
"completes text"              "answers questions"             "helpful & safe"
```

### Stage 1: Pre-training

```
Data:     Trillions of tokens from web, books, code (CommonCrawl, Wikipedia, GitHub, etc.)
Task:     Predict the next token — that's it
Cost:     Millions of dollars, thousands of GPUs, weeks/months
Result:   A "base model" — brilliant at pattern continuation, but not at following instructions

Example base model behavior:
  You: "What is 2+2?"
  Base model: "What is 2+3? What is 2+4? What is 2+5?"
  (It's a text completer — sees your question as the start of a list)
```

### Stage 2: Instruction Fine-tuning

```
Data:     Thousands of (instruction, ideal response) pairs — human-curated
Task:     Learn to respond helpfully, not just complete text
Result:   An "instruct model" — now it actually answers instead of continuing

  You: "What is 2+2?"
  Instruct model: "2+2 equals 4."
```

### Stage 3: RLHF (Reinforcement Learning from Human Feedback)

```
Process:
  1. Generate multiple responses to the same prompt
  2. Human raters rank responses best → worst
  3. Train a "reward model" that predicts human preferences
  4. Use RL (PPO algorithm) to optimize the LLM for higher reward

Before RLHF: "The answer is 4. But some might argue 5 in non-standard systems..."
After RLHF:  "2+2 = 4."

RLHF makes models:
  ✅ More helpful — direct, useful answers
  ✅ More harmless — refuses dangerous requests
  ✅ More honest — admits uncertainty
```

### Visual Summary

```
Internet Text (trillions of tokens)
        │
        ▼
┌─────────────────┐
│  Pre-training    │  "Learn language itself"
│  (Base Model)    │  Cost: $$$$$
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Instruction     │  "Learn to follow instructions"
│  Fine-tuning     │  Cost: $$
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RLHF           │  "Learn human preferences"
│  (Alignment)    │  Cost: $$
└────────┬────────┘
         │
         ▼
   Chat-ready LLM (GPT-4, Claude, etc.)
```

---

## Attention — The Single Idea That Made LLMs Work

Before architectures, understand this one mechanism. Everything else is scaffolding around it.

**Plain English:** When the model generates the next token, it looks at every previous token and decides *how much each one matters* for this prediction. That weighted look-back is **attention**.

**Analogy:** You're translating the sentence *"The cat that the dog chased was black."* To pick the right word for "was," you have to know the subject is "cat," not "dog." Your eyes snap back to "cat" — not equally to every word, but **more to the relevant ones**. Attention is that snap-back, made mathematical.

**How it works (intuition, not linear algebra):**

```
Generating the next token for:  "The cat sat on the ___"

For each token, the model computes 3 vectors:
  Query  (Q) — "what am I looking for right now?"
  Key    (K) — "what do I offer?"
  Value  (V) — "what information do I carry?"

Raw score for token i looking at token j:
  s(i, j) = Q_i · K_j / √d              (dot product, scaled by √d for stability)

Turn the row of scores into weights that sum to 1:
  attention(i, :) = softmax( [s(i, 1), s(i, 2), ..., s(i, n)] )

New representation of token i = Σ_j attention(i, j) × V_j

Intuitively for predicting the word after "on the":
  - "sat"  → high attention (tells us something is seated)
  - "cat"  → high attention (the subject — "on the mat"? "on the chair"?)
  - "The"  → low attention (not semantically useful here)

Note: In decoder-only models (GPT/Claude/Llama), a **causal mask** prevents
attending to future tokens — position i can only see positions ≤ i.
```

**Why this was the breakthrough (vs. RNNs):**

```
RNN (pre-2017):                    Transformer (2017+):
  token 1 → hidden state             every token sees every other
    ↓        (compressed summary)    token DIRECTLY — no bottleneck
  token 2 → hidden state
    ↓        (summary degrades)      Parallelizable on GPUs
  token 3 → hidden state             → train on trillions of tokens
    ↓                                → enables scale → enables LLMs
  token 100 → "what was token 1?"
              (forgotten)
```

RNNs had to squeeze all context into a single running hidden state — long-range info decayed. Attention lets token 5000 look directly at token 1 with no decay. That single change, plus GPU-parallelizable math, is what unlocked the scale era.

**Multi-head attention:** The model runs many attention operations in parallel (e.g., 32 "heads"), each learning to focus on different kinds of relationships — one head might track subject-verb agreement, another might track coreference ("it" → "the cat"), another might track syntax. You don't program what each head does; they specialize during training.

**Common misconception:** ❌ "Attention is how the model 'thinks'." ✅ Attention is how the model **routes information** between tokens. Thinking — if we want to call it that — emerges from stacking many attention + feedforward layers (typically 32–100+) on top of each other.

> 💡 **Key Insight:** A Transformer is mostly just: attention layer → feedforward layer → repeat 32+ times. Simple building block, insane scale.

---

## The Key Architectures

### GPT (Generative Pre-trained Transformer) — OpenAI

**Decoder-only** Transformers. They read text left-to-right and generate one token at a time.

```
Architecture: Decoder-only Transformer
Direction:    Left → Right (autoregressive)
Good at:      Text generation, conversation, coding, reasoning

Step 1: "The"           → predicts "cat"
Step 2: "The cat"       → predicts "sat"
Step 3: "The cat sat"   → predicts "on"
...each step sees ALL previous tokens but NOTHING ahead
```

### Claude — Anthropic

Also decoder-only, but designed with safety as a core principle through **Constitutional AI**.

```
Key features:
  - Constitutional AI (RLHF + AI feedback for safety)
  - Very long context windows (up to 200K tokens)
  - Extended thinking for complex problems

Claude Haiku      → fastest, cheapest
Claude Sonnet     → fast, great for coding (best balance)
Claude Opus       → most capable, deep reasoning
```

### Llama — Meta (Open Source)

Run it locally. Fine-tune it. It's free.

```
Llama 3: 8B, 70B, 405B parameters (2024)

Why it matters:
  - Free to use (open weights)
  - Can run on consumer hardware
  - Can fine-tune for your specific use case
  - Powers thousands of specialized variants (CodeLlama, etc.)
```

### Architecture Comparison

```
                    GPT-4o          Claude          Llama 3
────────────────────────────────────────────────────────────
Type            Decoder-only    Decoder-only    Decoder-only
Context window  128K            200K            8K–128K
Open source     ❌              ❌              ✅
Run locally     ❌              ❌              ✅
Fine-tune       Via API only    Via API only    ✅ Full access
Safety approach RLHF            Constitutional  Community
```

---

## Scaling Laws

**Model performance improves predictably** as you scale parameters, data, and compute. This is the most important empirical finding in modern AI — and the reason frontier labs spend hundreds of millions on single training runs.

### The law (in one line)

Loss on held-out text falls as a **power law** in three knobs: parameter count **N**, training tokens **D**, and compute **C**. Double any one (holding others fixed, within bounds) and loss drops by a predictable amount.

```
             Test loss (lower = better)
                 ▲
                 │╲
                 │ ╲
                 │  ╲___        ← Power law: log(loss) ∝ -log(scale)
                 │      ╲___
                 │           ╲___
                 └────────────────► log(Parameters or Data or Compute)
```

### The two landmark papers (worth knowing by name)

| Paper | Year | Key finding |
|-------|------|-------------|
| **Kaplan et al. ("Scaling Laws")** | 2020 | Loss follows smooth power laws in N, D, C. Bigger seemed better. |
| **Hoffmann et al. ("Chinchilla")** | 2022 | GPT-3-era models were **undertrained** — for a fixed compute budget, you should roughly balance parameters and tokens (~20 tokens per parameter). |

Chinchilla's correction is the practical one: you can't just make a model bigger; you have to feed it proportionally more data. A 70B model trained on 300B tokens is wasted — it wants ~1.4T tokens.

### Why this matters for you as an AI Engineer

You won't train a frontier model. But scaling laws drive the decisions you *do* make:

1. **Why new models keep getting better** — labs are moving along a known curve, not hoping for miracles. Plan roadmaps assuming frontier capabilities continue to improve ~yearly.
2. **Why smaller models keep catching up** — better data and training recipes (Llama 3, Phi, Gemma) push the same capability down to smaller sizes. The cheap model this year ≈ the frontier model two years ago. Rerun your model-selection evals every ~6 months.
3. **Why "just use the biggest model" is lazy** — returns diminish sharply. A task that Haiku solves at 95% doesn't need Opus at 96% for 60× the cost.
4. **Why RAG / agents / fine-tuning matter** — they're *orthogonal* axes of improvement. Stacking a smart pipeline on a mid-tier model often beats a naive call to a flagship one.

### Common misconception

❌ "Scaling laws guarantee we'll get AGI by making bigger models."
✅ The laws describe **loss on next-token prediction**, not capability in the real world. Emergent abilities, reasoning quality, and usefulness don't follow the same clean curve — and data is finite. Scaling is *necessary* for progress, not sufficient.

---

## Gotchas & Pitfalls

```
❌ "The LLM knows the answer" → ✅ It generates plausible tokens. It can confidently
                                    produce wrong information (hallucination).

❌ "Bigger model = always better" → ✅ A fine-tuned 7B model often beats GPT-4 on
                                       a specific narrow task.

❌ "Temperature 0 = reliable" → ✅ It's deterministic, not accurate. Still makes mistakes.

❌ "Long context = it remembers everything" → ✅ Models lose focus on middle-of-context
                                                 info ("lost in the middle" problem).

❌ "The model understands my intent" → ✅ It pattern-matches on your words. Ambiguous
                                          prompts produce ambiguous outputs.

❌ "Tokens = words" → ✅ 1 token ≈ 0.75 words. "Tokenization" = 2 tokens.

❌ "Fine-tuning teaches the model new facts" → ✅ Fine-tuning teaches behavior/style.
                                                  For facts, use RAG.
```

---

## When to Use Which Model

| Scenario | Best choice | Why |
|----------|-------------|-----|
| Simple classification/extraction | Claude Haiku or GPT-4o-mini | Cheap, fast, good enough |
| Code generation / debugging | Claude Sonnet | Excellent coding performance |
| Complex reasoning, long docs | Claude Opus or GPT-4o | Maximum capability |
| Privacy-sensitive tasks | Llama 3 (local) | Data never leaves your machine |
| High volume, cost-sensitive | Haiku / GPT-4o-mini | 60× cheaper than flagship |
| Fine-tuning needed | Llama 3 | Open weights, full control |

---

## Production Notes

### Cost (2026 ballpark — always check current pricing)

| Tier | Example model | Input $/1M tok | Output $/1M tok | Use for |
|------|---------------|----------------|-----------------|---------|
| Small | Haiku / GPT-4o-mini / Gemini Flash | ~$0.25–1 | ~$1–4 | High-volume classification, extraction, routing |
| Mid | Sonnet / GPT-4o | ~$3 | ~$15 | Default production workhorse |
| Flagship | Opus / o1 | ~$15 | ~$75 | Hard reasoning, agent planners, code |

**Back-of-envelope rule:** output tokens cost ~4–5× input tokens. A chatbot answering with 300 tokens on 2K input on Sonnet ≈ $0.011 per turn → 100K turns/month ≈ $1,100.

### Latency (typical p50 / p95)

| Op | p50 | p95 | Notes |
|----|-----|-----|-------|
| Time-to-first-token (TTFT) | 300–600 ms | 1–2 s | Streaming makes this the number users *feel* |
| Per-output-token | 20–60 ms | 80–150 ms | Flagship is slower than small tier |
| Full response (300 tok) | 6–15 s | 15–30 s | Never block UI on this — stream or async |

Longer context inflates TTFT (loading + attention scales). 100K-token prompts can push p95 TTFT to 3–5 s even on mid-tier.

### Failure modes to plan for

- **5xx / overloaded** — provider capacity events happen; retry with exponential backoff + jitter, and keep a fallback provider.
- **Rate limits (429)** — respect `Retry-After`; at scale, batch or queue.
- **Hallucination on facts** — the model will confidently invent. Ground with RAG or tools; never trust unsourced factual claims.
- **"Lost in the middle"** — long contexts bury information placed in the middle. Put critical instructions and retrieved facts near the top *and* bottom.
- **Silent model swaps** — providers sometimes ship snapshot updates. Pin model versions (`claude-sonnet-4-6` not `claude-sonnet-latest`) and run evals before upgrading.

### What to monitor

- **Cost per request** and **$/day by feature** — the number one surprise bill source.
- **TTFT + full-response p50/p95** per endpoint.
- **Output-token distribution** — a sudden long-tail spike usually means a prompt bug (loops, repeated apologies).
- **Error rate by type** (429, 5xx, timeout, validation) — alert on each separately.
- **Hallucination / accuracy** via a small online eval sample (LLM-as-judge on 1–5% of traffic).

See [production-llm-patterns.md](production-llm-patterns.md) for caching, retries, and fallbacks; [../ml-ops/llm-observability.md](../ml-ops/llm-observability.md) for how to wire the dashboards.

---

## Related Concepts (The Map)

| If you know... | LLM concept is like... |
|----------------|----------------------|
| Database queries | Prompt = query, model = database of learned patterns |
| REST APIs | LLM API = stateless request/response (send full context each time) |
| Autocomplete (IDE) | LLMs are autocomplete at massive scale |
| React state | Context window = useState — only what's in state exists to the component |
| Compression algorithms | Training = compressing all of human text into weights |

**Next topics to connect:**
- **Prompt Engineering** → how to talk to LLMs effectively
- **RAG** → give LLMs access to knowledge outside their training
- **Agents** → let LLMs take actions in the world
- **Fine-tuning** → customize model behavior for your domain

---

## Cheat Sheet

| Term | One-line definition |
|------|---------------------|
| Token | Chunk of text (~0.75 words) — the unit of LLM I/O |
| Parameter | Numerical weight learned during training — where knowledge lives |
| Context window | Max tokens the model processes at once (prompt + response) |
| Temperature | Randomness dial: 0 = deterministic, 1+ = creative |
| Pre-training | Training on raw internet text to predict next tokens |
| Instruction tuning | Fine-tuning on (instruction, answer) pairs |
| RLHF | Training with human preference rankings for safety/helpfulness |
| Scaling laws | Bigger (params × data × compute) = predictably better |
| Emergent abilities | Capabilities that appeared at scale, not programmed |
| Hallucination | Model generates confident but false information |
| Base model | Pre-trained only — completes text, doesn't follow instructions |
| Instruct model | Base + instruction tuning — follows commands |

**Core formula:** `output = argmax(P(next_token | all_previous_tokens))`

**Remember these 3 things:**
1. LLMs predict the next token — everything else is emergent from that
2. Tokens cost money; context window limits what the model "sees"
3. Base model ≠ chat model — 3 training stages separate them

---

## Self-Check Questions

1. **What is an LLM actually doing when it generates a response?**

<details>
<summary>Answer</summary>
It's predicting the probability distribution over the next token, sampling from that distribution, then repeating the process. Each token becomes part of the context for predicting the next. It's autoregressive — it generates one token at a time.
</details>

2. **Why does a base model just repeat patterns instead of answering questions?**

<details>
<summary>Answer</summary>
Because pre-training only teaches next-token prediction on raw internet text. If you show it "What is 2+2?", it's seen that pattern followed by more questions in lists, so it continues the pattern. Instruction fine-tuning teaches it to respond as an assistant instead.
</details>

3. **If you have a 200K token context window, why can't you just put your entire codebase in the prompt?**

<details>
<summary>Answer</summary>
Several reasons: (1) cost — you pay for every input token, (2) the "lost in the middle" problem — models lose focus on info in the middle of huge contexts, (3) irrelevance — flooding the model with noise makes it harder to focus on the relevant parts. RAG solves this by fetching only relevant chunks.
</details>

4. **What's the difference between temperature and top_p?**

<details>
<summary>Answer</summary>
Temperature scales the probability distribution (high temp = flatter = more random). Top_p (nucleus sampling) truncates to the top p% of the probability mass before sampling. Both control randomness but via different mechanisms. In practice, most engineers just tune temperature and leave top_p at default (1.0).
</details>

5. **Why do scaling laws matter for you as an AI engineer?**

<details>
<summary>Answer</summary>
Because they tell you that improvements are predictable and continuous — there's no magic ceiling. But they also tell you returns are diminishing, which is why smart techniques (RAG, prompting, fine-tuning) often beat just using a bigger model. You don't always need the biggest model.
</details>

---

## Go Deeper

1. **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** — The original Transformer paper. Read the abstract and look at Figure 1. Understanding self-attention is the single most important thing for an AI engineer. (30 min)

2. **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** by Jay Alammar — The best visual explanation of how Transformers work. If you read one resource, make it this. (45 min)

3. **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)** — Kaplan et al.'s paper that established the scaling laws. Even just reading the abstract and conclusion gives you critical mental models. (20 min)

4. **[Andrej Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)** — Best hands-on video for understanding what's actually happening inside an LLM. You'll build a small GPT in PyTorch. Essential. (2 hours)

5. **[Tiktokenizer](https://tiktokenizer.vercel.app/)** — Interactive tokenizer playground. Paste any text and see exactly how it gets tokenized. Spend 10 minutes here and you'll understand tokens better than most people. (10 min)

---

**What's next?** Now that you understand what LLMs are and how they work, learn **how to talk to them effectively**: [Prompt Engineering →](prompt-engineering.md)
