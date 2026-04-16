# LLM Fundamentals

## What Is an LLM?

A Large Language Model (LLM) is a neural network trained on massive amounts of text that can **generate**, **understand**, and **reason** about language. At its core, it's a next-token predictor — given some text, it predicts what comes next.

```
Input:  "The capital of France is"
LLM:    "Paris" (highest probability next token)
```

But scale changes everything. When you train a Transformer on trillions of tokens with billions of parameters, something remarkable happens — it develops abilities nobody explicitly programmed: reasoning, coding, translation, math, and more. These are called **emergent abilities**.

## Frontend Analogy

```javascript
// Think of an LLM like an incredibly smart autocomplete

// Your IDE autocomplete:
//   - Trained on your project's code
//   - Predicts the next variable/method name
//   - Simple pattern matching

// An LLM autocomplete:
//   - Trained on the entire internet's text
//   - Predicts the next word/token in ANY context
//   - Understands meaning, context, logic, even humor

// Both are "next token prediction" — the difference is scale
```

## The Key Architectures

### GPT (Generative Pre-trained Transformer) — OpenAI

GPT models are **decoder-only** Transformers. They read text left-to-right and generate one token at a time.

```
Architecture: Decoder-only Transformer
Direction:    Left → Right (causal/autoregressive)
Good at:      Text generation, conversation, coding, reasoning

GPT-3:   175B parameters  (2020)
GPT-4:   ~1.8T parameters (2023, estimated, mixture of experts)
GPT-4o:  Multimodal — text, images, audio, video
```

**How GPT generates text:**
```
Step 1: "The"           → predicts "cat"
Step 2: "The cat"       → predicts "sat"
Step 3: "The cat sat"   → predicts "on"
Step 4: "The cat sat on"→ predicts "the"
...each step sees ALL previous tokens but NOTHING ahead
```

### Claude — Anthropic

Claude is also a decoder-only Transformer, but designed with a strong focus on safety, helpfulness, and honesty (the "HHH" framework).

```
Key features:
- Constitutional AI (RLHF + AI feedback for safety)
- Very long context windows (up to 200K tokens)
- Strong reasoning and instruction following
- Extended thinking for complex problems

Claude 3.5 Sonnet → fast, great for coding
Claude Opus 4     → most capable, deep reasoning
Claude Haiku      → fastest, cheapest
```

### Llama — Meta (Open Source)

Meta's open-source LLM family. Important because you can **run it locally** and **fine-tune it**.

```
Llama 2:  7B, 13B, 70B parameters (2023)
Llama 3:  8B, 70B, 405B parameters (2024)

Why it matters:
- Free to use (open weights)
- Can run on your own hardware
- Can fine-tune for your specific use case
- Huge community of fine-tuned variants
```

## How LLMs Are Trained (3 Stages)

Training an LLM is like teaching a person — start broad, then specialize, then refine based on feedback.

### Stage 1: Pre-training (Learn Language)

```
Data:     Trillions of tokens from the internet, books, code
Task:     Predict the next token
Cost:     Millions of $$$, thousands of GPUs, weeks/months
Result:   A "base model" — good at completing text, but not at following instructions

Example base model behavior:
  You: "What is 2+2?"
  Base model: "What is 2+3? What is 2+4? What is 2+5?"
  (It just continues the pattern — it's a text completer, not an assistant)
```

### Stage 2: Supervised Fine-tuning / Instruction Tuning (Learn to Follow Instructions)

```
Data:     Thousands of (instruction, ideal response) pairs
Task:     Learn to follow instructions and give helpful answers
Result:   An "instruct model" — now it answers questions instead of just completing text

Example instruction-tuned behavior:
  You: "What is 2+2?"
  Instruct model: "2+2 equals 4."
```

### Stage 3: RLHF — Reinforcement Learning from Human Feedback (Learn Preferences)

```
Process:
1. Generate multiple responses to the same prompt
2. Humans rank responses from best to worst
3. Train a "reward model" that predicts human preferences
4. Use RL (PPO algorithm) to optimize the LLM to get higher rewards

Before RLHF: "The answer is 4. But some might say 5. It depends on context..."
After RLHF:  "2+2 = 4."

RLHF makes models:
- More helpful (direct, useful answers)
- More harmless (refuses dangerous requests)
- More honest (admits when unsure)
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

## Scaling Laws

One of the most important discoveries in AI: **model performance improves predictably** as you scale up three things:

```
Performance ∝ f(Parameters, Data, Compute)

More parameters  → better (bigger model)
More data        → better (more training text)
More compute     → better (more GPU hours)

The relationship is a smooth power law — no sudden jumps.
This is why companies keep making bigger models: the math says it works.
```

```
             Performance
                 ▲
                 │        ╱
                 │      ╱
                 │    ╱       ← Smooth, predictable improvement
                 │  ╱
                 │╱
                 └──────────► Scale (params × data × compute)
```

### But: Diminishing Returns

Each 10x increase in compute gives roughly the same improvement. Going from GPT-3 to GPT-4 cost ~100x more compute but wasn't 100x better. This is why techniques like RAG, agents, and fine-tuning matter — they're smarter than just "make bigger."

## Tokens — The Language LLMs Speak

LLMs don't read words — they read **tokens**. A token is roughly 3/4 of a word (or about 4 characters).

```
"Hello, world!"      → ["Hello", ",", " world", "!"]         = 4 tokens
"Tokenization"       → ["Token", "ization"]                  = 2 tokens
"ChatGPT is cool"    → ["Chat", "G", "PT", " is", " cool"]  = 5 tokens
```

### Why Tokens Matter for You

```javascript
// As an AI engineer, tokens = cost + speed + limits

// 1. COST — you pay per token
const cost = (inputTokens * inputPrice) + (outputTokens * outputPrice);
// Claude Sonnet: ~$3/M input, ~$15/M output tokens
// GPT-4o: ~$5/M input, ~$15/M output tokens

// 2. SPEED — more tokens = slower response
// 100 tokens ≈ 75 words ≈ takes ~1 second to generate

// 3. CONTEXT WINDOW — max tokens the model can "see"
// Claude:  200K tokens ≈ 150,000 words ≈ ~500 pages
// GPT-4o:  128K tokens ≈ 96,000 words ≈ ~300 pages
```

## Temperature — Controlling Randomness

Temperature controls how "creative" vs "deterministic" the model is.

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

### When to Use What

```
Temperature 0:     Factual Q&A, code generation, data extraction
Temperature 0.3-0.7: General chat, writing, creative tasks
Temperature 1.0+:  Brainstorming, creative writing, poetry
```

## Context Window — The Model's "Working Memory"

The context window is the total number of tokens the model can process at once — this includes both your input AND the model's output.

```javascript
// Think of it like your browser's localStorage — there's a size limit

// Context window = input tokens + output tokens
// ┌─────────────────────────────────────────────┐
// │  System prompt  │  Chat history  │  Response │
// │  (500 tokens)   │  (50K tokens)  │ (2K max)  │
// └─────────────────────────────────────────────┘
//              Total: must fit within context window

// When chat history exceeds the window, older messages get dropped
// This is why long conversations can "forget" earlier context
```

### Context Window Sizes

```
Model               Context Window    ≈ Pages of Text
──────────────────────────────────────────────────────
GPT-4o              128K tokens       ~300 pages
Claude 3.5 Sonnet   200K tokens       ~500 pages
Claude Opus 4       200K tokens       ~500 pages
Llama 3 (8B)        8K tokens         ~20 pages
Gemini 1.5 Pro      1M tokens         ~2,500 pages
```

## Key Concepts Summary

| Concept | What It Means | Why You Care |
|---------|--------------|--------------|
| Parameters | Weights the model learned during training | More = smarter but slower & costlier |
| Tokens | Chunks of text (~4 chars each) | You pay per token, there are limits |
| Context Window | Max tokens model can see at once | Determines how much info you can send |
| Temperature | Randomness dial (0=exact, 1=creative) | Controls response style |
| Pre-training | Learning language from internet text | Creates the base intelligence |
| Instruction Tuning | Learning to follow commands | Makes it an assistant |
| RLHF | Learning from human preferences | Makes it helpful and safe |
| Scaling Laws | Bigger = better (predictably) | Why models keep growing |
| Emergent Abilities | Skills that appear at scale | Reasoning, coding, etc. |

## What's Next?

Now that you understand what LLMs are and how they work, the next step is learning **how to talk to them effectively** — that's [Prompt Engineering](prompt-engineering.md).
