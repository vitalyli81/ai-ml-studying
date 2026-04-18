# LLM APIs & SDKs

## TL;DR

LLM APIs let you integrate AI into your applications by sending HTTP requests and getting AI-generated responses back. As a frontend developer, this is your most natural entry point — you already know how to call APIs and handle async operations. The Anthropic SDK wraps the API with TypeScript-first ergonomics; the Vercel AI SDK wraps multiple providers for React/Next.js chat UIs. Master streaming, structured outputs, and prompt caching — those three features separate toy demos from production apps.

> 💡 **Key Insight:** LLM APIs are stateless — you send the full conversation history every request. There are no sessions. This changes how you architect your app's state management.

---

## The Mental Model

**Think of an LLM API like a very smart database query.**

You send a structured request (your "query" = the prompt), you get a structured response (the "result" = the generated text). The difference: the "database" is a billion-parameter neural network that generates novel responses instead of returning stored records.

| Real world | Technical concept |
|------------|------------------|
| SQL query sent to database | API request with messages array |
| Database connection | API client (Anthropic, OpenAI instance) |
| Query parameters (WHERE clause) | Prompt, system message, model params |
| Result rows returned | Response tokens generated |
| Stored procedure | System prompt (reusable behavior spec) |
| Streaming query result | Streaming response (tokens arrive progressively) |
| Cost per query | Cost per token (input + output) |

---

## Why It Exists (Problem → Solution)

**The problem:** Training LLMs costs millions of dollars. Most teams can't train their own. They need a way to use powerful models without owning the infrastructure.

**What came before:** You'd need massive GPU clusters, ML ops teams, and months of work just to run a model, let alone train one.

**What changed:** Cloud providers serve LLMs as APIs. You get access to frontier models like Claude Opus or GPT-4 for fractions of a cent per request. For a frontend developer, this means: write API call → ship AI feature. No ML expertise needed.

---

## Core Concepts

### 1. The Messages API (How Conversations Work)

**Plain English:** Every call to an LLM API sends an array of messages — the full conversation history. The model generates the next message. There is no session; each call is stateless.

**Analogy:** It's like a REST API with no cookies or sessions. Every request must include everything the server needs to process it. Imagine if every HTTP request required you to send your full browsing history — that's exactly how LLM APIs work.

```typescript
// You already know this pattern:
const response = await fetch('/api/users');   // stateless HTTP
const users = await response.json();

// LLM API is the same pattern:
const response = await anthropic.messages.create({
  model: 'claude-sonnet-4-6',
  max_tokens: 1024,
  messages: [
    { role: 'user', content: 'What is React?' },
    { role: 'assistant', content: 'React is a JavaScript library...' },
    { role: 'user', content: 'How does it compare to Vue?' },  // ← latest message
  ]
});
// The model sees the FULL history and generates the next assistant turn
```

```
Request structure:
┌──────────────────────────────────────────────────┐
│ model:       which model to use                   │
│ max_tokens:  maximum tokens to generate           │
│ system:      persistent rules/persona (optional)  │
│ messages:    [                                    │
│   { role: 'user',      content: '...' },         │
│   { role: 'assistant', content: '...' },         │
│   { role: 'user',      content: '...' },  ← new │
│ ]                                                 │
└──────────────────────────────────────────────────┘
```

**Common misconception:** The model "remembers" your conversation. It doesn't. You are responsible for sending the full conversation history every request. If you don't, it has no memory.

---

### 2. Streaming — Essential for Chat UIs

**Plain English:** Without streaming, you wait 5–10 seconds for the full response, then see it all at once. With streaming, tokens arrive in real-time as the model generates them — like watching someone type.

**Analogy:** Streaming is like watching a YouTube video as it loads vs. waiting for the entire video to download before pressing play. Same content, completely different user experience.

```
Without streaming:
  User clicks Send → ⏳ 5 seconds → entire response appears at once
  (feels broken, users click again thinking it didn't work)

With streaming:
  User clicks Send → first word appears in 0.3 seconds → words arrive continuously
  (feels alive, users read as it generates)
```

```typescript
// Without streaming (bad UX for chat):
const response = await anthropic.messages.create({
  model: 'claude-sonnet-4-6',
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'Explain React hooks.' }],
});
console.log(response.content[0].text);  // appears all at once after 5s

// With streaming (great UX):
const stream = anthropic.messages.stream({
  model: 'claude-sonnet-4-6',
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'Explain React hooks.' }],
});

stream.on('text', (text) => {
  process.stdout.write(text);  // each token appears immediately
});

const final = await stream.finalMessage();  // complete message when done
```

**Common misconception:** Streaming is only for aesthetics. Actually, streaming reduces perceived latency significantly (time-to-first-token is ~300ms even when full response takes 10s) and is necessary for long-running generations to avoid gateway timeouts.

---

### 3. Structured Outputs (Guaranteed JSON)

**Plain English:** Instead of asking the model to "return JSON" and hoping it does, you define an exact schema and the API guarantees the response matches it. Zero parse errors.

**Analogy:** The difference between asking a contractor to "build something roughly like the blueprint" vs. having a contract that legally requires exact spec compliance. Structured outputs are the legal contract.

```typescript
// Asking for JSON in the prompt (unreliable):
"Extract name and email. Return JSON."
// → Sometimes returns JSON, sometimes adds explanation text, sometimes malformed

// Structured output via tool_choice (guaranteed):
const response = await anthropic.messages.create({
  model: 'claude-sonnet-4-6',
  max_tokens: 512,
  tools: [{
    name: 'extract_contact',
    description: 'Extract contact info from text',
    input_schema: {
      type: 'object',
      properties: {
        name:      { type: 'string' },
        email:     { type: 'string' },
        sentiment: { type: 'string', enum: ['positive', 'negative', 'neutral'] },
      },
      required: ['name', 'email', 'sentiment'],
    }
  }],
  tool_choice: { type: 'tool', name: 'extract_contact' },  // force this tool
  messages: [{ role: 'user', content: 'John (john@acme.com) loves the product!' }]
});

const result = response.content[0].input;
// Guaranteed: { name: "John", email: "john@acme.com", sentiment: "positive" }
// Use it directly in your app — no parsing needed
```

**Common misconception:** JSON mode = structured outputs. JSON mode only guarantees syntactically valid JSON. Structured outputs (tool use / response_format with schema) guarantee your exact schema — field names, types, enums, required fields.

---

### 4. Prompt Caching — Save 90% on Repeated Context

**Plain English:** If you send the same large system prompt or document with every request, prompt caching avoids re-processing it. The API caches it and charges a fraction of the normal price for subsequent requests.

**Analogy:** It's exactly like CDN caching for HTTP. The first request is expensive (origin server). Subsequent requests are cheap (cache hit). Same data, same result, fraction of the cost.

```typescript
// Without caching: 200K tokens × 100 requests = 20M tokens = ~$60
// With caching: 200K tokens cached once + 100 × small query tokens = ~$6

const response = await anthropic.messages.create({
  model: 'claude-sonnet-4-6',
  max_tokens: 1024,
  system: [
    {
      type: 'text',
      text: hugeDocumentation,            // 50K tokens of context
      cache_control: { type: 'ephemeral' },  // ← mark for caching
    }
  ],
  messages: [{ role: 'user', content: 'How do I deploy?' }],
});
// First request: full price (cache miss)
// Next requests within 5 minutes: 90% cheaper (cache hit)
```

**Common misconception:** Caching is automatic. You have to explicitly mark content with `cache_control: { type: 'ephemeral' }`. The cache also has a 5-minute TTL — requests must be within 5 minutes of each other to benefit.

---

## How It Actually Works (Step-by-Step)

```
Your app                    Anthropic API              Model
    │                            │                       │
    │  1. Create API client       │                       │
    │  const a = new Anthropic()  │                       │
    │                            │                       │
    │  2. Build request           │                       │
    │  messages.create({...})    │                       │
    │                            │                       │
    │──── HTTP POST /v1/messages ►│                       │
    │     (with API key header)   │                       │
    │                            │──── Route to model ──►│
    │                            │                       │ 3. Tokenize prompt
    │                            │                       │ 4. Run attention
    │                            │                       │ 5. Generate tokens
    │                            │◄── Response tokens ───│    one by one
    │                            │
    │◄── HTTP 200 + JSON ─────────│
    │    response.content[0].text │
    │                            │
    │  6. Parse & display         │
```

---

## Code in Practice

### Minimal: Basic message

```typescript
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic();  // uses ANTHROPIC_API_KEY env var

const response = await anthropic.messages.create({
  model: 'claude-sonnet-4-6',
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'Explain React hooks in 3 sentences.' }]
});

console.log(response.content[0].text);
```

### Practical: Multi-turn conversation with system prompt

```typescript
const messages = [
  { role: 'user' as const, content: 'My name is Vitaly.' },
  { role: 'assistant' as const, content: 'Nice to meet you, Vitaly!' },
  { role: 'user' as const, content: 'What is my name?' },
];

const response = await anthropic.messages.create({
  model: 'claude-sonnet-4-6',
  max_tokens: 1024,
  system: 'You are a helpful coding assistant. Be concise.',
  messages,
});
// → "Your name is Vitaly!"
// The model "remembers" because the full history is in the messages array
```

### Production: Next.js streaming chat route

```typescript
// app/api/chat/route.ts
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic();

export async function POST(req: Request) {
  const { messages } = await req.json();

  const stream = await anthropic.messages.stream({
    model: 'claude-sonnet-4-6',
    max_tokens: 2048,
    system: 'You are a helpful coding assistant.',
    messages,
  });

  // Return as SSE stream to the frontend
  return new Response(stream.toReadableStream(), {
    headers: { 'Content-Type': 'text/event-stream' },
  });
}
```

```tsx
// app/page.tsx — consuming the stream
'use client';
import { useState } from 'react';

export default function Chat() {
  const [response, setResponse] = useState('');

  async function send(input: string) {
    setResponse('');
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [{ role: 'user', content: input }] }),
    });
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      setResponse(prev => prev + decoder.decode(value));  // live update
    }
  }

  return (/* your chat UI */);
}
```

---

## The Vercel AI SDK (Recommended for React/Next.js)

The Vercel AI SDK abstracts streaming, state management, and multi-provider support into a clean React hook.

```bash
npm install ai @ai-sdk/anthropic
```

```typescript
// app/api/chat/route.ts
import { anthropic } from '@ai-sdk/anthropic';
import { streamText } from 'ai';

export async function POST(req: Request) {
  const { messages } = await req.json();
  const result = streamText({
    model: anthropic('claude-sonnet-4-6'),
    system: 'You are a helpful coding assistant.',
    messages,
  });
  return result.toDataStreamResponse();
}
```

```tsx
// app/page.tsx — full streaming chat in 15 lines
'use client';
import { useChat } from 'ai/react';

export default function Chat() {
  const { messages, input, handleInputChange, handleSubmit } = useChat();

  return (
    <div>
      {messages.map(m => (
        <div key={m.id}><strong>{m.role}:</strong> {m.content}</div>
      ))}
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} placeholder="Ask..." />
        <button type="submit">Send</button>
      </form>
    </div>
  );
}
// useChat handles: message state, streaming, loading states, error handling, history
```

---

## API Comparison: Anthropic vs OpenAI

```
                        Anthropic              OpenAI
────────────────────────────────────────────────────────────
Endpoint               /v1/messages           /v1/chat/completions
Auth header            x-api-key              Authorization: Bearer
System prompt          Separate system field  { role: 'system' } message
Response text          content[0].text        choices[0].message.content
Streaming              .stream() method       stream: true option
Structured output      Tool use + tool_choice response_format: json_schema
Context window         200K (Claude)          128K (GPT-4o)
Best models            Claude Opus/Sonnet     GPT-4o, o3
```

```typescript
// OpenAI SDK (for comparison)
import OpenAI from 'openai';
const openai = new OpenAI();

const response = await openai.chat.completions.create({
  model: 'gpt-4o',
  messages: [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user', content: 'Hello!' },
  ],
});
console.log(response.choices[0].message.content);
```

---

## Error Handling & Best Practices

```typescript
async function safeLLMCall(prompt: string): Promise<string | null> {
  try {
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 1024,
      messages: [{ role: 'user', content: prompt }],
    });
    return response.content[0].text;

  } catch (error) {
    if (error instanceof Anthropic.RateLimitError) {
      // 429 — SDK has automatic retries, but you can add custom backoff
      console.warn('Rate limited');
      return null;
    } else if (error instanceof Anthropic.APIError) {
      console.error(`API Error ${error.status}: ${error.message}`);
      return null;
    }
    throw error;  // unexpected errors bubble up
  }
}
```

### Cost Control

```typescript
// 1. Use cheapest model that works for the task
// Classification/extraction: Haiku (60× cheaper than Opus)
// Code gen / complex reasoning: Sonnet or Opus

// 2. Set max_tokens appropriately — don't set 4096 for a one-word answer
max_tokens: 10   // For classification
max_tokens: 200  // For short summaries
max_tokens: 2048 // For full code generation

// 3. Cache repeated context
cache_control: { type: 'ephemeral' }  // on static system prompts / docs

// 4. Batch async jobs (50% cheaper via batch API)
const batch = await anthropic.messages.batches.create({
  requests: prompts.map(p => ({
    custom_id: p.id,
    params: { model: 'claude-haiku-4-5', max_tokens: 100, messages: [{ role: 'user', content: p.text }] }
  }))
});
```

---

## Architecture for a Production AI App

```
┌──────────────────────────────────────────────────────────┐
│                  Frontend (React/Next.js)                  │
│  ┌─────────────────┐  ┌───────────────┐  ┌────────────┐  │
│  │  Chat UI         │  │  Streaming    │  │  Auth /    │  │
│  │  (useChat hook)  │  │  Rendering    │  │  Session   │  │
│  └────────┬────────┘  └───────────────┘  └────────────┘  │
│           │                                               │
├───────────┼───────────────────────────────────────────────┤
│           │           API Layer (Next.js / Express)        │
│           ▼                                               │
│  ┌────────────────┐  ┌───────────────┐  ┌────────────┐   │
│  │  /api/chat     │  │  Rate Limit   │  │  Usage     │   │
│  │  (streaming)   │  │  Middleware   │  │  Tracking  │   │
│  └────────┬───────┘  └───────────────┘  └────────────┘   │
│           │                                               │
├───────────┼───────────────────────────────────────────────┤
│           │           AI Layer                             │
│           ▼                                               │
│  ┌────────────────┐  ┌───────────────┐  ┌────────────┐   │
│  │  Anthropic SDK │  │  RAG /        │  │  Prompt    │   │
│  │  (streaming)   │  │  Vector DB    │  │  Templates │   │
│  └────────────────┘  └───────────────┘  └────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## Gotchas & Pitfalls

```
❌ Storing API keys in frontend code → ✅ Always call LLM APIs from server-side routes
   Browser code is public — your API key will be stolen and billed

❌ Not setting max_tokens → ✅ Always set max_tokens explicitly
   Default behavior varies; accidentally generating huge responses spikes costs

❌ Building sessions/memory on the server → ✅ Understand the stateless model
   LLMs have no memory. Conversation history lives in your app's state/DB.

❌ Waiting for full response in chat UI → ✅ Always stream chat responses
   Users think the app is broken when they wait 5+ seconds for a response

❌ Using Opus for everything → ✅ Match model to task
   Haiku is 60× cheaper and perfectly capable of classification/extraction

❌ No error handling → ✅ Always handle rate limits and API errors
   LLM APIs occasionally return 429s, 500s, or time out on long generations

❌ Asking for JSON without guaranteeing it → ✅ Use tool_choice or response_format
   "Return JSON" in a prompt is unreliable; structured outputs are guaranteed
```

---

## When to Use / When NOT to Use LLM APIs

**Use when:**
- Building chat, Q&A, content generation, or summarization features
- Automating tasks that need language understanding (classification, extraction)
- You need the latest frontier models without managing infrastructure
- Prototyping AI features — APIs get you from zero to demo in hours

**Don't use when:**
- Simple pattern matching / regex would suffice (far cheaper)
- Real-time, sub-100ms latency is required (LLM generation takes 0.5–10s)
- Your data is too sensitive to send to a third-party API (use local models)
- Task is purely numerical/tabular (use traditional ML models)

---

## Production Notes

### Cost estimation (per feature)

The only formula you need:

```
cost/request = (input_tokens × in_price + output_tokens × out_price) / 1_000_000
```

Then: `monthly_cost = cost/request × requests/month`.

**Worked example** — chatbot on Sonnet, 2K input (incl. system prompt + history), 400 output, 50K requests/day:
- `(2000 × $3 + 400 × $15) / 1M = $0.012/request`
- `$0.012 × 50K × 30 = $18,000/month`
- With prompt caching (90% cache hit on 1.8K of the input): drops to ~$5,000/month.

**Order-of-magnitude shortcuts:**
- Haiku/Flash ≈ $1 per 1M "typical" chatbot turns.
- Sonnet ≈ $10 per 1M turns.
- Opus ≈ $50 per 1M turns.

### Latency SLAs to design for

| Metric | Small tier | Mid tier | Flagship |
|--------|-----------|----------|----------|
| TTFT p50 | 200–400 ms | 400–700 ms | 700 ms–1.5 s |
| TTFT p95 | 800 ms | 1.5 s | 3 s |
| Tokens/sec | 80–150 | 40–80 | 20–40 |

**Rule:** never block a user-facing response on a non-streaming LLM call. Use streaming on every chat endpoint; batch/async for background jobs.

### Failure modes

- **429 rate-limited** — respect `Retry-After`, apply exponential backoff + jitter, shed load. Don't retry in tight loops; you'll deepen the outage.
- **5xx / capacity events** — providers have regional capacity blips. Keep a cross-provider fallback (Anthropic → OpenAI → local).
- **Timeouts mid-stream** — the stream can die after N tokens. Detect and resume or restart; don't assume a clean end.
- **Schema violation on tool/JSON mode** — providers mostly enforce it, but network truncation still happens. Validate every response; retry once on parse failure.
- **Content-policy refusals** — treat as a user-visible error, not a crash; log the prompt for safety review.
- **Token-limit overflow** — input + max_output > context window throws at request time. Pre-estimate with the tokenizer before sending; trim history first.

### What to monitor

- **Cost per request and per feature** (tag every call with `feature_id`).
- **TTFT p50/p95** per endpoint, broken down by model.
- **Error rate by status code** (429 vs 5xx vs timeout vs validation).
- **Retry count** — a rising p95 retry count = the upstream is degrading before it 5xx's.
- **Cache hit rate** on cached system prompts (see [production-llm-patterns.md](production-llm-patterns.md)).
- **Token-usage drift** — a sudden spike in output tokens usually = prompt regression.

See [../ml-ops/reliability-patterns.md](../ml-ops/reliability-patterns.md) for retry/fallback code and [../ml-ops/llm-observability.md](../ml-ops/llm-observability.md) for tracing.

---

## Related Concepts (The Map)

| If you know... | LLM API concept is like... |
|----------------|---------------------------|
| REST API calls | LLM API is just a POST request with a messages array |
| WebSockets / SSE | Streaming responses use the same Server-Sent Events pattern |
| HTTP sessions / cookies | LLM APIs are sessionless — you manage conversation state |
| React Query | Vercel AI SDK's `useChat` is like React Query for LLM conversations |
| CDN caching | Prompt caching is the same concept — cache expensive processing |

**Connected topics:**
- **Prompt Engineering** → what goes inside those messages
- **RAG** → how to inject retrieved context into API calls
- **Agents & Tool Use** → tool_use response type powers agent loops
- **Fine-tuning** → when API models aren't specialized enough

---

## Cheat Sheet

| Concept | What to Remember |
|---------|-----------------|
| Messages API | Stateless — send full conversation every request |
| Tokens | Cost: input < output. Use Haiku for cheap tasks, Opus for complex |
| Streaming | Essential for chat UIs — always stream |
| Structured outputs | Use `tool_choice` (Anthropic) or `response_format` (OpenAI) for guaranteed JSON |
| Prompt caching | Add `cache_control: ephemeral` to static system prompts — 90% savings |
| Vercel AI SDK | Best DX for React/Next.js — useChat handles everything |
| Model selection | Haiku=cheap/fast, Sonnet=balanced, Opus=powerful |
| Error handling | Always handle RateLimitError and APIError |

**Model cost tiers (Anthropic, approximate):**
```
Haiku   $0.25/$1.25 per M tokens    ← Simple tasks, high volume
Sonnet  $3/$15 per M tokens         ← Most production use cases
Opus    $15/$75 per M tokens        ← Complex reasoning, maximum quality
```

**Remember these 3 things:**
1. API keys go server-side only — never in frontend code
2. Always stream chat responses for good UX
3. Match model tier to task complexity — Haiku is 60× cheaper than Opus

---

## Self-Check Questions

1. **Why do you have to send the full conversation history with every API call?**

<details>
<summary>Answer</summary>
LLM APIs are stateless HTTP endpoints — there are no sessions on the server side. Each request is independent. The "memory" of a conversation exists only in the messages array you send. This is a design choice: it makes the API horizontally scalable (any server can handle any request) but shifts state management responsibility to your application.
</details>

2. **A user sees a 6-second blank screen before their chat response appears. What's wrong and how do you fix it?**

<details>
<summary>Answer</summary>
You're not streaming. The response is being generated on the server and only sent when complete. Fix: use `anthropic.messages.stream()` (or `streamText` in Vercel AI SDK) and return a `ReadableStream` to the frontend. The user will see the first tokens in ~300ms instead of waiting 6 seconds.
</details>

3. **You're paying $100/day on LLM API calls. Your app sends a 10K-token system prompt with every request. What's the cheapest fix?**

<details>
<summary>Answer</summary>
Enable prompt caching: add `cache_control: { type: 'ephemeral' }` to the system prompt. This costs ~90% less for cache hits. Cache TTL is 5 minutes, so requests within that window hit the cache. If your users make multiple requests in a session, this alone could reduce costs by 80-90%.
</details>

4. **You need the LLM to always return a JSON object with specific fields. What's the safest approach?**

<details>
<summary>Answer</summary>
Use tool_choice with a defined input_schema (Anthropic) or response_format with json_schema (OpenAI). This guarantees the response matches your exact schema — right field names, right types, required fields present. Never rely on asking for JSON in the prompt text alone; it will fail on edge cases.
</details>

5. **When should you use Claude Haiku vs Sonnet vs Opus?**

<details>
<summary>Answer</summary>
Haiku: simple, well-defined tasks — classification, sentiment analysis, entity extraction, short Q&A. Fast and 60× cheaper than Opus. Sonnet: most production use cases — code generation, summarization, RAG, agents. Great balance of capability and cost. Opus: complex reasoning, nuanced writing, difficult coding problems where quality matters more than cost.
</details>

---

## Go Deeper

1. **[Anthropic API Reference](https://docs.anthropic.com/en/api/getting-started)** — The authoritative reference. Bookmark the Messages endpoint page. When you hit unexpected behavior, the answer is usually here. (reference, ongoing)

2. **[Vercel AI SDK Documentation](https://sdk.vercel.ai/docs)** — Best learning resource for building React/Next.js AI apps. The "Examples" section has complete working apps you can clone and study. (2 hours)

3. **[Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook)** — Practical examples from Anthropic's team: tool use, RAG, prompt caching, vision, agents. Copy-paste starting points for every feature. (ongoing reference)

4. **[LLM Pricing Calculator](https://llmpricecheck.com/)** — Compare token costs across providers. Essential for making model selection decisions based on actual cost. (5 min, bookmark it)

5. **[OpenAI Cookbook](https://github.com/openai/openai-cookbook)** — Despite being OpenAI-specific, many patterns (streaming, structured outputs, embeddings) translate directly to Anthropic. Rich set of practical notebooks. (ongoing reference)

---

**What's next?** You can call LLMs from your app. Now learn how to let them take real actions — search the web, query databases, call other APIs: [Agents & Tool Use →](agents-tool-use.md)
