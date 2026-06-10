# Agents & Tool Use

## TL;DR

An AI agent is an LLM that can take actions in the real world — not just generate text, but call functions, search the web, query databases, and execute multi-step plans. Tool use (function calling) is the mechanism: you define functions the LLM can call, it decides which to call and when, you execute them, and pass results back. The agent loops — think → act → observe → think — until the task is complete. This turns a passive text generator into an autonomous assistant.

> 💡 **Key Insight:** You never let the LLM *execute* tools directly. The LLM *decides* what to call; your code executes it. You stay in control of what can happen.

---

## The Mental Model

**Think of an agent like a highly capable intern with a phone book.**

The intern is smart but stuck at a desk. You give them a list of phone numbers (tools) they can call to get things done. They decide who to call, when to call, and in what order — but *you* control which numbers are in the phone book. The intern orchestrates; the real work happens through calls.

| Real world | Technical concept |
|------------|------------------|
| Intern's brain / reasoning | LLM |
| Phone book of contacts | List of available tools (tool definitions) |
| Intern decides who to call | LLM chooses which tool to invoke |
| Intern dials the number | Your code executes the tool |
| Contact provides information | Tool returns result |
| Intern reports back to you | LLM generates final response with results |
| Task takes multiple calls | Agent loop (multi-step reasoning) |

---

## Why It Exists (Problem → Solution)

**The problem:** LLMs are brilliant reasoners trapped in a bubble. They can't access real-time data, call APIs, query your database, or take any action in the world. They can only generate text based on their training data.

**What came before:** Humans had to manually break tasks into steps, run each step, and paste results into the next prompt. Tedious and doesn't scale.

**What changed:** Function calling (2023) gave LLMs a structured way to request actions. The model signals "I want to call this function with these arguments" and your code handles the actual execution. This unlocks a new category of AI applications — agents that autonomously complete multi-step tasks.

```
Without tool use:
  User: "What's the weather in Paris?"
  LLM:  "I don't have real-time weather data." ← useless

With tool use:
  User: "What's the weather in Paris?"
  LLM:  → calls get_weather("Paris")
        → receives { temp: 18, condition: "sunny" }
  LLM:  "It's 18°C and sunny in Paris right now!" ← useful
```

---

## Core Concepts

### 1. Tool Definitions — The Agent's Capabilities

**Plain English:** A tool definition tells the LLM what a function does, when to use it, and what arguments it takes. The LLM reads these descriptions to decide which tool to call.

**Analogy:** Tool descriptions are like API documentation. A developer reads docs to know when and how to call a function. The LLM does the same thing — if the description is vague, it'll call the wrong tool or with wrong arguments.

```typescript
const tools = [
  {
    name: 'get_weather',
    description: 'Get current weather for a city. Use when users ask about weather conditions, temperature, or forecast.',
    input_schema: {
      type: 'object',
      properties: {
        city: {
          type: 'string',
          description: 'City name and country, e.g. "Paris, France"'
        },
        units: {
          type: 'string',
          enum: ['celsius', 'fahrenheit'],
          description: 'Temperature units'
        }
      },
      required: ['city']
    }
  }
];
```

**Common misconception:** The name is the most important part. Actually, the **description** is what the LLM reads to decide whether to call a tool. A perfectly named tool with a bad description will be misused.

---

### 2. The 4-Step Tool Use Flow

**Plain English:** The LLM never executes tools itself. It *requests* a tool call; you *execute* it; you *return* the result; the LLM *responds* with a final answer using that result.

**Analogy:** It's like submitting a support ticket. The LLM writes the ticket ("please look up weather for Paris"), your system processes it (calls the weather API), and sends back the result. The LLM then writes the customer-facing response based on what was returned.

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ 1. You   │     │ 2. LLM   │     │ 3. You   │     │ 4. LLM   │
│ send msg │────►│ requests │────►│ execute  │────►│ generates│
│ + tools  │     │ a tool   │     │ the tool │     │ final    │
│          │     │ call     │     │ & return │     │ response │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
```

```typescript
import Anthropic from '@anthropic-ai/sdk';
const anthropic = new Anthropic();

// 1. Send message with tools
const response = await anthropic.messages.create({
  model: 'claude-sonnet-4-6',
  max_tokens: 1024,
  tools,
  messages: [{ role: 'user', content: "What's the weather in Paris?" }]
});

// 2. Check if LLM wants to call a tool
if (response.stop_reason === 'tool_use') {
  const toolCall = response.content.find(c => c.type === 'tool_use');
  // { type: 'tool_use', id: 'tu_123', name: 'get_weather', input: { city: 'Paris' } }

  // 3. Execute the tool (your code, your control)
  const weatherData = await fetchWeatherAPI(toolCall.input.city);

  // 4. Send result back to LLM
  const final = await anthropic.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 1024,
    tools,
    messages: [
      { role: 'user', content: "What's the weather in Paris?" },
      { role: 'assistant', content: response.content },  // LLM's tool request
      {
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: toolCall.id,
          content: JSON.stringify(weatherData)
        }]
      }
    ]
  });

  console.log(final.content[0].text);
  // "It's currently 18°C and sunny in Paris!"
}
```

**Common misconception:** Tool use is complex to implement. The basic flow is just 2 API calls. The complexity comes from the agent loop (multi-step), not single tool use.

---

### 3. The Agent Loop — Multi-Step Autonomous Reasoning

**Plain English:** Real tasks need multiple tool calls. The agent loops: call → get result → think → call again → ... until done. You just keep sending results back until `stop_reason === 'end_turn'`.

**Analogy:** Think of it as a reducer / state machine:
```
state = { messages: [], status: 'thinking' }

LLM_RESPONSE (tool_use) → status: 'executing'
TOOL_COMPLETE            → status: 'thinking'   ← back to LLM
LLM_RESPONSE (end_turn)  → status: 'done'        ← final answer
```

```typescript
async function agentLoop(userMessage: string): Promise<string> {
  const messages = [{ role: 'user' as const, content: userMessage }];

  while (true) {
    const response = await anthropic.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 4096,
      system: 'You are a helpful assistant. Use tools when needed.',
      tools,
      messages,
    });

    messages.push({ role: 'assistant', content: response.content });

    // If LLM is done (no more tool calls), return the answer
    if (response.stop_reason === 'end_turn') {
      return response.content.find(c => c.type === 'text')?.text ?? '';
    }

    // Execute all tool calls (might be multiple in parallel)
    const toolResults = await Promise.all(
      response.content
        .filter(c => c.type === 'tool_use')
        .map(async tc => ({
          type: 'tool_result' as const,
          tool_use_id: tc.id,
          content: JSON.stringify(await executeTool(tc.name, tc.input)),
        }))
    );

    messages.push({ role: 'user', content: toolResults });
    // Loop back to LLM with results
  }
}
```

**Common misconception:** The agent decides when it's done. The agent loops until `stop_reason === 'end_turn'`, but you should always set a max iteration limit. Without it, a confused agent can loop forever and drain your budget.

---

### 4. Parallel vs Sequential Tool Calls

**Plain English:** When tools don't depend on each other, the LLM can request them all in a single response — execute in parallel. When they do depend on each other, it has to do one, see the result, then decide the next.

**Analogy:** A chef prepping multiple dishes. Independent steps (chop onions, boil water) happen at once. Dependent steps (taste the sauce → decide if it needs salt) have to be sequential — you can't season before tasting.

**The two patterns:**

```
PARALLEL (independent):                  SEQUENTIAL (dependent):
─────────────────────────────            ──────────────────────────────
User: "Weather in Paris AND Tokyo"       User: "Email me that PR's author"
                                          
LLM response (one turn):                 Turn 1:
  tool_use: get_weather("Paris")           LLM: get_pr(42)
  tool_use: get_weather("Tokyo")         ← 2 blocks   Tool: { author: "alice@x.com", ... }
                                          
You: Promise.all([both])                  Turn 2:
You send BOTH results back at once.        LLM: send_email("alice@x.com", ...)
                                          (the second call needs the first's result)
1 round-trip to the model.               2 round-trips — unavoidable.
```

**How the LLM decides:** It's emergent from training, but you can influence it:

- **System prompt hint:** `"When a task involves independent lookups, request them in parallel."`
- **Tool descriptions that suggest independence:** `"get_weather: Safe to call for multiple cities in one turn."`
- **Task phrasing:** "Compare X and Y" triggers parallel calls more reliably than "Tell me about X. Then Y." — the word "and" nudges the model toward batching.

```typescript
// Handle both patterns with the same loop
const toolCalls = response.content.filter(c => c.type === 'tool_use');

// Promise.all handles 1 or N calls identically — parallel if multiple, fine if single
const results = await Promise.all(
  toolCalls.map(async tc => {
    try {
      const output = await executeTool(tc.name, tc.input);
      return { type: 'tool_result' as const, tool_use_id: tc.id, content: JSON.stringify(output) };
    } catch (err) {
      // Return structured error — the LLM can often recover (try different args, give up gracefully)
      return {
        type: 'tool_result' as const,
        tool_use_id: tc.id,
        content: JSON.stringify({ error: String(err) }),
        is_error: true,
      };
    }
  })
);
```

**Common misconception:** "Parallel means faster, always use it." ✅ Only when calls are independent. If call B needs call A's output, forcing parallelism produces hallucinated arguments. Trust the model's choice — it's usually right — but log the ratio of parallel:sequential turns so you can spot regressions.

---

> 🧠 **Quick recall — answer out loud before scrolling on** (all answers are above):
> 1. Who decides which tool to call, and who actually executes it?
> 2. The agent loop in one sentence — and what signals "done"?
> 3. Which field of a tool definition does the LLM actually read to choose tools?
> 4. Parallel vs sequential tool calls — what determines which the model should do?
> 5. The two non-negotiable safety rails for any agent loop?

---

## How the Agent Loop Works (Step-by-Step)

```
Task: "Compare weather in Paris and Tokyo, then find flights between them"

Step 1: User message arrives
        ↓
Step 2: LLM thinks: "I need weather for both cities — I can call these simultaneously"
        → Returns 2 tool_use blocks in one response
        ↓
Step 3: Your code executes both in parallel:
        get_weather("Paris") → 18°C, sunny
        get_weather("Tokyo") → 22°C, cloudy
        ↓
Step 4: Results sent back to LLM
        ↓
Step 5: LLM thinks: "Now I need flights"
        → Returns 1 tool_use block
        ↓
Step 6: Your code executes:
        search_flights("Paris", "Tokyo") → [Air France $650, ANA $720, Turkish $480]
        ↓
Step 7: Results sent back to LLM
        ↓
Step 8: LLM has everything it needs
        → stop_reason: 'end_turn'
        → Generates formatted answer
        ↓
Step 9: Return final response to user
```

```
Message history grows:
[user: question]
[assistant: tool_use × 2]
[user: tool_result × 2]
[assistant: tool_use × 1]
[user: tool_result × 1]
[assistant: end_turn — final answer] ← return this
```

---

## Code in Practice

### Designing Good Tools

The LLM chooses tools based on descriptions. Invest in descriptions.

```typescript
// ❌ Bad tool definition — vague, LLM won't know when to use it
{
  name: 'db_query',
  description: 'Query the database',
  input_schema: { type: 'object', properties: { q: { type: 'string' } } }
}

// ✅ Good tool definition — specific, helpful, tells the LLM exactly when to use it
{
  name: 'search_products',
  description: `Search the product catalog by name, category, or price range.
               Use when the user asks about products, pricing, or availability.
               Returns up to 10 matching products with name, price, and stock status.
               Do NOT use for order history or customer account information.`,
  input_schema: {
    type: 'object',
    properties: {
      query:     { type: 'string', description: 'Search term or product name' },
      category:  { type: 'string', enum: ['electronics', 'clothing', 'books'] },
      max_price: { type: 'number', description: 'Maximum price in USD' },
    },
    required: ['query']
  }
}
```

### Real-World Agent: Customer Support

```typescript
const supportTools = [
  {
    name: 'lookup_order',
    description: 'Look up order details by order ID. Use when customer provides an order number.',
    input_schema: {
      type: 'object',
      properties: { order_id: { type: 'string', description: 'Order ID, e.g. "ORD-12345"' } },
      required: ['order_id']
    }
  },
  {
    name: 'search_knowledge_base',
    description: 'Search help articles for product info, policies, and troubleshooting guides.',
    input_schema: {
      type: 'object',
      properties: { query: { type: 'string' } },
      required: ['query']
    }
  },
  {
    name: 'create_support_ticket',
    description: 'Create a ticket to escalate to a human agent. Use when you cannot resolve the issue.',
    input_schema: {
      type: 'object',
      properties: {
        summary:  { type: 'string', description: 'Brief issue summary' },
        priority: { type: 'string', enum: ['low', 'medium', 'high'] },
      },
      required: ['summary', 'priority']
    }
  }
];

const SUPPORT_SYSTEM = `You are a customer support agent for TechCorp.
Use available tools to resolve customer issues. 
If you cannot resolve within 3 tool calls, create a support ticket.
Always be empathetic and clear.`;
```

---

## Agent Safety & Guardrails

Agents can take real actions. Safety is not optional.

```typescript
// 1. Categorize tools by risk
const readOnly = ['search_products', 'get_weather', 'lookup_order'];
const write    = ['send_email', 'create_order', 'update_profile'];
const danger   = ['delete_account', 'execute_sql', 'deploy_code'];

// 2. Require confirmation for write operations
async function executeTool(name: string, input: any) {
  if (write.includes(name) || danger.includes(name)) {
    const ok = await askUser(`AI wants to: ${name}(${JSON.stringify(input)}). Allow?`);
    if (!ok) return { error: 'User denied this action' };
  }
  return await toolImplementations[name](input);
}

// 3. Validate LLM-generated inputs — never trust them blindly
async function runSql(query: string) {
  if (!query.trim().toUpperCase().startsWith('SELECT')) {
    return { error: 'Only SELECT queries allowed' };  // No writes ever
  }
  return await db.query(query);
}

// 4. Cap the agent loop
const MAX_STEPS = 10;
let steps = 0;
while (steps < MAX_STEPS) {
  steps++;
  // ... agent loop ...
}
if (steps >= MAX_STEPS) return 'Task exceeded maximum steps — please try a simpler request.';
```

```
Safety hierarchy:
────────────────
Read operations        → Auto-approve (search, lookup, fetch)
Write operations       → Require user confirmation
Destructive operations → Require explicit confirmation + audit log
Never allow            → Arbitrary code execution outside sandbox
```

---

## Agent Frameworks

Build agents from scratch (above) or use frameworks.

```python
# LangChain — most popular, most ecosystem
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent

llm = ChatAnthropic(model="claude-sonnet-4-6")
agent = create_tool_calling_agent(llm, tools, prompt)
result = agent.invoke({"input": "What's the weather in Paris?"})
```

```typescript
// Vercel AI SDK — best for Next.js, integrates with streaming UI
import { anthropic } from '@ai-sdk/anthropic';
import { generateText, tool } from 'ai';
import { z } from 'zod';

const result = await generateText({
  model: anthropic('claude-sonnet-4-6'),
  tools: {
    weather: tool({
      description: 'Get weather for a city',
      parameters: z.object({ city: z.string() }),
      execute: async ({ city }) => fetchWeather(city),
    }),
  },
  maxSteps: 5,  // caps the agent loop
  prompt: "What's the weather in Paris?",
});
```

**When to use frameworks vs. from scratch:**
- **From scratch**: you understand the loop, small apps, full control
- **LangChain**: large ecosystem, pre-built tools, complex multi-agent graphs
- **Vercel AI SDK**: Next.js apps, streaming UI, simpler tool integration

---

## Gotchas & Pitfalls

```
❌ Vague tool descriptions → ✅ Specific descriptions that say when AND when NOT to use
   "Query database" vs "Search product catalog by name/category. Don't use for orders."

❌ No max iteration limit → ✅ Always set MAX_STEPS
   A confused agent will loop until you've spent all your budget

❌ Trusting LLM-generated inputs → ✅ Validate all tool inputs before execution
   The LLM can hallucinate tool arguments — always sanitize SQL, file paths, IDs

❌ Allowing writes without confirmation → ✅ Require human approval for state changes
   "Create order for all items in cart" with wrong cart = real money charged

❌ All tools in one definition → ✅ One tool = one responsibility
   Combining search + purchase into one tool means the LLM calls it wrong

❌ Ignoring error returns → ✅ Return structured errors the LLM can act on
   Return { error: 'Product not found' } so the LLM can try a different search

❌ Sequential tool execution → ✅ Parallel execution for independent tools
   Don't await get_weather("Paris"); await get_weather("Tokyo") — use Promise.all
```

---

## When to Use / When NOT to Use Agents

**Use agents when:**
- Task requires multiple steps that depend on intermediate results
- You need real-time data the LLM doesn't have (weather, stock prices, DB records)
- The steps required aren't fully known upfront (the LLM needs to decide)
- Automating workflows that currently require a human to orchestrate

**Don't use agents when:**
- A single LLM call with the right prompt solves it (simpler = better)
- You need sub-second response times (agent loops take seconds)
- The task is deterministic and predictable (use a regular pipeline)
- Safety requirements make autonomous action unacceptable (always keep humans in the loop)

---

## Production Notes

### Cost — agents multiply token spend

An agent turn is *N* model calls plus the tool output tokens from each. Rule of thumb:

```
cost/user_request ≈ (avg_steps × avg_input_tokens × in_price +
                     avg_steps × avg_output_tokens × out_price +
                     total_tool_output_tokens × in_price) / 1M
```

**Worked example** — customer support agent, Sonnet, 3 tool calls avg, 4K input/800 output per step, tools return 500 tokens each:
- Per request: `3 × (4000 × $3 + 800 × $15 + 500 × $3) / 1M ≈ $0.077`
- 10K requests/day → **~$23K/month**. A single-shot Sonnet call for the same task would be ~$3K/month. Agents are 5–10× more expensive than one-shot calls.

**Cost levers, biggest first:**
1. Cap max steps (most loops don't need >5).
2. Route: use Haiku for tool-call decisions, Sonnet/Opus only for the final synthesis.
3. Cache the stable system prompt (tool definitions + role) across steps.
4. Trim tool outputs before feeding back (summarize long API responses).

### Latency (p50 / p95)

| Config | p50 | p95 |
|--------|-----|-----|
| 1 tool call + final answer | 3–6 s | 10–15 s |
| 3–5 tool calls | 8–20 s | 30–60 s |
| Long-running research (10+ steps) | 30 s–2 min | 2–5 min |

Stream intermediate tool calls to the UI (`"Searching docs..."`, `"Found 3 results, checking..."`) — silence kills UX.

### Failure modes

- **Infinite loop** — agent repeats the same tool call with the same args. Mitigation: max-steps cap (hard), dedupe identical consecutive calls (soft).
- **Hallucinated tool args** — model invents a field the schema doesn't have. Mitigation: strict JSON schema validation; on failure, return the validation error *to the model* so it retries correctly.
- **Tool failure cascades** — external API returns 500; agent panics or loops. Mitigation: pass the error text back to the model; it can often recover or decide to stop.
- **Prompt injection via tool output** — retrieved content contains "ignore previous instructions." Mitigation: wrap tool outputs in clear delimiters, and never let tool output change the system prompt.
- **Destructive action mis-fires** — agent deletes the wrong thing. Mitigation: dry-run mode, explicit user confirmation for writes, allow-list of safe tools.
- **Cost runaway** — one buggy prompt puts 10K agents into long loops. Mitigation: per-request budget cap (max $0.50) that hard-stops the loop.

### What to monitor

- **Avg steps per request** and **p95 steps** — sudden rise = loop bug.
- **Tool-call success rate** per tool (404s, 5xx, schema fails).
- **Cost per agent run** (p50 and p95) with an alert on p99 spikes.
- **Human-override rate** — fraction of agent outputs the user rejects/edits.
- **Task-completion rate** on a golden set, via LLM-as-judge ([evals.md](evals.md)).
- **Schema-validation failure rate** per tool.

See [mcp.md](mcp.md) for the standard tool protocol, [../ml-ops/safety-guardrails.md](../ml-ops/safety-guardrails.md) for injection defense, and [../ml-ops/reliability-patterns.md](../ml-ops/reliability-patterns.md) for retries and budget caps.

---

## Related Concepts (The Map)

| If you know... | Agent concept is like... |
|----------------|-------------------------|
| React component with useEffect | Agent = component with side effects that fetch external data |
| Redux reducer | Agent loop = reducer: state + action → new state, until done |
| Middleware pipeline | Tool chain = middleware that adds data to the request |
| AWS Lambda orchestration | Agent = orchestrator that calls Lambda functions in sequence |
| State machines (XState) | Agent loop is a state machine: thinking → acting → observing → thinking |

**Connected topics:**
- **Prompt Engineering** → system prompt and ReAct pattern for agent behavior
- **LLM APIs & SDKs** → tool_use response type and how to handle it
- **RAG** → common tool for agents to search knowledge bases
- **Fine-tuning** → teach models to better follow tool-use conventions

---

## Cheat Sheet

| Concept | What to Remember |
|---------|-----------------|
| Tool use | LLM signals a call; your code executes it — you stay in control |
| Agent loop | Think → Act → Observe → repeat until `stop_reason === 'end_turn'` |
| Tool description | The most important field — LLM decides based on this |
| Parallel calls | Independent tools run concurrently — use Promise.all |
| MAX_STEPS | Always cap the loop — prevent runaway agents |
| Safety | Read=auto, Write=confirm, Delete=explicit confirm + log |
| ReAct pattern | Reason then act — explicit Thought/Action/Observation structure |

**The minimal agent loop (production-shaped):**
```typescript
while (steps++ < MAX_STEPS) {
  const res = await llm.create({ messages, tools });
  messages.push({ role: 'assistant', content: res.content });

  // Handle every stop_reason — not just end_turn
  if (res.stop_reason === 'end_turn')   return getText(res);
  if (res.stop_reason === 'max_tokens') throw new Error('Hit max_tokens mid-thought — raise limit or shorten task');
  if (res.stop_reason === 'stop_sequence') return getText(res);
  if (res.stop_reason !== 'tool_use')   throw new Error(`Unexpected stop: ${res.stop_reason}`);

  // Execute tools — wrap errors so the LLM can recover instead of the loop crashing
  const results = await Promise.all(
    getToolCalls(res).map(async tc => {
      try   { return toolResult(tc.id, await executeWithValidation(tc)); }
      catch (e) { return toolResult(tc.id, { error: String(e) }, { is_error: true }); }
    })
  );
  messages.push({ role: 'user', content: results });
}
throw new Error(`Agent exceeded ${MAX_STEPS} steps`);
```

**Tool design checklist:**
1. CLEAR NAME — `search_products`, not `sp` or `tool_3`
2. DETAILED DESC — when to use, what it returns, edge cases
3. TYPED PARAMS — use enums, required fields, descriptions
4. SINGLE PURPOSE — one tool = one action
5. SAFE DEFAULTS — reads first, writes need confirmation
6. USEFUL ERRORS — return messages the LLM can act on

**Remember these 3 things:**
1. The LLM decides what to call; your code decides what can run
2. Always cap agent loops — no MAX_STEPS = infinite loop risk
3. Tool descriptions matter more than tool names — invest in them

---

## Self-Check Questions

1. **An agent calls `delete_user(user_id: "123")` autonomously. Who is responsible for this action being safe?**

<details>
<summary>Answer</summary>
You are — the engineer who built the agent. The LLM can only call tools you've given it. If `delete_user` is in your tool list without a confirmation gate, you've allowed it. The fix: categorize destructive tools separately, require explicit user confirmation before executing, and log all write operations. Never expose destructive tools without guards.
</details>

2. **What happens if you don't include a `MAX_STEPS` limit in your agent loop?**

<details>
<summary>Answer</summary>
The agent can loop indefinitely if it gets confused, encounters a tool error it keeps retrying, or is given an impossible task. This drains your API budget (each loop is billable API calls) and leaves the user waiting forever. Always set a sensible limit (10–20 steps for most tasks) and return a graceful failure message when exceeded.
</details>

3. **The LLM keeps calling the wrong tool. What's the first thing you fix?**

<details>
<summary>Answer</summary>
The tool description. The LLM selects tools by reading descriptions — if the description is vague, ambiguous, or doesn't say when NOT to use it, the model will guess wrong. Rewrite the description to be specific about the exact use case, what the tool returns, and explicitly state scenarios where it should NOT be used.
</details>

4. **You ask the agent to "compare weather in 5 cities." It calls get_weather 5 times sequentially. How do you speed this up?**

<details>
<summary>Answer</summary>
The LLM should return all 5 tool_use blocks in a single response (parallel calls). If it doesn't, check: (1) your system prompt might imply sequential thinking, (2) the model might not have seen these tools used together before. Once it returns multiple tool_use blocks, execute them with `Promise.all()` — this turns 5 sequential calls (~5s) into 1 parallel batch (~1s).
</details>

5. **When should you use an agent framework (LangChain, Vercel AI SDK) vs. building the loop yourself?**

<details>
<summary>Answer</summary>
Build it yourself first — understanding the loop at the code level makes debugging much easier. Use a framework when: you need pre-built tool integrations (LangChain has hundreds), you're building a Next.js streaming chat app (Vercel AI SDK's useChat + tools is optimal), or you need multi-agent orchestration (LangGraph). Frameworks add abstraction overhead — don't use them until you understand what they're abstracting.
</details>

---

## Go Deeper

1. **[Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)** — The official guide with complete code examples for every tool use pattern: single tool, multiple tools, parallel calls, streaming with tools. Start here. (1 hour)

2. **[Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)** — Anthropic's research post on what actually works in production agents. Covers when NOT to use agents, common failure modes, and design principles. Essential reading. (30 min)

3. **[LangGraph Documentation](https://langchain-ai.github.io/langgraph/)** — Best framework for multi-agent systems and complex agent workflows. Even if you don't use it, the concepts (nodes, edges, state) clarify agent architecture. (2 hours)

4. **[ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)** — The paper that formalized the Reason+Act pattern. Understanding the original formulation helps you design better prompts for agents. (20 min)

5. **[Anthropic Cookbook — Agents](https://github.com/anthropics/anthropic-cookbook/tree/main/tool_use)** — Working code examples of real agents: customer support, code execution, web search. Copy-paste starting points for your own agents. (ongoing reference)

---

**What's next?** Agents use tools to act in the world. Sometimes you need the model itself to behave differently — write in your brand voice, understand your domain terminology. That's [Fine-tuning →](fine-tuning-llms.md)
