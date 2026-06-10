# LLM System Design (Interview & Real-World)

## TL;DR

**System design** is the interview round where you're handed a vague prompt — *"Design a RAG chatbot for 10 million users"* — and asked to architect it out loud in 45 minutes. There's no single right answer; you're scored on how you scope the problem, reason about tradeoffs, estimate scale, and recover when the interviewer pushes back. For AI/LLM roles, it blends classic system design (load balancers, caches, queues, databases) with LLM-specific concerns (token cost, latency budgets, evals, hallucination, prompt injection). This is the **single highest-signal round for senior roles** — it's where you prove you can own a system, not just a feature.

> 💡 **Key Insight:** They're not testing whether you know the "correct" architecture. They're testing whether you drive the conversation: clarify → estimate → sketch → justify tradeoffs → handle failure. A confident wrong answer you defend with reasoning beats a memorized right one you can't explain.

---

## The Mental Model

**Think of it like being an architect pitching a building, not a contractor laying bricks.**

A client says "I want a building." A bad architect immediately starts drawing walls. A good one asks: how many people? Offices or apartments? What's the budget? What's the worst earthquake it must survive? *Then* sketches, explains why steel over wood here, and shows where the fire exits go before the client asks.

| Architect pitching a building | LLM system design interview |
|-------------------------------|-----------------------------|
| "How many occupants? What's the budget?" | Clarifying questions (scale, latency, cost, quality bar) |
| Back-of-envelope: floors × people | Capacity estimation (QPS, tokens/day, $/month) |
| Blueprint with rooms and hallways | High-level architecture diagram (boxes and arrows) |
| "Steel here because earthquakes" | Justifying each component with a tradeoff |
| Fire exits, sprinklers | Failure modes, fallbacks, rate limits |
| "Phase 2: add a parking garage" | Scaling the design when the interviewer 10×'s the load |

---

## Why It Exists (Problem → Solution)

**The problem:** Anyone can build a demo. The hard part is the system *around* the model — the part that survives 10M users, a provider outage at 3 a.m., a malicious prompt, and a finance team asking why the bill tripled.

**What it filters for:** Coding rounds test "can you write a function." System design tests "can you own a service." At senior level, the second matters more — you'll spend your career deciding *where the cache goes* and *what happens when the LLM API 500s*, not writing leetcode.

**What's LLM-specific:** Classic system design assumes deterministic, cheap, fast operations. LLM calls are **non-deterministic, expensive (per-token), slow (seconds), and fallible (hallucinate, get jailbroken)**. That changes the architecture: you cache aggressively, you stream, you eval, you guardrail, you budget tokens like money.

---

## The Framework (memorize this — it's your script)

Every LLM system design answer follows the same 6 beats. Walk them in order, out loud. **Spend the first 5–10 minutes NOT drawing.**

```
1. CLARIFY      → ask questions, pin down scope and the one metric that matters
2. ESTIMATE     → QPS, tokens/day, cost/month, latency budget (back-of-envelope)
3. API + DATA   → define the interface and the data model first
4. HIGH-LEVEL   → draw the boxes: client → gateway → app → model → stores
5. DEEP DIVE    → interviewer picks one box; you go deep on tradeoffs
6. SCALE/FAIL   → bottlenecks, caching, fallbacks, failure modes, monitoring
```

> 💡 **The #1 mistake:** jumping to step 4 (drawing boxes) in minute one. The interviewer wants to see you *scope* before you *build*. Resist the urge to architect until you've clarified and estimated.

---

## Core Concepts

### 1. Clarifying Questions — Scope Before You Solve

**One-liner:** Turn a deliberately vague prompt into a concrete, bounded problem.

**Analogy:** A waiter who repeats your order back before sending it to the kitchen. "So that's a RAG chatbot, internal docs only, 10M users but ~50K daily active, answers must cite sources, p95 under 3 seconds — correct?"

**The four dimensions to always pin down:**

```
SCALE      → How many users? QPS? Peak vs average? Read-heavy or write-heavy?
LATENCY    → Real-time chat (stream, <3s) or async batch (minutes OK)?
QUALITY    → What's the cost of a wrong answer? Legal/medical = high. Casual = low.
CONSTRAINTS→ Budget? Data privacy (PII, on-prem)? Which models allowed?
```

**The single most important question:** *"What's the one metric we optimize for?"* Latency, cost, accuracy, and coverage trade off against each other. Forcing the interviewer to rank them tells you the whole shape of the answer.

**Common misconception:** People think asking questions makes you look unsure. The opposite — senior engineers scope ruthlessly. Juniors start coding immediately.

---

### 2. Capacity Estimation — Back-of-Envelope Math

**One-liner:** Rough numbers that determine your architecture: do you need 1 server or 1,000?

**Analogy:** Sizing a restaurant. 50 covers a night means one chef. 5,000 means an industrial kitchen, multiple shifts, and a supply chain. You can't design the kitchen until you've guessed the covers.

**The numbers you'll be asked to derive:**

```
USERS → QPS
  10M total users, 50K daily active, 5 queries each   = 250K queries/day
  250K / 86,400 sec                                    ≈ 3 QPS average
  Peak is ~10× average                                 ≈ 30 QPS peak

QPS → TOKENS
  30 QPS × (1K input + 500 output tokens)              = 45K tokens/sec
  Per day: 250K queries × 1.5K tokens                  ≈ 375M tokens/day

TOKENS → COST  (use round numbers; ~$3/M in, ~$15/M out for a mid-tier model)
  Input:  250M tok/day × $3/M   ≈ $750/day
  Output: 125M tok/day × $15/M  ≈ $1,875/day
  Total                          ≈ $2,600/day ≈ $80K/month
  → "That's the lever caching and a cheaper model attack."

LATENCY BUDGET  (work backwards from the p95 target, e.g. 3s)
  Retrieval 200ms + prompt build 50ms + LLM TTFT 800ms + stream rest
  → stream the output so perceived latency = time-to-first-token, not total
```

> 💡 **You don't need exact numbers.** Round aggressively (a day ≈ 100K seconds, not 86,400). The interviewer wants to see the *method* and that you know cost/latency are first-class design forces in LLM systems.

**Common misconception:** Estimation is just trivia. No — the numbers *drive the design*. $80K/month justifies a caching layer and a model-routing tier. 30 QPS means you don't need exotic sharding. The math tells you what to build.

---

### 3. API & Data Model First

**One-liner:** Define the contract (request/response) and what you store before drawing infrastructure.

**Analogy:** Agreeing on the menu and the recipe cards before designing the kitchen. The interface constrains everything downstream.

```python
# The API contract — this anchors the whole design
POST /v1/chat
{
  "session_id": "uuid",          # multi-turn state lives server-side or in the client?
  "message": "How do I reset my password?",
  "stream": true                  # almost always true for chat
}
# Response: server-sent events (tokens) + a final message with citations

# The data you persist
sessions     (session_id, user_id, created_at, ...)
messages     (message_id, session_id, role, content, tokens, cost, latency_ms)
documents    (doc_id, source, version, ...)        # for RAG corpus
embeddings   (chunk_id, doc_id, vector, metadata)  # in the vector store
traces       (trace_id, message_id, retrieved_ids, prompt, model, judge_score)
```

**Common misconception:** Data modeling is a backend afterthought. In LLM systems it's central — *where does multi-turn state live? how do you version the corpus? where do traces go for eval?* These decisions shape the whole system.

---

### 4. The Reference Architecture (your default sketch)

**One-liner:** The boxes-and-arrows you draw for almost any LLM app, then customize.

```
                         ┌─────────────────────────────────────────────┐
  Client (web/mobile)    │              APPLICATION TIER                 │
  ──────────────►  API   │                                               │
  (streaming SSE)  Gateway├─► Rate limiter ─► Auth ─► Guardrails (in) ──┐ │
                         │                                              │ │
                         │   ┌──────────────────────────────────────┐  │ │
                         │   │  Orchestrator (the "brain")           │◄─┘ │
                         │   │  • build prompt   • route model       │    │
                         │   │  • call tools     • assemble context  │    │
                         │   └──┬──────────┬──────────┬──────────────┘    │
                         │      │          │          │                   │
                         └──────┼──────────┼──────────┼───────────────────┘
                                ▼          ▼          ▼
                        ┌───────────┐ ┌─────────┐ ┌──────────────┐
                        │ Semantic  │ │ Vector  │ │  LLM Provider│
                        │  + exact  │ │   DB    │ │  (+ fallback │
                        │   cache   │ │ (RAG)   │ │   provider)  │
                        └───────────┘ └─────────┘ └──────┬───────┘
                                                         │
                                          Guardrails (out) ─► stream to client
                                                         │
                                          Async: trace + eval + cost log ─► Observability
```

**What each box buys you (say this out loud):**

| Component | Why it's there | Tradeoff |
|-----------|----------------|----------|
| API gateway | One entry point, TLS, routing | Single point of failure → make it HA |
| Rate limiter | Stop abuse and runaway cost | Too aggressive → bad UX for power users |
| Guardrails (in) | Block prompt injection, PII | Adds latency; can false-positive |
| Orchestrator | Coordinates prompt/RAG/tools/model | Where complexity concentrates — keep it stateless |
| Semantic + exact cache | Cut cost & latency on repeat queries | Stale answers; cache invalidation is hard |
| Vector DB | Retrieve grounding context (RAG) | Index freshness, re-embedding on model swap |
| LLM + fallback | The actual generation | Cost, latency, non-determinism, outages |
| Guardrails (out) | Block unsafe/hallucinated output | Adds latency before streaming |
| Observability (async) | Traces, cost, evals — off the hot path | Storage cost; must not block the response |

> 💡 **Keep the orchestrator stateless.** Push session state to a store (Redis/Postgres) so you can scale the app tier horizontally behind a load balancer. This is the classic "stateless app + stateful store" pattern — same as any web backend.

---

### 5. The Tradeoffs They'll Push On

**One-liner:** The interviewer 10×'s the scale or constrains the budget and watches you adapt.

These are the levers. Know which direction each one moves cost / latency / quality:

```
LEVER                    PULL FOR LATENCY/COST    PULL FOR QUALITY
─────────────────────────────────────────────────────────────────────
Model size               smaller/cheaper model     bigger/flagship model
Model routing            cheap model for easy Qs    always use the best
Caching (exact+semantic) cache hard, serve stale    no cache, always fresh
Prompt caching           cache the long system msg  (free win, almost always do it)
top_k / chunk size       fewer, smaller chunks      more context
Streaming                stream (better perceived)  (always stream chat)
Reranking                skip it                    add a cross-encoder pass
Guardrails depth         lightweight regex          full LLM-based moderation
Batch vs real-time       batch async jobs           per-request real-time
```

**The senior move:** propose **model routing / cascades** — a cheap model (Haiku-tier) handles easy queries and a router escalates hard ones to a flagship. This single pattern attacks the cost estimate from step 2 directly. Pair it with **prompt caching** (cache the static system prompt + retrieved context) for an easy 5–10× cost cut on repeated prefixes.

**Common misconception:** There's an optimal architecture. No — every choice is a tradeoff against the metric you pinned in step 1. If they said "cost matters most," you route to cheap models and cache hard. If they said "accuracy is everything" (legal/medical), you use the flagship, add reranking, and never serve stale cache.

---

### 6. Failure Modes & Reliability

**One-liner:** What breaks, and what the system does when it breaks.

**Analogy:** A restaurant's contingencies — the supplier doesn't show (provider outage), a customer has an allergy (guardrail), the kitchen is slammed (rate limit + queue).

The failures you must mention (interviewers wait for these):

```
LLM provider outage      → fallback to a second provider; cached/degraded response
LLM provider slow/timeout→ timeout + retry with backoff; stream partial; circuit breaker
Rate-limited (429)       → exponential backoff + jitter; queue non-urgent work
Hallucination            → RAG grounding + LLM-as-judge sampling + "I don't know" fallback
Prompt injection         → input guardrails; treat retrieved docs as untrusted; delimit
Cost spike               → per-user + global token budgets; alerts; kill switch
Bad deploy (prompt regr.)→ eval gate in CI; canary rollout; instant rollback
Vector index stale       → version the corpus; delta re-embed; pin embedding model
```

> 💡 **Always close with monitoring.** "I'd track p50/p95 latency by stage, cost per query, cache hit rate, guardrail trigger rate, and an LLM-as-judge quality score on 1% of traffic." This signals you've actually run something in production. See [../ml-ops/llm-observability.md](../ml-ops/llm-observability.md).

---

> 🧠 **Quick recall — answer out loud before scrolling on** (all answers are above):
> 1. The 6 beats of the framework, in order — and which one do people skip in minute one?
> 2. The single most important clarifying question?
> 3. Derive it: 50K DAU × 5 queries → QPS average and peak?
> 4. What makes an LLM call architecturally different from a normal upstream API call?
> 5. The "senior move" that attacks the cost estimate directly?

---

## How It Actually Works — A Worked Example

**Prompt:** *"Design a customer-support RAG chatbot for an e-commerce company. 10M registered users."*

**Step 1 — Clarify (out loud):**
> "A few questions: ~50K daily active users sound right? Answers must cite the help-center doc they came from? Real-time chat with streaming, so p95 under ~3s? English-only to start? And what's the one metric — I'd guess answer accuracy, since a wrong refund-policy answer is costly. Agreed? Then I'll optimize for accuracy with cost as a guardrail."

**Step 2 — Estimate:**
> "50K DAU × 5 messages ≈ 250K/day ≈ 3 QPS average, ~30 peak. ~1.5K tokens/message → ~375M tokens/day → roughly $2–3K/day if I use a flagship for everything. That cost says: cache and route models."

**Step 3 — API & data:**
> "`POST /v1/chat` with `session_id`, `message`, `stream:true`; SSE response with a citations array. I store sessions, messages (with cost/latency/tokens), the doc corpus, embeddings in the vector DB, and traces for eval."

**Step 4 — Draw the reference architecture** (the diagram above), narrating each box.

**Step 5 — Deep dive** (interviewer: *"go deep on retrieval"*):
> "Recursive chunking ~400 tokens, 50 overlap. Hybrid search (BM25 + vector) because users paste exact error codes *and* paraphrase. top_k=5 → cross-encoder rerank → top_3 into the prompt. I'd measure recall@k on a golden query set in CI before blaming the model." (See [rag.md](rag.md).)

**Step 6 — Scale & fail:**
> "At 10× load I scale the stateless app tier behind the LB and add read replicas for the metadata DB; the vector DB and LLM provider are the real bottlenecks. Failures: fallback to a second LLM provider, exponential backoff on 429s, eval-gated canary deploys with instant rollback, per-user token budgets, and input guardrails treating retrieved docs as untrusted. I'd monitor cost/query, cache hit rate, and a 1%-sampled LLM-judge accuracy score."

That's a complete senior-level answer in 45 minutes.

---

## Code in Practice

A system design round is mostly talking and drawing, but interviewers love when you can drop to a concrete skeleton of the orchestrator — it proves the boxes map to real code.

```python
# The orchestrator — the "brain" box. Stateless; state lives in stores.
async def handle_chat(req: ChatRequest) -> StreamingResponse:
    # 1. Guardrails (in) + rate limit happen at the gateway, before this.
    # 2. Exact + semantic cache check — cheapest possible path.
    if cached := await cache.get(req.message, req.session_id):
        return stream(cached)

    # 3. Retrieve grounding context (RAG).
    chunks = await retriever.search(req.message, top_k=5)
    chunks = await reranker.rerank(req.message, chunks, top_n=3)

    # 4. Route the model by difficulty — the cost lever.
    model = "haiku-tier" if is_simple(req.message) else "sonnet-tier"

    # 5. Build the prompt; cache the static system + context prefix.
    prompt = build_prompt(system=SYSTEM, context=chunks, message=req.message)

    # 6. Call the LLM with a fallback provider + retry/backoff.
    stream_resp = await llm.stream(model, prompt, fallback="other-provider")

    # 7. Guardrails (out) wrap the stream; trace + cost log fire async.
    asyncio.create_task(observability.log(req, chunks, model))
    return guarded_stream(stream_resp)
```

Each numbered comment is a box in the diagram. If you can narrate this, you've connected architecture to implementation — exactly what they want.

---

## Gotchas & Pitfalls

```
❌ Drawing boxes in minute one → ✅ Clarify + estimate for 5–10 min first
   The interviewer wants scoping. Architecture without scope scores low.

❌ "I'd just use GPT/Claude" with no surrounding system → ✅ The system IS the answer
   The model is one box. Caching, RAG, guardrails, fallbacks, evals are the design.

❌ Ignoring cost → ✅ Estimate $/month and name the levers (routing, caching)
   For LLM systems, cost is a first-class design force, not an afterthought.

❌ Treating LLM calls like cheap deterministic functions → ✅ Slow, costly, fallible
   Design for seconds of latency, per-token cost, non-determinism, and outages.

❌ No streaming → ✅ Stream chat responses
   Perceived latency = time-to-first-token. A frozen UI for 3s feels broken.

❌ Forgetting failure modes → ✅ Provider fallback, retries, rate limits, kill switch
   "What happens when the LLM API is down?" is a guaranteed question.

❌ No evals / monitoring → ✅ Close with the eval gate + the metrics you'd track
   Skipping this is the clearest "never shipped to prod" signal.

❌ Over-engineering for scale you don't have → ✅ 30 QPS doesn't need exotic sharding
   Match the design to the estimated load. Don't bring Kafka to a 3-QPS fight.
```

---

## When to Use Which Pattern

| If the prompt emphasizes... | Reach for... |
|-----------------------------|--------------|
| "Answers from our documents" | RAG (vector DB + hybrid search + rerank) — [rag.md](rag.md) |
| "Take actions / multi-step" | Agent + tool use, with guardrails on each tool — [agents-tool-use.md](agents-tool-use.md) |
| "Millions of users, tight budget" | Model routing/cascade + aggressive caching + prompt caching |
| "Real-time chat" | Streaming (SSE), TTFT-optimized, stateless app + state store |
| "Process a backlog / nightly" | Batch async jobs + queue, not real-time per request |
| "Can't leak data / on-prem" | Self-hosted open model, self-hosted vector DB, no external API |
| "Must be accurate (legal/med)" | Flagship model + rerank + LLM-judge + human-in-the-loop + no stale cache |

---

## Related Concepts (The Map)

| If you know... | LLM system design is like... |
|----------------|------------------------------|
| Designing a web backend (LB + app + DB + cache) | Same skeleton — the LLM provider is a slow, costly, flaky upstream API |
| Frontend perf budgets (TTFB, LCP) | Latency budgets — work backwards from p95, stream to fix perceived latency |
| CDN + cache invalidation | Semantic + exact caching — same "fast but stale" tradeoff, harder invalidation |
| Microservice fallbacks / circuit breakers | Provider fallback + retries + circuit breaker on the LLM call |
| Rate limiting an API you've shipped | Same, but the cost-per-call is 100–1000× higher, so budgets matter more |

**Connected topics:**
- **[RAG](rag.md)** → the retrieval box, the most common deep-dive
- **[Production LLM Patterns](production-llm-patterns.md)** → caching, retries, cost — the reliability boxes
- **[Agents & Tool Use](agents-tool-use.md)** → when the system takes actions, not just answers
- **[Evals](evals.md)** → the CI gate that makes deploys safe; always mention it
- **[../ml-ops/reliability-patterns.md](../ml-ops/reliability-patterns.md)** → fallbacks, circuit breakers, backoff in depth
- **[../ml-ops/llm-observability.md](../ml-ops/llm-observability.md)** → the monitoring you close every answer with

---

## Cheat Sheet

| Term | One-line definition |
|------|---------------------|
| QPS | Queries per second; peak ≈ 10× average |
| Capacity estimation | Back-of-envelope users → QPS → tokens → $/month |
| Latency budget | Work backwards from p95; stream to optimize perceived latency |
| Reference architecture | Client → gateway → orchestrator → {cache, vector DB, LLM+fallback} → observability |
| Model routing / cascade | Cheap model for easy queries, escalate hard ones — the cost lever |
| Prompt caching | Cache the static system+context prefix — easy 5–10× cost cut |
| Stateless app tier | App holds no session state → scales horizontally behind a LB |
| Circuit breaker | Stop calling a failing upstream; fail fast and fall back |
| Eval gate | CI step that blocks a deploy if the golden-set score drops |

**The 6-beat framework (your script):**
```
1. Clarify   — scope + the one metric that matters
2. Estimate  — QPS, tokens, $/month, latency budget
3. API/Data  — contract and what you store
4. High-level— draw the reference architecture
5. Deep dive — go deep where they point
6. Scale/Fail— bottlenecks, fallbacks, monitoring
```

**Remember these 3 things:**
1. **Scope before you sketch.** Spend 5–10 minutes clarifying and estimating before drawing.
2. **Cost and latency are first-class.** Estimate $/month; name the levers (routing, caching).
3. **Always close with failure modes + monitoring.** It's the "I've shipped this" signal.

---

## Self-Check Questions

1. **The interviewer says "design a chatbot." What are the first things out of your mouth?**

<details>
<summary>Answer</summary>
Clarifying questions, not architecture. Pin down scale (users/QPS/peak), latency requirement (real-time stream vs async), quality bar (cost of a wrong answer), and constraints (budget, privacy, allowed models). Then ask the single most important one: "what's the one metric we optimize for?" Only after that do you estimate and sketch.
</details>

2. **You estimate ~$80K/month in LLM costs. The interviewer says that's too high. What do you propose?**

<details>
<summary>Answer</summary>
Pull the cost levers: (1) **Model routing/cascade** — a cheap model handles easy queries, escalate only hard ones to the flagship. (2) **Prompt caching** — cache the static system prompt + retrieved context prefix for a 5–10× cut on repeated prefixes. (3) **Semantic + exact caching** for repeated queries. (4) Smaller/fewer retrieved chunks. (5) Cap max output tokens. Name the quality tradeoff for each — you're trading some accuracy/freshness for cost.
</details>

3. **Why keep the orchestrator/app tier stateless, and where does session state go?**

<details>
<summary>Answer</summary>
A stateless app tier scales horizontally — you can run N identical instances behind a load balancer and any one can handle any request. Session/multi-turn state lives in an external store (Redis for speed, Postgres for durability). This is the standard "stateless compute + stateful store" web pattern; LLM systems are no different.
</details>

4. **"What happens when the LLM provider has an outage?" — walk through your answer.**

<details>
<summary>Answer</summary>
Layered defenses: (1) **Fallback provider** — route to a second LLM vendor with an equivalent model. (2) **Retries with exponential backoff + jitter** for transient errors/timeouts. (3) **Circuit breaker** — after repeated failures, stop hammering the dead provider and fail fast. (4) **Degraded response** — serve a cached answer or an honest "I'm having trouble, try again" rather than hanging. (5) **Alerting** so a human knows. See [../ml-ops/reliability-patterns.md](../ml-ops/reliability-patterns.md).
</details>

5. **How do you make sure a prompt change doesn't silently break the system in production?**

<details>
<summary>Answer</summary>
An **eval gate in CI**: every prompt/model change runs against a golden dataset and must not drop the score below threshold, or the deploy is blocked. Then **canary rollout** — ship to 1–5% of traffic, watch online quality (LLM-as-judge on a sample) + cost + latency, and **roll back instantly** if metrics regress. Evals turn "did this help?" from a guess into a number. See [evals.md](evals.md).
</details>

---

## Go Deeper

1. **[System Design Primer](https://github.com/donnemartin/system-design-primer)** — The canonical free resource for the *classic* half (load balancers, caching, sharding, CAP). LLM design sits on top of these fundamentals; learn them first. (ongoing reference)

2. **[Chip Huyen — "Building LLM applications for production"](https://huyenchip.com/2023/04/11/llm-engineering.html)** — The best single essay on what actually goes into a production LLM system: prompt versioning, evals, cost, latency, the whole stack. Read it twice. (45 min)

3. **[Anthropic — Building effective agents](https://www.anthropic.com/research/building-effective-agents)** — When the design is agentic, this is the reference for workflow vs. agent patterns and when to keep it simple. (30 min)

4. **[Hamel Husain — "Your AI product needs evals"](https://hamel.dev/blog/posts/evals/)** — Why the eval gate is the backbone of any serious LLM system, with concrete practice. The mindset interviewers probe for. (40 min)

5. **Practice out loud.** Pick three prompts — *RAG support bot*, *coding agent over MCP*, *content moderation pipeline* — and run all 6 beats on a whiteboard, timing yourself to 45 minutes. Designing in your head ≠ designing under interview pressure. (ongoing)

---

**What's next?** You can now scope and defend a full LLM system. Pair this with the reliability and observability deep-dives that back up your "scale & fail" beat: [../ml-ops/reliability-patterns.md →](../ml-ops/reliability-patterns.md)
