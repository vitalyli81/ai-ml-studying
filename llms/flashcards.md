# LLM & AI Engineering Flashcards — Spaced Repetition Deck

> **How to use this file:** Don't read it — *quiz yourself with it.* Cover the **A:** line, answer out loud, then check. Review misses the same day, after 2 days, and after a week (see [README.md](README.md) → How to Study This Phase). Each `Q:`/`A:` pair is one Anki card. This deck doubles as interview prep — most of these ARE interview questions.

---

## LLM Fundamentals

**Q:** A chatbot "remembers" your name across turns. What is actually happening?
**A:** Nothing on the model side — every API call is a blank slate. Your code resends the entire conversation history each turn; the model re-reads it fresh. Memory is an illusion you build (and pay for, in input tokens).

**Q:** The context window in one sentence — and what happens to facts outside it?
**A:** The fixed-size "desk" holding everything the model can see in one call (system prompt + history + retrieved docs + its reply). Facts not on the desk and not in the trained weights don't exist to the model — which is why RAG exists.

**Q:** Why does an LLM hallucinate instead of saying "I don't know"?
**A:** It was trained to produce *plausible* next tokens, not *true* ones. When it lacks knowledge, the most likely-sounding continuation is generated in the same confident tone. Truth-checking is something you engineer (RAG, grounding, evals), not a property it has.

**Q:** The engineer's mental model of an LLM in one line?
**A:** A stateless function — text in → text out — with a fixed-size desk (context) and no built-in truth-checking. Memory, knowledge, and reliability are yours to build around it.

**Q:** Base model vs instruct model vs chat model — what separates them?
**A:** Pre-training (next-token on raw text → completes text), instruction fine-tuning ((instruction, answer) pairs → follows commands), RLHF (human preference rankings → helpful/safe). Ask a base model "What is 2+2?" and it continues the list: "What is 2+3?"

**Q:** What did Chinchilla correct about scaling laws?
**A:** GPT-3-era models were undertrained — for a fixed compute budget, balance parameters and tokens (~20 tokens per parameter). You can't just make models bigger; you must feed them proportionally more data.

**Q:** "Lost in the middle" — what is it and what's the practical rule?
**A:** Models lose focus on information buried in the middle of long contexts. Put critical instructions and the top retrieved facts near the start AND end of the prompt; the middle is a graveyard.

---

## Prompt Engineering

**Q:** The 5 building blocks of a production prompt?
**A:** System (role, rules), Context (background/docs), Task (what to do), Format (exact output spec), Examples (show, don't tell). Most bad prompts are missing one.

**Q:** Why does "think step by step" work, mechanically?
**A:** Tokens are generated sequentially, and each generated step becomes context for the next. The intermediate tokens act as a scratchpad — jumping straight to the answer skips that working memory. (For simple lookups it adds noise and cost — skip it.)

**Q:** Few-shot examples vs fine-tuning — where's the line?
**A:** 2–3 high-quality examples in the prompt handle custom formats and patterns. If you need more than ~5 examples, consistent behavior across thousands of calls, or examples cost too many tokens per call — that's fine-tuning territory.

**Q:** The prompt debugging ladder — first three rungs?
**A:** (1) Read the output literally — the model did what you asked; what did you actually ask? (2) Check the raw tokens for hidden chars/truncation. (3) Remove ambiguity — replace every "it/this/that" with a concrete noun. Swapping models is step 8, not step 1.

**Q:** What is self-consistency and when is it worth 5× the cost?
**A:** Sample the same prompt N times at temp > 0 and take the majority answer. Worth it for high-stakes reasoning where reliability beats cost/latency; never for routine calls.

**Q:** The #1 cost lever in prompting?
**A:** Move long, stable instructions into the system prompt and enable prompt caching. Few-shot examples ride along on every call — cached, they're ~90% cheaper.

---

## RAG

**Q:** What does RAG separate, and why does that matter?
**A:** Knowing (your searchable database) from reasoning (the LLM). Update the database in seconds — no retraining. Facts stay current, answers cite sources, hallucination drops.

**Q:** Starting numbers: chunk size, overlap, top_k?
**A:** 300–500 token chunks, 50-token overlap, top_k = 3–5. Too-small chunks lose context; too-large add noise; no overlap splits answers across boundaries.

**Q:** What is recall@k, and how does it localize the bug?
**A:** The fraction of eval queries where the correct chunk appears in the top-k retrieved. Low recall@10 → chunking/embedding problem. Low recall@3 but high recall@10 → add a re-ranker. High recall but wrong answers → the prompt/LLM, not retrieval.

**Q:** What is groundedness and why measure it separately from retrieval?
**A:** Whether every claim in the answer is supported by the retrieved context. Retrieval can be perfect and the LLM can still ignore the context and hallucinate — two different failure modes, two different metrics.

**Q:** What is HyDE and when do you reach for it?
**A:** Generate a hypothetical answer to a vague query, then search for documents similar to *that* instead of the query. Fixes retrieval for queries like "it's not working" that carry no technical vocabulary.

**Q:** Hybrid search — why combine BM25 with vectors?
**A:** Semantic search misses exact terms (error codes, SKUs, names); keyword search misses paraphrases. Users do both. Hybrid re-ranks by both scores and wins on each failure mode.

**Q:** You change embedding models. What must happen?
**A:** Re-embed the entire corpus. Vectors from different models are incompatible spaces — and pin the embedding model version to avoid this by accident.

---

## LLM APIs & SDKs

**Q:** Why are LLM APIs stateless, and what does that shift to you?
**A:** No sessions — each request is independent (which makes them horizontally scalable). Conversation state lives in YOUR app: load history, append turn, send all of it, save the reply.

**Q:** What does streaming actually improve?
**A:** Perceived latency (TTFT ~300ms instead of a 5–10s blank screen) — not total generation time. It also avoids gateway timeouts on long generations. Rule: always stream user-facing chat.

**Q:** JSON mode vs structured outputs (tool_choice / response_format with schema)?
**A:** JSON mode only guarantees syntactically valid JSON. Structured outputs guarantee YOUR schema — field names, types, enums, required fields. Production parsing uses the latter. (And still validate — schema-valid can be semantically nonsense.)

**Q:** Prompt caching: what's cached, the TTL, and the discount?
**A:** The prompt-prefix computation (KV-cache), not the response. Mark blocks with `cache_control: {type: 'ephemeral'}`; ~5-minute TTL; cached input tokens cost ~90% less and TTFT drops.

**Q:** The cost formula and the three shortcuts?
**A:** `(input_tokens × in_price + output_tokens × out_price) / 1M` per request. Shortcuts: Haiku ≈ $1, Sonnet ≈ $10, Opus ≈ $50 per million "typical" chatbot turns. Output tokens cost ~5× input.

**Q:** The one security rule for LLM API keys?
**A:** Server-side only — never in frontend code. Browser code is public; a shipped key will be stolen and billed. All LLM calls go through your API routes.

---

## Agents & Tool Use

**Q:** The division of responsibility in tool use?
**A:** The LLM *decides* what to call (it emits a structured tool_use request); your code *executes* it and returns the result. You control what can actually happen — never give the model direct execution.

**Q:** The agent loop in one sentence, and the two non-negotiable rails?
**A:** Call the model → if `stop_reason === 'tool_use'`, execute the tools, append results, loop → until `end_turn`. Rails: MAX_STEPS cap (a confused agent loops forever, billably) and confirmation gates on write/destructive tools.

**Q:** The LLM keeps picking the wrong tool — what do you fix first?
**A:** The tool *description* — it's what the model reads to choose. Specify when to use it, what it returns, and explicitly when NOT to use it. Names matter far less.

**Q:** Parallel vs sequential tool calls — what determines which?
**A:** Data dependency. Independent lookups (weather in Paris AND Tokyo) → multiple tool_use blocks in one turn, executed with Promise.all. Dependent steps (get PR → email its author) → must be sequential; forcing parallel produces hallucinated arguments.

**Q:** A tool throws an exception mid-loop. What's the production pattern?
**A:** Catch it and return a structured error as the tool_result (`is_error: true`) — the model can often recover (retry different args, or gracefully give up). Crashing the loop helps nobody.

**Q:** Why are agents 5–10× more expensive than single calls, and the top cost lever?
**A:** Each step is a full model call carrying the growing history plus tool outputs. Levers: cap steps, route tool-decisions to a cheap model and only synthesis to a big one, cache the stable system prompt, trim tool outputs.

---

## MCP

**Q:** What problem does MCP solve, and what's the canonical analogy?
**A:** The N×M integration problem — N agents × M tools each needing custom glue. Like LSP for editors×languages: build one MCP server, every compliant host can use it. "USB-C for AI."

**Q:** Host, client, server — one phrase each?
**A:** Host: the LLM app (Claude Desktop, Cursor, your agent). Client: one connection manager inside the host, 1:1 per server. Server: a small process exposing tools/resources/prompts over JSON-RPC.

**Q:** Tools vs resources vs prompts?
**A:** Tools: actions the *model* chooses to invoke (side effects allowed). Resources: read-only data with URIs the *host/user* attaches. Prompts: reusable templates the *user* picks. Don't collapse them.

**Q:** What is sampling in MCP?
**A:** The reverse direction: a *server* asks the *host* to run an LLM completion on its behalf (with user consent) — so servers can reason without their own API keys. Not the same as tool calling.

**Q:** stdio vs HTTP transport — the rule of thumb?
**A:** Start with stdio (host spawns the server as a local subprocess — fast, simple, single-user). Go HTTP/SSE only when you need remote, multi-user, or scaled access.

**Q:** The security reality of installing an MCP server?
**A:** It runs with the host's permissions — filesystem, network — like installing a CLI, not visiting a webpage. There is no sandbox unless you built one. Vet servers, pin versions, treat tool outputs as untrusted input.

---

## Evals

**Q:** "You don't iterate on a prompt, you iterate on a prompt ___" — finish it and explain.
**A:** "...against an eval set." Without a golden dataset, "better" has no meaning — you're shipping vibes. With one, every prompt/model change produces a number and a diff.

**Q:** Minimum viable eval set for shipping tomorrow?
**A:** 10–20 hand-written examples: ~3 happy paths, 3 edge cases, 3 adversarial/safety, 3 real user questions — with deterministic checks (contains, length, JSON-valid). Grow it from every bug report.

**Q:** Deterministic checks vs LLM-as-judge — the ordering rule?
**A:** Deterministic first: if a rule fits (contains, schema-valid, length, regex), it's 100× cheaper and never argues. Judge only what code can't express — subjective quality, semantic equivalence.

**Q:** The three classic judge biases and fixes?
**A:** Position bias (favors the first of a pair → randomize/swap order), verbosity bias (favors longer answers → explicit conciseness rubric), self-preference (scores its own outputs higher → judge with a different model family). Always validate the judge against ~50–100 human labels.

**Q:** Pass rate went 82% → 91%. Why can't you ship on that?
**A:** Aggregates hide category regressions — check the per-category diff. +9% overall with safety dropping 100% → 98% is a blocked deploy. Also check which specific examples flipped.

**Q:** Offline vs online evals — what does each uniquely catch?
**A:** Offline (golden set in CI): regressions on known cases, before deploy. Online (judge on 1–5% of real traffic): distribution shift and failure modes you never thought to include. Both required.

---

## Fine-tuning LLMs

**Q:** The one-line rule that saves weeks of wasted effort?
**A:** Fine-tuning changes HOW the model responds (style, format, behavior); RAG changes WHAT it knows (facts). Fine-tune on your product docs and it will mimic their style while still hallucinating the facts.

**Q:** The decision tree before any fine-tune?
**A:** Does the right prompt solve it? → done. Can better prompting fix it? → fix the prompt. Is the gap about KNOWING? → RAG. Only "reliably DOING a behavior that prompting can't reach" → fine-tune.

**Q:** LoRA in one picture?
**A:** Freeze W (the big matrix); train two small matrices A and B alongside it; output = W·x + scale·(B·A)·x. Like a CSS override file on top of an untouched stylesheet — ~0.1–1% of parameters, adapter ships as a 10–100 MB file.

**Q:** The r (rank) parameter — start where, and what happens at r=256 on 50 examples?
**A:** Start r=16 (r=8 for simple style/format changes). r=256 on 50 examples = massive capacity + tiny data = memorization; the model parrots training responses verbatim.

**Q:** QLoRA's three tricks?
**A:** (1) Quantize the frozen base to 4-bit NF4, (2) double quantization (quantize the quantization constants), (3) paged optimizers (spill to CPU). Net: 70B fine-tunes on one 48 GB GPU; 7–13B on a 24 GB consumer card.

**Q:** Catastrophic forgetting after fine-tuning — symptom and fix?
**A:** Great at your task, suddenly can't do general things (support bot that can't code). Fix: mix 5–20% general instruction data into training, prefer LoRA over full fine-tuning, keep general-capability benchmarks in your eval.

**Q:** The serving break-even math — when does self-hosting a tuned model pay?
**A:** Break-even ≈ GPU $/hr ÷ (API $/req − self-host $/req) ≈ $2 / $0.0095 ≈ 210 req/hr (~5K/day). Below that, API + good prompting wins; above it, a self-hosted tuned small model can save 10–100×.

---

## Production Patterns

**Q:** What fraction of a production LLM system is the model, and what's the rest?
**A:** ~30%. The rest is plumbing: caching, streaming, structured outputs, token budgeting, conversation state, retries/fallbacks, observability. That plumbing is the AI engineer's job.

**Q:** You edit one word at the START of a cached system prompt. What happens?
**A:** Cache invalidation cascade — every downstream token's position shifts, the KV-prefix no longer matches, hit rate crashes to 0%, cost roughly doubles. Keep the cached block byte-stable; add changes after it. Golden-hash the prefix in CI.

**Q:** The long-conversation strategies, simplest to fanciest?
**A:** Truncation (drop oldest turns) → sliding window + pinned system prompt → summarize old turns → RAG over chat history → hierarchical summarization. Decide the policy before you ship, not at turn 50.

**Q:** Retry strategy for 429s and 5xx?
**A:** Exponential backoff + jitter; respect `Retry-After`; circuit-breaker a persistently failing provider; fall back to a second provider; never retry in tight loops (you deepen the outage — retry storms).

**Q:** Is temperature 0 fully deterministic?
**A:** No — kernel nondeterminism and load balancing across GPU nodes still cause small variations. It's greedy sampling, not a determinism guarantee (and definitely not an accuracy guarantee).

**Q:** The six things to log on every LLM call?
**A:** Model version, input/output/cached token counts, latency (TTFT + total), cost estimate, conversation_id, prompt version. Without these, production debugging is guesswork.

---

## System Design

**Q:** The 6-beat framework for any LLM system design interview?
**A:** Clarify → Estimate → API & data model → High-level architecture → Deep dive → Scale & failure modes. The #1 mistake: drawing boxes in minute one — scope before you sketch.

**Q:** The single most important clarifying question?
**A:** "What's the ONE metric we optimize for?" Latency, cost, accuracy, and coverage trade off against each other; the ranking shapes the entire architecture.

**Q:** Do the estimation chain: 50K DAU × 5 queries/day?
**A:** 250K queries/day ≈ 3 QPS average (day ≈ 100K sec), peak ≈ 10× ≈ 30 QPS. At ~1.5K tokens/query → ~375M tokens/day → mid-tier model ≈ $2.6K/day ≈ $80K/month. The number tells you what to build: caching + model routing.

**Q:** What makes an LLM call different from a normal upstream API call, architecturally?
**A:** It's non-deterministic, expensive per-token, slow (seconds), and fallible (hallucinates, gets injected). Hence: cache aggressively, stream, eval-gate deploys, guardrail inputs/outputs, budget tokens like money.

**Q:** The "senior move" for the cost question?
**A:** Model routing/cascade — a cheap model handles easy queries, a router escalates hard ones to the flagship — paired with prompt caching on the static prefix. Attacks the cost estimate directly with a named quality tradeoff.

**Q:** The guaranteed interview question, and your layered answer?
**A:** "What happens when the LLM provider goes down?" Fallback provider → retries with backoff + jitter → circuit breaker → degraded/cached response → alerting. Close every design with failure modes + the metrics you'd monitor.

**Q:** How do you stop a prompt change from silently breaking production?
**A:** Eval gate in CI (golden set must not regress, per-category thresholds) → canary rollout to 1–5% with online judge + cost + latency watch → instant rollback. "Did it help?" becomes a number.
