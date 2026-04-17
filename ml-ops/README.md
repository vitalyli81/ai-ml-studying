# Phase 6: MLOps & Production

> Taking AI models from your laptop to the real world — reliably, repeatedly, at scale.

**MLOps** (Machine Learning Operations) is the discipline of **shipping and maintaining AI systems in production**. You already know how to build models — MLOps is everything that makes them actually work for real users over time.

## The Mental Model

> 💡 **Think of it like this:** If building a model is like **cooking a dish at home**, MLOps is like **running a restaurant**.
>
> Cooking once for yourself is easy. Running a restaurant means: ingredients arrive on a schedule, the kitchen runs 24/7, every plate is consistent, you track cost per dish, you handle allergies (safety), you train new chefs (CI/CD), and when a customer complains you can trace which batch of beef was bad (observability).

| Cooking at home | Running a restaurant (MLOps) |
|-----------------|------------------------------|
| Cook once, eat, done | Serve thousands of orders per day |
| You taste and adjust | Automated quality checks on every plate |
| No one complains if it's off | One bad plate = bad review, lost trust |
| Fresh ingredients today | Ingredients drift — supplier changes, seasons change |
| You know the recipe | Recipe needs to be versioned so any chef can reproduce it |

## Frontend Analogy

You already do the web equivalent of MLOps every day:

```javascript
// You don't ship code straight from localhost to users. You have:
// - CI/CD (GitHub Actions)          → CI/CD for ML (MLflow, W&B)
// - Error tracking (Sentry)          → LLM observability (Langfuse, Helicone)
// - Feature flags (LaunchDarkly)     → Shadow deploys, canaries
// - Rate limiting, retries           → Same, but for model calls
// - A/B tests on UI                  → A/B tests on prompts and models
// - Input validation (Zod)           → Guardrails (PII, prompt injection)
// - CDN caching                      → Prompt caching + semantic caching
//
// MLOps = "DevOps for models." Same instincts, new failure modes.
```

The new failure modes are what make MLOps its own discipline: models **drift** (the world changes, accuracy decays), outputs are **non-deterministic** (same prompt, different answer), and **quality is fuzzy** (no green/red test — you need evals).

## Research ML vs. Production ML

```
Research ML:                    Production ML:
──────────────────────────────────────────────────────────────
"Does the model work?"          "Does it work for users, today, at 3 a.m., under load?"
Run once on a notebook          Runs 24/7, must survive restarts and outages
1 dataset, 1 result             Continuous data, distribution drifts over time
You validate manually           Automated evals gate every deploy
No one else uses it             10k–10M requests per day
Cost = your laptop              Cost = line item on the P&L
```

## Topics

| # | Topic | File | Why it matters |
|---|-------|------|----------------|
| 1 | Model Serving | [model-serving.md](model-serving.md) | Turn a `.pkl` / API key into a real endpoint (FastAPI, Docker) |
| 2 | Vector Databases | [vector-databases.md](vector-databases.md) | The storage layer behind every RAG system (Pinecone, pgvector, Chroma) |
| 3 | LLM Observability | [llm-observability.md](llm-observability.md) | Traces, cost, latency, prompt versions — you can't fix what you can't see |
| 4 | Safety & Guardrails | [safety-guardrails.md](safety-guardrails.md) | PII redaction, prompt injection defense, output validation |
| 5 | Reliability Patterns | [reliability-patterns.md](reliability-patterns.md) | Retries, fallbacks, circuit breakers, caching — survive provider outages |
| 6 | Experimentation | [experimentation.md](experimentation.md) | A/B tests, shadow deploys, canaries — prove a change actually helps |
| 7 | Evaluation & Monitoring | [evaluation-monitoring.md](evaluation-monitoring.md) | Offline evals + online quality metrics — catch regressions before users do |
| 8 | CI/CD for ML | [cicd-for-ml.md](cicd-for-ml.md) | Automate training, evaluation, and safe rollout (MLflow, W&B) |

## Learning Path

```
         Foundation                Production quality             Release engineering
──────────────────────────    ────────────────────────────    ──────────────────────────
Model Serving  ─┐
                ├─▶  Observability ─▶ Guardrails ─▶ Reliability ─▶ Experimentation ─┐
Vector DBs    ──┘                                                                    │
                                                                                     ▼
                                                           Evaluation & Monitoring
                                                                     │
                                                                     ▼
                                                              CI/CD for ML
```

Read in this order if you're new:

1. **Model Serving** — how a model becomes a callable API
2. **Vector Databases** — storage layer for RAG and semantic search
3. **LLM Observability** — before users arrive, you need traces
4. **Safety & Guardrails** — input/output checks that protect users and the business
5. **Reliability Patterns** — retries, fallbacks, caching, circuit breakers
6. **Experimentation** — A/B tests, shadow deploys, canaries — proving a change works
7. **Evaluation & Monitoring** — offline evals + online quality metrics
8. **CI/CD for ML** — automate training, evaluation, and safe deploys

## The Four Pillars (memorize these)

Every production AI system stands on four pillars. If any one is missing, you're gambling.

```
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │ OBSERVABILITY│   │    SAFETY    │   │  RELIABILITY │   │  EVALUATION  │
        ├──────────────┤   ├──────────────┤   ├──────────────┤   ├──────────────┤
        │ See what     │   │ Block bad    │   │ Survive when │   │ Prove quality│
        │ happened     │   │ inputs &     │   │ providers or │   │ before and   │
        │ (traces,     │   │ outputs      │   │ the model    │   │ after every  │
        │ cost, latency│   │ (PII, jail-  │   │ misbehaves   │   │ deploy       │
        │ prompts)     │   │ break, etc.) │   │ (retries,    │   │ (golden sets,│
        │              │   │              │   │ fallbacks)   │   │ LLM-judge)   │
        └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

> 💡 **Key Insight:** Evals are the one pillar most people skip — and the one that separates "it demoed well" from "it works in production." Start writing them on day one.

## What You'll Be Able to Do After This Phase

- Deploy any ML model as a REST API with Docker
- Build a production RAG system backed by a real vector DB
- Instrument every LLM call with traces, costs, and prompt versions
- Defend against prompt injection, PII leaks, and unsafe outputs
- Survive provider outages with retries, fallbacks, and caching
- Run A/B tests and canary rollouts on prompt or model changes
- Monitor LLM outputs for quality, cost, and safety regressions
- Set up automated pipelines that retrain, evaluate, and redeploy models
- Track experiments so you can reproduce any result from 3 months ago

## Your Frontend Advantage (again)

Most MLOps tools expose dashboards, trace viewers, A/B UIs, and eval consoles. You already know how to build and reason about those — you've shipped them for years. When the ML team's eval UX is painful, you can fix it. That's a rare combination on an AI team.
