# Phase 6: MLOps & Production

> Taking AI models from your laptop to the real world — reliably, repeatedly, at scale.

MLOps (Machine Learning Operations) is the discipline of **shipping and maintaining AI systems in production**. You already know how to build models — MLOps is everything that makes them actually work for real users over time.

## Topics

| # | Topic | File |
|---|-------|------|
| 1 | Model Serving | [model-serving.md](model-serving.md) |
| 2 | Vector Databases | [vector-databases.md](vector-databases.md) |
| 3 | LLM Observability | [llm-observability.md](llm-observability.md) |
| 4 | Safety & Guardrails | [safety-guardrails.md](safety-guardrails.md) |
| 5 | Reliability Patterns | [reliability-patterns.md](reliability-patterns.md) |
| 6 | Experimentation (A/B, Shadow, Canary) | [experimentation.md](experimentation.md) |
| 7 | Evaluation & Monitoring | [evaluation-monitoring.md](evaluation-monitoring.md) |
| 8 | CI/CD for ML | [cicd-for-ml.md](cicd-for-ml.md) |

## Learning Path

```
         Foundation                 Production quality            Release engineering
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

## The MLOps Mindset

```
Research ML:                Production ML:
────────────────────────────────────────────────────────
"Does the model work?"       "Does it work for users?"
Run once on a notebook       Runs 24/7 reliably
1 dataset, 1 result          Continuous data, drift
You validate manually        Automated evaluation pipelines
No one else uses it          10,000 requests per day
```

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
