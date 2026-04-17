# MLOps & Production — Doc Generation

> This folder uses the canonical doc prompt. **See [../prompt-rules.md](../prompt-rules.md)** for the full template and rules.

## How to use it

When generating a new doc in this folder, copy the prompt from `../prompt-rules.md` and set:

```
TOPIC: <one of the topics below, or any MLOps/Production subtopic>
```

Save the result as `<topic-slug>.md` in this folder, then link it from [README.md](README.md).

## Topics covered in this folder

- **Model Serving** — FastAPI, Docker, model optimization
- **Vector Databases** — Pinecone, ChromaDB, pgvector, Weaviate
- **LLM Observability** — Langfuse, Helicone, LangSmith — traces, cost, latency, prompt versioning
- **Safety & Guardrails** — PII detection/redaction, prompt-injection defense, output validation
- **Reliability Patterns** — retries, backoff, provider fallbacks, caching (exact + semantic), circuit breakers
- **Experimentation** — A/B tests, shadow deploys, canaries
- **Evaluation & Monitoring** — offline evals, online quality metrics, drift detection
- **CI/CD for ML** — experiment tracking (MLflow, W&B), prompt/model versioning

See [README.md](README.md) for the recommended reading order.
