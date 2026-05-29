# Eval Harness — Starter Scaffold

The project most candidates **can't** show — which is exactly why it wins interviews. A golden dataset + scorer that grades the [RAG chatbot](../rag-chatbot/) so "did my change help?" becomes a number, not a guess. Backed by [llms/evals.md](../../llms/evals.md).

## What you're building

```
golden_set.jsonl ──► [run pipeline on each input] ──► [score each output] ──► report
   (questions +         (call rag.ask)                  - keyword checks
    expectations)                                       - LLM-as-judge groundedness
                                                        → pass rate, recall@k
```

## Run it

```bash
# from this folder
uv venv && source .venv/bin/activate
uv pip install anthropic chromadb
export ANTHROPIC_API_KEY=sk-ant-...

uv run python eval.py
```

By default it evaluates the sibling [rag-chatbot](../rag-chatbot/). Point it at your own pipeline by editing the import in [eval.py](eval.py).

## The scaffold

- [golden_set.jsonl](golden_set.jsonl) — a handful of seed examples (question + expected behavior). **Grow this to 20–50** from real/imagined user queries, edge cases, and adversarial inputs.
- [eval.py](eval.py) — runs each example through the pipeline and scores it two ways:
  1. **Deterministic checks** — does the answer contain expected keywords? Does it correctly refuse when it should?
  2. **LLM-as-judge groundedness** — is every claim supported by the retrieved context? (catches hallucination)

## Grow it (the interview-worthy upgrades)

- [ ] **Expand the golden set to 40+** with tags (`easy`, `edge_case`, `adversarial`, `out_of_corpus`)
- [ ] **Add retrieval metrics** — recall@k against known-correct chunk IDs (debug retrieval *separately* from generation)
- [ ] **Wire it into CI** — fail the build if pass rate drops below threshold (the eval gate)
- [ ] **Track results over time** — append each run's scores to a CSV; plot regressions
- [ ] **Per-tag breakdown** — overall 85% can hide 40% on adversarial inputs

## The interview payoff

The whole point is one sentence you can say with conviction:

> "I changed the chunk size and my eval caught recall@3 dropping from 0.85 to 0.61, so I reverted it before it shipped."

That sentence proves you can *measure* quality and *catch regressions* — the highest-leverage skill for an AI engineer, and the thing the behavioral round is fishing for.
