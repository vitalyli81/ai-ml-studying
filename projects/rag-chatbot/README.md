# RAG Chatbot — Starter Scaffold

A minimal, runnable RAG chatbot you grow into a portfolio piece. Answers questions from *your* documents, with citations. Backed by the concepts in [llms/rag.md](../../llms/rag.md).

## What you're building

```
docs/ ──► [chunk] ──► [embed] ──► ChromaDB        (index once)
                                      │
user question ──► [embed] ──► [search top-k] ──► [build prompt] ──► Claude ──► answer + sources
```

## Run it

```bash
# from this folder
uv venv && source .venv/bin/activate
uv pip install anthropic chromadb
export ANTHROPIC_API_KEY=sk-ant-...

# drop a few .md or .txt files into ./docs first, then:
uv run python rag.py "What is the refund policy?"
```

## The scaffold

[rag.py](rag.py) is ~80 lines: load docs → chunk → embed into Chroma → retrieve → generate with citations. It works, but it's deliberately naive. **Your job is to make it good** — and to be able to explain each upgrade in an interview.

## Grow it (the interview-worthy upgrades)

Each of these is a talking point. Do them in order and measure the effect with the [eval harness](../eval-harness/):

- [ ] **Chunking strategy** — start fixed-size, move to recursive; measure recall@k before/after
- [ ] **Hybrid search** — add BM25 keyword search alongside vectors (users paste exact terms *and* paraphrase)
- [ ] **Reranking** — cross-encoder pass over top-k → top-3; measure the precision lift
- [ ] **Citations in the response** — return which source each claim came from
- [ ] **"I don't know" fallback** — set a similarity floor; don't hallucinate on out-of-corpus queries
- [ ] **Streaming** — stream tokens to the client (perceived latency = time-to-first-token)
- [ ] **Metadata filtering** — restrict retrieval by source/department
- [ ] **A simple UI** — your frontend edge: a chat interface with streaming + visible citations

## Defend your choices

Before calling it done, make sure you can answer:
- Why this chunk size / overlap / top_k? (point to eval numbers, not vibes)
- What happens when retrieval returns nothing relevant?
- Where would this break at 10× traffic? (see [llms/system-design.md](../../llms/system-design.md))
- How did you know an upgrade helped? (the eval harness)
