# Phase 7: Portfolio Projects

> Reading the docs makes you fluent. **Shipping these makes you hireable.**

Interviewers care more about what you've built than what you've read. The behavioral / project-deep-dive round (round 4 in the [interview loop](../README.md#getting-hired-the-interview-loop)) is won here — when they ask *"how did you evaluate that?"*, you need a real golden set and a real regression to point to, not a paraphrase of [llms/evals.md](../llms/evals.md).

This folder holds **starter scaffolds** for the two highest-leverage projects. They're intentionally minimal — your job is to flesh them out, ship them, and be able to defend every design choice.

## Why these two first

```
RAG chatbot      → exercises Phases 4–6 end to end (embeddings, retrieval,
                   prompting, APIs, streaming). The canonical AI app.

Eval harness     → the differentiator. Most candidates can't show one.
                   Build it FOR the RAG bot and you have a story:
                   "here's the golden set, here's the regression it caught."
```

The other Phase 7 projects (fine-tuned model, full agent over MCP) build on the same muscles — do them after these two land.

## The projects

| Project | Folder | Proves you can... | Maps to |
|---------|--------|-------------------|---------|
| RAG chatbot | [rag-chatbot/](rag-chatbot/) | Build the canonical AI app end-to-end | [llms/rag.md](../llms/rag.md), [llms/llm-apis-sdks.md](../llms/llm-apis-sdks.md) |
| Eval harness | [eval-harness/](eval-harness/) | Prove quality and catch regressions | [llms/evals.md](../llms/evals.md) |

## How to work them

1. **Get the scaffold running first.** Make the smallest version work before adding features.
2. **Build the eval harness against the RAG bot.** They're a pair: the harness scores the bot.
3. **Write down one regression the eval caught.** "I changed chunk size and recall@3 dropped from 0.85 to 0.61" is an interview-winning sentence.
4. **Put it on GitHub with a real README** — architecture diagram, the metric you optimized, what you'd do at 100×. Treat the README like the system-design round on paper. See [llms/system-design.md](../llms/system-design.md).
5. **Use `uv`** for environments (see [python/README.md](../python/README.md)).

> 💡 **The interview story you're building:** "I built a RAG support bot, wrote a 40-example golden set with an LLM-as-judge groundedness check, ran it in CI, and caught a chunking regression before it shipped." That single sentence clears the domain *and* behavioral rounds.
