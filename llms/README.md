# Phase 5: LLMs & AI Engineering

> From understanding AI models to building real AI-powered products.

This is where your frontend skills become a superpower. You already know how to build UIs, call APIs, and ship products — now you'll learn the AI backend that powers them.

## Topics

| # | Topic | File | Why it matters |
|---|-------|------|----------------|
| 1 | LLM Fundamentals | [llm-fundamentals.md](llm-fundamentals.md) | The mental model — tokens, context, temperature, training |
| 2 | Prompt Engineering | [prompt-engineering.md](prompt-engineering.md) | How you actually control model behavior |
| 3 | RAG (Retrieval-Augmented Generation) | [rag.md](rag.md) | Give the model *your* knowledge |
| 4 | LLM APIs & SDKs | [llm-apis-sdks.md](llm-apis-sdks.md) | Wire models into your app (Anthropic, OpenAI, Vercel AI SDK) |
| 5 | Production LLM Patterns | [production-llm-patterns.md](production-llm-patterns.md) | Caching, streaming, retries, cost & context management |
| 6 | Agents & Tool Use | [agents-tool-use.md](agents-tool-use.md) | Let the model take actions, not just talk |
| 7 | MCP (Model Context Protocol) | [mcp.md](mcp.md) | The standard wire format for tools — "USB-C for AI" |
| 8 | Evals | [evals.md](evals.md) | The highest-leverage skill — prove quality, catch regressions |
| 9 | Fine-tuning (LoRA, QLoRA, PEFT) | [fine-tuning-llms.md](fine-tuning-llms.md) | Customize the model itself — last resort, not first |

## Learning Path

```
Fundamentals ─► Prompting ─► RAG ─► APIs/SDKs ─► Production Patterns
   (theory)    (controlling   (add your   (wire it      (cache, stream,
                the model)    knowledge)  up in code)    retry, cost)
                                                │
                                                ▼
                                          Agents ─► MCP
                                          (actions) (standard tools)
                                                │
                                                ▼
                                           Evals ─► Fine-tuning
                                         (measure) (if prompting
                                                    isn't enough)
```

**Read this order** — each topic assumes the previous ones. Evals sits late but you should start writing them as soon as you ship anything; the doc is just placed where it makes sense pedagogically.

## What You'll Be Able to Build After This Phase

- AI chatbots with streaming responses and prompt caching
- RAG systems that answer questions from your own documents, with citations
- AI agents that can browse the web, write code, and call external services over MCP
- Eval harnesses that gate deploys and catch regressions before users do
- Custom fine-tuned models for narrow, high-volume tasks
- Full-stack AI applications with React/Next.js frontends

## How to Study This Phase

1. **Read the doc.** Each file is self-contained with analogies, code, gotchas, and self-check questions.
2. **Build the smallest version.** After each topic, write 30–50 lines that actually run. A streaming chat, a 10-doc RAG, a 3-tool agent, a 10-example eval set.
3. **Write the evals first.** From topic 4 onward, every project gets a tiny golden set. "Did my change help?" has to be answerable.
4. **Follow the "Go Deeper" links selectively.** The papers matter less than the practitioner blogs — read Simon Willison, Hamel Husain, and the Anthropic docs before the arXiv PDFs.

> 💡 **Key Insight for your track:** As a frontend engineer moving into AI, your differentiator is the full stack — you can ship a complete AI feature from UI to model to evals. Most ML specialists can't. Lean into it.
