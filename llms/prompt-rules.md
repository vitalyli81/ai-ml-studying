# LLMs & AI Engineering — Doc Generation

> This folder uses the canonical doc prompt. **See [../prompt-rules.md](../prompt-rules.md)** for the full template and rules.

## How to use it

When generating a new doc in this folder, copy the prompt from `../prompt-rules.md` and set:

```
TOPIC: <one of the topics below, or any LLM/AI-Engineering subtopic>
```

Save the result as `<topic-slug>.md` in this folder, then link it from [README.md](README.md).

## Topics covered in this folder

- **LLM Fundamentals** — GPT, Claude, Llama architectures, scaling laws, tokens, context windows
- **Prompt Engineering** — techniques, chain-of-thought, few-shot, system prompts
- **RAG** — vector databases, embeddings, chunking, hybrid search, reranking
- **LLM APIs & SDKs** — Anthropic SDK, OpenAI API, structured outputs, streaming
- **Production LLM Patterns** — prompt caching, cost management, retries, fallbacks
- **Agents & Tool Use** — function calling, multi-step reasoning, agent frameworks
- **MCP (Model Context Protocol)** — standard wire format for tools
- **Evals** — golden datasets, LLM-as-judge, regression testing
- **Fine-tuning** — LoRA, QLoRA, PEFT (and when NOT to fine-tune)

See [README.md](README.md) for the recommended reading order.
