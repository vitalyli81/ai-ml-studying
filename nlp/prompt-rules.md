# NLP & Transformers — Doc Generation

> This folder uses the canonical doc prompt. **See [../prompt-rules.md](../prompt-rules.md)** for the full template and rules.

## How to use it

When generating a new doc in this folder, copy the prompt from `../prompt-rules.md` and set:

```
TOPIC: <one of the topics below, or any NLP/Transformers subtopic>
```

Save the result as `<topic-slug>.md` in this folder, then link it from [README.md](README.md).

## Topics covered in this folder

- **Text Processing** — tokenization, embeddings (modern sentence embeddings + historical Word2Vec/GloVe)
- **Transformer Architecture** — attention, encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5)
- **Hugging Face Ecosystem** — loading, fine-tuning, and sharing pretrained models
- **Core NLP Tasks** — text classification, NER, seq2seq summarization/translation
- **Fine-tuning for NLP** — adapting pretrained models to your data

See [README.md](README.md) for the recommended reading order.
