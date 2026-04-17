# Deep Learning — Doc Generation

> This folder uses the canonical doc prompt. **See [../prompt-rules.md](../prompt-rules.md)** for the full template and rules.

## How to use it

When generating a new doc in this folder, copy the prompt from `../prompt-rules.md` and set:

```
TOPIC: <one of the topics below, or any Deep Learning subtopic>
```

Save the result as `<topic-slug>.md` in this folder, then link it from [README.md](README.md).

## Topics covered in this folder

- **Neural Network Fundamentals** — perceptrons, backpropagation, activation functions
- **Frameworks** — PyTorch (primary), TensorFlow/Keras (secondary)
- **CNNs** — image classification, transfer learning
- **RNNs & Sequence Models** — LSTMs, GRUs (conceptual; Transformers replaced them)
- **Training Practices** — regularization, batch norm, learning rate scheduling, dropout, early stopping
- **Transformers** — attention mechanism, the architecture behind LLMs

See [README.md](README.md) for the recommended reading order.
