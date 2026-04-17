# AI/ML Engineer Transition Plan

> Frontend Developer → AI Engineer

## Phase 1: Python & Math Foundations (4-6 weeks)

- **Python for ML** — syntax, NumPy, Pandas, Matplotlib
- **Linear Algebra** — vectors, matrices, transformations, eigenvalues
- **Calculus & Optimization** — derivatives, gradients, gradient descent
- **Probability & Statistics** — distributions, Bayes' theorem, hypothesis testing

## Phase 2: Classical Machine Learning (3-4 weeks — literacy, not mastery)

> For an AI Engineer track, aim to understand these well enough to reason about them — not to ship production classical-ML systems.

- **Supervised Learning** — linear/logistic regression, decision trees, SVMs, random forests
- **Unsupervised Learning** — clustering (K-means), dimensionality reduction (PCA)
- **Model Evaluation** — cross-validation, bias-variance tradeoff, metrics (precision, recall, F1)
- **Scikit-learn** — hands-on projects with real datasets
- **Feature Engineering** — encoding, scaling, selection techniques

## Phase 3: Deep Learning (4-6 weeks — compressed for AI Engineer track)

- **Neural Network Fundamentals** — perceptrons, backpropagation, activation functions
- **Frameworks** — PyTorch (primary), TensorFlow/Keras (secondary)
- **CNNs** — image classification, transfer learning
- **RNNs & Sequence Models** — LSTMs/GRUs (conceptual only — skim, then move to Transformers)
- **Training Practices** — regularization, batch norm, learning rate scheduling

## Phase 4: NLP & Transformers (4-6 weeks)

- **Text Processing** — tokenization, embeddings (Word2Vec, GloVe)
- **Transformer Architecture** — attention mechanism, encoder-decoder
- **Hugging Face Ecosystem** — fine-tuning pretrained models
- **Text Classification, NER, Summarization** — practical NLP tasks

## Phase 5: LLMs & AI Engineering (8-10 weeks — the core of the role)

- **LLM Fundamentals** — GPT, Claude, Llama architectures, scaling laws
- **Prompt Engineering** — techniques, chain-of-thought, few-shot learning, system prompts
- **RAG (Retrieval-Augmented Generation)** — vector databases, embeddings, chunking strategies, hybrid search, reranking
- **LLM APIs & SDKs** — Anthropic SDK, OpenAI API, structured outputs / JSON mode, streaming
- **Production LLM patterns** — prompt caching, token & cost management, context window strategies, multi-turn state, retries & fallbacks
- **Agents & Tool Use** — function calling, multi-step reasoning, agent frameworks (LangChain/LangGraph, LlamaIndex, Claude Agent SDK), MCP (Model Context Protocol)
- **Evals (first-class skill)** — golden datasets, LLM-as-judge, prompt regression testing, offline vs. online eval
- **Fine-tuning** — LoRA, QLoRA, PEFT techniques (know when NOT to fine-tune — prompt+RAG usually wins)

## Phase 6: MLOps & Production (4-6 weeks)

- **Model Serving** — FastAPI, Docker, model optimization
- **Vector Databases** — Pinecone, ChromaDB, pgvector, Weaviate
- **LLM Observability** — Langfuse, Helicone, LangSmith — traces, cost & latency tracking, prompt versioning
- **Safety & Guardrails** — PII detection/redaction, prompt injection defense, jailbreak resistance, output validation
- **Reliability** — rate limiting, retries with backoff, provider fallbacks, caching layers (semantic + exact)
- **Experimentation** — A/B testing prompts and models, shadow deployments
- **CI/CD for ML** — experiment tracking (MLflow, W&B), prompt/model versioning

## Phase 7: Portfolio Projects

1. **ML-powered web app** — leverage your frontend skills with an ML backend
2. **RAG chatbot** — end-to-end retrieval-augmented generation system (with hybrid search + reranking)
3. **Fine-tuned model** — domain-specific LLM adaptation
4. **AI agent** — autonomous agent with tool use and multi-step reasoning
5. **Eval harness** — build a golden-dataset + LLM-as-judge eval pipeline for one of the projects above (interviewers ask about this)

---

## Your Advantage as a Frontend Developer

- Building AI-powered UIs and demos (streaming, chat interfaces)
- Full-stack AI apps — you already own the frontend half
- Understanding user experience for AI products
- JavaScript/TypeScript ML tooling (TensorFlow.js, Transformers.js, Vercel AI SDK)

## Key Resources

| Resource | Type |
|----------|------|
| fast.ai | Free course (practical deep learning) |
| Andrej Karpathy YouTube | Neural networks from scratch |
| 3Blue1Brown | Visual math/linear algebra |
| Hugging Face courses | NLP & Transformers |
| DeepLearning.AI | LLM specializations |
| Kaggle | Datasets & competitions |
