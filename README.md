# AI/ML Engineer Transition Plan

> Frontend Developer → AI Engineer

A self-paced curriculum that takes you from Python fundamentals to shipping production AI systems. Each phase has a dedicated folder with teaching-quality docs — analogies, code, gotchas, and self-check questions — plus a folder `README.md` that lays out the learning order.

## How to Use This Repo

```
ai-ml-studying/
├── README.md            ← you are here (the roadmap)
├── prompt-rules.md      ← the doc-generation prompt (one source of truth)
├── prompt-example.md    ← worked example of applying the prompt
│
├── ml/            → Phase 2: Classical ML
├── deep-learning/ → Phase 3: Deep Learning
├── nlp/           → Phase 4: NLP & Transformers
├── llms/          → Phase 5: LLMs & AI Engineering  (the core of the role)
└── ml-ops/        → Phase 6: MLOps & Production
```

1. Pick the phase you're on (below) and open that folder's `README.md` — it has a topic order and a learning path diagram.
2. Work each topic: read the doc, build the smallest runnable version (30–50 lines), answer the self-check questions.
3. From Phase 5 onward, **write an eval the same day you ship the feature.** "Did my change help?" has to be answerable.

> 💡 **For your track:** Phases 1–3 are foundations — learn enough to reason, not to ship. Phase 5 (LLMs & AI Engineering) is where you'll actually spend your career, so give it the most time.

## The Phases

```
  Phase 1       Phase 2        Phase 3         Phase 4          Phase 5           Phase 6         Phase 7
┌──────────┐  ┌──────────┐  ┌───────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────┐
│ Python & │→ │ Classical│→ │   Deep    │→ │     NLP     │→ │     LLMs     │→ │   MLOps    │→ │ Portfolio│
│   Math   │  │    ML    │  │  Learning │  │ & Trans-    │  │  & AI Eng.   │  │ & Produc-  │  │ Projects │
│          │  │(literacy)│  │(compressed│  │  formers    │  │  (the core)  │  │    tion    │  │          │
└──────────┘  └──────────┘  └───────────┘  └─────────────┘  └──────────────┘  └────────────┘  └──────────┘
  4-6 wks       3-4 wks       4-6 wks        4-6 wks          8-10 wks         4-6 wks         ongoing
                   │             │               │                 │                │
                   ▼             ▼               ▼                 ▼                ▼
                 ml/        deep-learning/     nlp/              llms/          ml-ops/
```

## Phase 1: Python & Math Foundations (4-6 weeks)

- **Python for ML** — syntax, NumPy, Pandas, Matplotlib
- **Linear Algebra** — vectors, matrices, transformations, eigenvalues
- **Calculus & Optimization** — derivatives, gradients, gradient descent
- **Probability & Statistics** — distributions, Bayes' theorem, hypothesis testing

> No folder yet — this phase is best learned from external courses (3Blue1Brown, fast.ai's prerequisite material). Come back here once Python + NumPy feel comfortable.

## Phase 2: Classical Machine Learning (3-4 weeks — literacy, not mastery) → [ml/](ml/)

> Aim to understand these well enough to reason about them — not to ship production classical-ML systems. You'll still use Random Forests and XGBoost as baselines, and the evaluation concepts (CV, precision/recall, bias-variance) apply to every model you ever ship.

- **Supervised Learning** — linear/logistic regression, decision trees, SVMs, random forests, gradient boosting
- **Unsupervised Learning** — clustering (K-means), dimensionality reduction (PCA)
- **Model Evaluation** — cross-validation, bias-variance tradeoff, metrics (precision, recall, F1, ROC/AUC)
- **Scikit-learn** — the unified `fit`/`predict`/`transform` API + Pipelines
- **Feature Engineering** — encoding, scaling, selection techniques

## Phase 3: Deep Learning (4-6 weeks — compressed for AI Engineer track) → [deep-learning/](deep-learning/)

- **Neural Network Fundamentals** — perceptrons, backpropagation, activation functions
- **Frameworks** — PyTorch (primary), TensorFlow/Keras (secondary)
- **CNNs** — image classification, transfer learning
- **RNNs & Sequence Models** — LSTMs/GRUs (**conceptual only** — skim, then move to Transformers)
- **Training Practices** — regularization, batch norm, learning rate scheduling
- **Transformers** — the architecture that powers everything in Phases 4 and 5

## Phase 4: NLP & Transformers (4-6 weeks) → [nlp/](nlp/)

- **Text Processing** — tokenization, embeddings (modern sentence embeddings matter more than Word2Vec/GloVe in 2026 — know the latter historically, use the former in practice)
- **Transformer Architecture** — attention, encoder vs. decoder vs. encoder-decoder
- **Hugging Face Ecosystem** — the "npm of AI" — where you get pretrained models
- **Text Classification, NER, Summarization** — practical NLP tasks you'll build

## Phase 5: LLMs & AI Engineering (8-10 weeks — the core of the role) → [llms/](llms/)

- **LLM Fundamentals** — GPT, Claude, Llama architectures, scaling laws, tokens, context windows
- **Prompt Engineering** — techniques, chain-of-thought, few-shot learning, system prompts
- **RAG (Retrieval-Augmented Generation)** — vector databases, embeddings, chunking strategies, hybrid search, reranking
- **LLM APIs & SDKs** — Anthropic SDK, OpenAI API, structured outputs / JSON mode, streaming
- **Production LLM patterns** — prompt caching, token & cost management, context window strategies, multi-turn state, retries & fallbacks
- **Agents & Tool Use** — function calling, multi-step reasoning, agent frameworks (LangChain/LangGraph, LlamaIndex, Claude Agent SDK)
- **MCP (Model Context Protocol)** — the standard wire format for tools — "USB-C for AI"
- **Evals (first-class skill)** — golden datasets, LLM-as-judge, prompt regression testing, offline vs. online eval
- **Fine-tuning** — LoRA, QLoRA, PEFT techniques (know when NOT to fine-tune — prompt + RAG usually wins)

## Phase 6: MLOps & Production (4-6 weeks) → [ml-ops/](ml-ops/)

- **Model Serving** — FastAPI, Docker, model optimization
- **Vector Databases** — Pinecone, ChromaDB, pgvector, Weaviate
- **LLM Observability** — Langfuse, Helicone, LangSmith — traces, cost & latency tracking, prompt versioning
- **Safety & Guardrails** — PII detection/redaction, prompt injection defense, jailbreak resistance, output validation
- **Reliability** — rate limiting, retries with backoff, provider fallbacks, caching layers (semantic + exact)
- **Experimentation** — A/B testing prompts and models, shadow deployments, canaries
- **Evaluation & Monitoring** — offline evals + online quality metrics, drift detection
- **CI/CD for ML** — experiment tracking (MLflow, W&B), prompt/model versioning

## Phase 7: Portfolio Projects

Interviewers care about shipped projects more than course certificates. Build 3–5 of these end-to-end.

1. **ML-powered web app** — leverage your frontend skills with an ML backend
2. **RAG chatbot** — end-to-end retrieval-augmented generation with hybrid search + reranking, citations in the UI
3. **Fine-tuned model** — domain-specific LLM adaptation (LoRA on a small open model)
4. **AI agent** — autonomous agent with tool use and multi-step reasoning over MCP
5. **Eval harness** — golden-dataset + LLM-as-judge eval pipeline for one of the projects above (**interviewers ask about this**)

---

## Your Advantage as a Frontend Developer

- Building AI-powered UIs and demos (streaming, chat interfaces, tool-call visualizations)
- Full-stack AI apps — you already own the frontend half
- Understanding user experience for AI products (latency, streaming, error states)
- JavaScript/TypeScript ML tooling (Vercel AI SDK, Transformers.js, TensorFlow.js)
- Fixing the painful dashboards that ML teams inherit — eval consoles, trace viewers, A/B UIs

## Key Resources

| Resource | Type | Why |
|----------|------|-----|
| [fast.ai](https://www.fast.ai/) | Free course (practical deep learning) | Top-down, code-first — fits your builder instinct |
| [Andrej Karpathy's YouTube](https://www.youtube.com/@AndrejKarpathy) | Neural nets from scratch | The clearest "how does this actually work" videos on the internet |
| [3Blue1Brown](https://www.3blue1brown.com/) | Visual math | Linear algebra and calculus intuitions that stick |
| [Hugging Face courses](https://huggingface.co/learn) | NLP & Transformers | The standard toolkit + hands-on fine-tuning |
| [DeepLearning.AI](https://www.deeplearning.ai/) | LLM specializations | Andrew Ng's short courses on prompt engineering, RAG, agents |
| [Simon Willison's blog](https://simonwillison.net/) | Practitioner blog | What's actually happening in LLM land, weekly |
| [Anthropic docs](https://docs.claude.com/) | Official docs | Reference for the Claude API, prompt caching, MCP, Agent SDK |
| [Kaggle](https://www.kaggle.com/) | Datasets & competitions | Sharpen classical ML instincts on real data |
