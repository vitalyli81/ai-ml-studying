# Evaluation & Monitoring

## TL;DR

Evaluation answers "is my AI system good?" before you ship. Monitoring answers "is it still good?" after you ship. Without both, you're flying blind — the model could be hallucinating, drifting, getting more expensive, or breaking silently and you'd never know. As an AI engineer, this is your safety net: it catches problems before users do.

> 💡 **Key Insight:** LLMs are probabilistic — they don't fail with error codes, they fail by giving wrong answers that look right. Traditional "is the server up?" monitoring is not enough. You need to monitor what the model *says*.

---

## The Mental Model

Think of it like **quality control in a factory** combined with **smoke alarms in a building**.

- Evaluation = Quality control before shipping: you inspect every product before it leaves the factory
- Monitoring = Smoke alarms after shipping: continuous sensors that alert you the moment something goes wrong

Mapping:
- Factory inspection (pre-ship) → Offline evaluation on a test set
- Smoke alarm (post-ship) → Online monitoring of live traffic
- Defective product rate → Failure rate / hallucination rate
- Fire alarm trigger → Alert when metric drops below threshold

You wouldn't ship electronics without testing them. You wouldn't run a building without smoke alarms. You shouldn't deploy AI without both evaluation and monitoring.

---

## Why It Exists

### The Problem

LLMs are black boxes that produce text. Unlike a `500 Internal Server Error`, a hallucination looks like a perfectly normal response. You can't tell from HTTP status codes that your model is wrong.

```
Traditional API monitoring:
  ✅ Is the server responding? (HTTP 200)
  ✅ Is it fast? (latency < 500ms)
  ❌ Is the answer CORRECT? (impossible to tell from status codes)
  ❌ Did it make something up? (no way to know without checking content)
  ❌ Is it staying on topic? (not visible in logs)
```

### The Solution

Treat the model's *output quality* as a first-class metric, just like latency and error rate.

```
LLM monitoring adds:
  ✅ Is the answer accurate? (LLM-as-judge or human review)
  ✅ Is it safe? (toxicity detection)
  ✅ Is it relevant? (semantic similarity to expected answer)
  ✅ Is it getting more expensive over time? (token usage trends)
  ✅ Is output quality drifting? (baseline comparison)
```

### What Changed

Specialized tools — LangSmith, Weights & Biases, Arize, Evidently — now give you dashboards for AI quality, not just infrastructure health.

---

## Core Concepts

### 1. Offline vs Online Evaluation

**One-line definition:** Offline = test before shipping; Online = monitor after shipping.

**Analogy:** Offline evaluation is a job interview (you assess the candidate before hiring). Online monitoring is a performance review (you assess after they're working). Both are necessary — you wouldn't skip either.

```
Offline Evaluation:
  When: Before deployment, after every model change
  Input: Curated test dataset (questions + expected answers)
  Output: Accuracy, F1, ROUGE, hallucination rate scores
  Goal:  "Is this version good enough to ship?"

Online Monitoring:
  When: Continuously, 24/7 in production
  Input: Real user traffic (sampled)
  Output: Live dashboards, alerts, trend graphs
  Goal:  "Is the deployed version still working correctly?"
```

**Common misconception:** "I evaluated it before shipping so I'm done." Wrong — data drifts, user behavior changes, edge cases emerge. Online monitoring is not optional.

---

### 2. Evaluation Metrics for LLMs

**One-line definition:** Quantitative scores that measure how good the model's outputs are on different dimensions.

**Analogy:** Like code test coverage — a number that gives you confidence without requiring you to manually read every output.

```
The main LLM metrics:

┌─────────────────────────────────────────────────────────────────────┐
│ METRIC          │ WHAT IT MEASURES              │ HOW              │
├─────────────────────────────────────────────────────────────────────┤
│ Exact Match     │ Is the answer word-for-word    │ string compare   │
│                 │ correct?                        │                  │
├─────────────────────────────────────────────────────────────────────┤
│ ROUGE-L         │ How much word overlap between  │ longest common   │
│                 │ output and reference?           │ subsequence      │
├─────────────────────────────────────────────────────────────────────┤
│ BERTScore       │ Semantic similarity (not just  │ embedding         │
│                 │ word overlap)                  │ cosine sim       │
├─────────────────────────────────────────────────────────────────────┤
│ LLM-as-Judge    │ Quality rating by another LLM  │ GPT-4/Claude     │
│                 │ (correctness, helpfulness)      │ grades the output│
├─────────────────────────────────────────────────────────────────────┤
│ Hallucination   │ Does output contain facts not  │ grounding check  │
│ Rate            │ in the context?                │ or LLM judge     │
├─────────────────────────────────────────────────────────────────────┤
│ Toxicity        │ Does output contain harmful    │ classifier model │
│                 │ content?                        │                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Common misconception:** ROUGE is the gold standard. It was designed for summarization. For open-ended generation, LLM-as-judge is far more meaningful.

---

### 3. LLM-as-Judge

**One-line definition:** Use a powerful LLM (GPT-4, Claude Opus) to evaluate another LLM's outputs — scoring accuracy, helpfulness, groundedness, and safety.

**Analogy:** It's like using a senior engineer to do code reviews. You couldn't automate the review fully (too nuanced), but you trust an expert to judge quality consistently.

```python
# LLM-as-judge example
JUDGE_PROMPT = """You are evaluating an AI assistant's response.

Question: {question}
Context provided to the AI: {context}
AI's response: {response}

Score the response on these dimensions (1-5 each):
1. Accuracy: Is the information correct based on the context?
2. Groundedness: Is every claim supported by the provided context?
3. Helpfulness: Does it fully answer the question?
4. Conciseness: Is it appropriately concise without losing information?

Return JSON: {{"accuracy": X, "groundedness": X, "helpfulness": X, 
               "conciseness": X, "overall": X, "issues": ["..."]}}"""

# Run for every sample in your test set
scores = []
for sample in test_set:
    judgment = judge_llm.generate(JUDGE_PROMPT.format(**sample))
    scores.append(json.loads(judgment))

avg_accuracy = sum(s["accuracy"] for s in scores) / len(scores)
print(f"Average accuracy: {avg_accuracy:.2f}/5")
```

**Common misconception:** LLM-as-judge is subjective and unreliable. Studies show GPT-4 and Claude agree with human evaluators ~80-90% of the time on most dimensions — comparable to human inter-rater agreement.

---

### 4. RAG-Specific Evaluation (RAGAS)

**One-line definition:** RAGAS (Retrieval Augmented Generation Assessment) is a framework specifically for evaluating RAG pipelines on both retrieval quality AND generation quality.

**Analogy:** For a library research assistant, you'd check two things: Did they find the right books? Did they summarize those books accurately? RAGAS does the same for RAG.

```
RAGAS evaluates 4 dimensions:

1. Context Precision
   "Were the retrieved documents actually relevant?"
   Retrieved 5 docs, 3 were relevant → precision = 60%

2. Context Recall
   "Did retrieval find ALL the information needed to answer?"
   Answer needed 4 facts, retrieval found 3 → recall = 75%

3. Faithfulness
   "Does the answer stick to the retrieved context (no hallucinations)?"
   Answer has 10 claims, 9 are in context → faithfulness = 90%

4. Answer Relevancy
   "Does the answer actually address the question?"
   0.0 (off-topic) to 1.0 (perfectly on-topic)
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset,  # your test questions + retrieved contexts + LLM answers
    metrics=[faithfulness, answer_relevancy, context_precision]
)
print(results)
# {'faithfulness': 0.85, 'answer_relevancy': 0.91, 'context_precision': 0.78}
```

---

### 5. Production Monitoring

**One-line definition:** Continuous, automated measurement of your deployed AI system's health, quality, and behavior.

**Analogy:** It's like your car's dashboard — you don't manually check engine temperature every minute; sensors do it automatically and alert you when something's wrong.

```
What to monitor in production:

Infrastructure metrics (standard):
  ✅ Latency (p50, p95, p99 response time)
  ✅ Error rate (HTTP 4xx, 5xx)
  ✅ Throughput (requests per second)
  ✅ Cost (tokens used × price per token)

LLM-specific metrics (the new stuff):
  ✅ Output quality score (sampled LLM-as-judge on 5-10% of traffic)
  ✅ Hallucination rate (grounding check on sampled outputs)
  ✅ Refusal rate (how often does the model refuse to answer?)
  ✅ Token usage per request (trending up = cost problem)
  ✅ Retrieval quality (for RAG: are retrieved docs relevant?)
  ✅ User feedback signals (thumbs up/down, corrections)
```

---

## How It Actually Works (Step-by-Step)

### Building an Evaluation Pipeline

```
Step 1: Create a golden test set
        50-200 questions with expected answers (hand-written or curated)
        Cover happy path, edge cases, tricky questions

Step 2: Run your model on every test question
        Store: question, retrieved_context, model_answer, expected_answer

Step 3: Score with multiple metrics
        - Exact match (for factual Q&A)
        - LLM-as-judge (for quality dimensions)
        - RAGAS metrics (for RAG systems)
        - Hallucination detection (for any LLM system)

Step 4: Set thresholds
        faithfulness > 0.85, answer_relevancy > 0.80
        If any metric drops below threshold → block deployment

Step 5: Add to CI/CD pipeline
        Every time you change the model, prompt, or retrieval:
        → Auto-run evaluation suite
        → Must pass all thresholds before deploy
        → Compare scores to previous version (must not regress)

Step 6: Monitor production
        Sample 5-10% of live traffic
        Run async evaluation on sampled outputs
        Alert if rolling 24h average drops below threshold
```

---

## Code in Practice

### 1. Basic Evaluation with LLM-as-Judge

```python
import anthropic
import json

client = anthropic.Anthropic()

def evaluate_response(question: str, context: str, response: str) -> dict:
    """Score an LLM response using Claude as judge."""
    
    prompt = f"""Evaluate this AI assistant response. Return ONLY valid JSON.

Question: {question}
Context given to AI: {context}
AI Response: {response}

Score each 1-5:
- accuracy: Is the answer factually correct based on the context?
- groundedness: Every claim is supported by context (no hallucinations)?
- helpfulness: Does it fully answer the question?

JSON format: {{"accuracy": N, "groundedness": N, "helpfulness": N, "verdict": "pass|fail"}}
verdict is "fail" if any score < 3."""

    result = client.messages.create(
        model="claude-sonnet-4-5-20241022",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(result.content[0].text)

# Evaluate a batch
test_cases = [
    {
        "question": "What is the refund window?",
        "context": "Customers may return items within 30 days of purchase.",
        "response": "You can return items within 30 days."
    },
    # ... more test cases
]

results = [evaluate_response(**tc) for tc in test_cases]
pass_rate = sum(1 for r in results if r["verdict"] == "pass") / len(results)
print(f"Pass rate: {pass_rate:.0%}")
```

### 2. RAGAS Evaluation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Your RAG system's outputs on the test set
data = {
    "question": ["What is the refund policy?", "How do I reset my password?"],
    "answer": ["You can return items within 30 days.", "Go to Settings > Security."],
    "contexts": [
        ["Items may be returned within 30 days of purchase with receipt."],
        ["To reset your password, navigate to Settings, then Security tab."]
    ],
    "ground_truth": ["Items can be returned within 30 days.", "Use Settings > Security to reset password."]
}

dataset = Dataset.from_dict(data)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

print(result)
# {'faithfulness': 0.9500, 'answer_relevancy': 0.9200, 
#  'context_precision': 0.8750, 'context_recall': 0.8333}
```

### 3. Production Monitoring with LangSmith

```python
from langsmith import Client
from langsmith.wrappers import wrap_anthropic
import anthropic

# LangSmith traces every LLM call automatically
langsmith_client = Client()
anthropic_client = wrap_anthropic(anthropic.Anthropic())  # Wrapped client

def answer_question(question: str, context: str) -> str:
    # This call is automatically traced in LangSmith dashboard
    # You see: inputs, outputs, latency, tokens, cost, errors
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-5-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}"
        }]
    )
    return response.content[0].text

# Add user feedback after the fact
langsmith_client.create_feedback(
    run_id="run-id-from-trace",
    key="user_rating",
    score=1.0,   # thumbs up
    comment="Correct and helpful"
)
```

---

## Gotchas & Pitfalls

```
❌ Evaluating only on your training distribution
   Test set looks like training data → inflated scores
   Deploy → real users ask different questions → model fails
✅ Include adversarial, edge case, and out-of-distribution examples

❌ Using ROUGE as your only metric
   "The cat sat." and "A feline was seated." score near-zero on ROUGE
   But they're semantically identical
✅ Use BERTScore or LLM-as-judge for semantic similarity

❌ Evaluating too infrequently
   "We evaluated it 3 months ago when we shipped"
   Data drifts, prompts change, models get updated
✅ Run evaluation on every model/prompt change in CI/CD

❌ No production sampling strategy
   Evaluating 100% of production traffic is expensive
   Evaluating 0% means flying blind
✅ Sample 5-10% of traffic for async quality evaluation

❌ Ignoring cost monitoring
   Token usage creeps up silently over months
   Suddenly your AI feature costs 5× what you budgeted
✅ Track cost per request, set alerts at 2× baseline

❌ Only monitoring the LLM, not the whole pipeline
   Retrieval can silently degrade (wrong docs returned)
   Model output quality looks fine but answers are wrong
✅ Monitor retrieval quality separately from generation quality
```

---

## When to Use / When NOT to Use

### Always Evaluate When:
- Before shipping any new model version or prompt change
- After updating the retrieval system in a RAG pipeline
- When switching between model providers or versions
- When adding new data to your knowledge base

### Always Monitor When:
- Any AI system is in production serving real users
- Your LLM costs are significant enough to care about
- The AI is making decisions that affect users (recommendations, support)
- You have SLAs or quality commitments to customers

### Can Skip/Simplify When:
- Pure research/experimentation (no users affected)
- One-off data processing tasks (not ongoing)
- The downstream consequence of errors is very low

---

## Related Concepts

| Concept | Connection |
|---------|------------|
| RAG | RAGAS specifically evaluates RAG retrieval + generation quality |
| CI/CD for ML | Evaluation gates are what CI/CD pipelines check before deployment |
| Model Serving | Monitoring is instrumented at the serving layer |
| Fine-tuning | You need evaluation to know if fine-tuning actually improved things |

---

## Cheat Sheet

```
Evaluation metrics:
  Exact Match    → factual Q&A with one correct answer
  ROUGE-L        → summarization (word overlap)
  BERTScore      → open-ended generation (semantic similarity)
  LLM-as-Judge   → quality dimensions (accuracy, helpfulness, safety)
  RAGAS          → RAG-specific (faithfulness, relevancy, precision)

What to monitor in prod:
  Infrastructure: latency, error rate, throughput, cost
  LLM-specific:   quality score, hallucination rate, refusal rate, token usage

Tools:
  LangSmith      → tracing + evaluation + monitoring (best-in-class)
  Weights & Biases → experiment tracking + evaluation
  RAGAS          → RAG evaluation framework (open source)
  Arize AI       → ML observability platform

Remember:
  1. Evaluation = before ship, Monitoring = after ship — both required
  2. LLM-as-judge correlates ~85-90% with human ratings — trust it
  3. Sample 5-10% of production traffic for async quality checks
```

---

## Self-Check Questions

<details>
<summary>Click to reveal answers</summary>

**Q1: Why can't you just monitor HTTP error rates for LLM systems?**
LLMs fail silently — they return HTTP 200 with a plausible-sounding but incorrect answer. Error rates only catch infrastructure failures. Quality monitoring catches model failures.

**Q2: What's the difference between faithfulness and accuracy in RAGAS?**
Faithfulness measures whether the answer is grounded in the retrieved context (no hallucinations). Accuracy measures whether the answer is factually correct. A model can be faithful (only uses context) but inaccurate (the context itself was wrong).

**Q3: Why use LLM-as-judge instead of human evaluation?**
Scale and cost. Human evaluation is slow and expensive — you might evaluate 100 samples. LLM-as-judge runs in seconds and costs cents, so you can evaluate thousands of samples automatically in CI/CD.

**Q4: What should trigger a production alert?**
When a rolling metric (e.g., 24-hour average faithfulness score) drops below a predefined threshold (e.g., 0.85). Also: sudden spike in latency, error rate, cost per request, or refusal rate.

**Q5: What's the minimum test set size for meaningful evaluation?**
At minimum 50-100 examples to get statistically meaningful averages. For production-critical systems, aim for 200-500, covering happy path, edge cases, and adversarial inputs. More is better, but quality > quantity.

</details>

---

## Go Deeper

| Resource | Why It's Worth Your Time |
|----------|--------------------------|
| [RAGAS Documentation](https://docs.ragas.io) | The definitive guide to evaluating RAG systems. Start with the "get started" guide — you'll have evaluation running in 30 minutes. |
| [LangSmith Docs](https://docs.smith.langchain.com) | Best end-to-end platform for LLM observability. The tracing tutorial is particularly good for understanding what to log. |
| [LMSYS Chatbot Arena](https://chat.lmsys.org) | See how real humans evaluate LLMs side-by-side — great intuition for what "good" looks like. |
| *Evaluating LLMs* by Hugging Face | Free course covering offline and online evaluation strategies. Very practical. |
| [Evidently AI Docs](https://docs.evidentlyai.com) | Open-source ML monitoring — excellent for understanding drift detection and monitoring dashboards. |
