# RAG (Retrieval-Augmented Generation)

## TL;DR

RAG gives LLMs access to external knowledge by fetching relevant documents at query time and including them in the prompt. Instead of relying on what the model memorized during training, you search your own data store and hand the relevant pieces to the model. The result: LLMs that answer questions about *your* documents, stay current, don't hallucinate from ignorance, and cite their sources.

> 💡 **Key Insight:** RAG separates *knowing* (your database) from *reasoning* (the LLM). You can update the database without touching the model.

---

## The Mental Model

**Think of RAG like a lawyer preparing for a case.**

A great lawyer doesn't memorize every law ever written. They have a research team that finds the relevant statutes and case precedents, then the lawyer reads those documents and argues the case based on them.

| Real world | Technical concept |
|------------|------------------|
| Lawyer's question to research team | User query |
| Research team searches law library | Vector database similarity search |
| Relevant case files found | Retrieved document chunks |
| Lawyer reads the files | Retrieved context inserted into prompt |
| Lawyer's argument | LLM's generated response |
| "According to statute 42B..." | Source citations in response |

Without RAG: the LLM is a lawyer arguing from memory. With RAG: they're arguing from the actual documents.

---

## Why It Exists (Problem → Solution)

**The problem:** LLMs are frozen in time. They know the world as of their training cutoff and know nothing about your company's internal docs, your product's API, this week's news, or your customer's account history.

**What came before:** People tried fine-tuning models on their data. This is expensive, slow to update, and teaches *behavior* rather than *facts*. If a refund policy changes, you'd have to retrain the model.

**What changed:** RAG separates knowledge from reasoning. You store knowledge in a searchable database. The LLM provides reasoning. Update the database in seconds — no model retraining.

| Problem | Without RAG | With RAG |
|---------|------------|----------|
| Knowledge cutoff | "I only know up to my training date" | Fetches current information |
| Hallucinations | Makes up plausible-sounding answers | Answers grounded in real docs |
| Private data | Can't access your company's docs | Searches your internal knowledge base |
| Source attribution | "I think..." (no source) | "According to refund-policy.md..." |
| Updates | Retrain model ($$$) | Update document store (free) |

---

## Core Concepts

### 1. Embeddings — Turning Text Into Numbers

**Plain English:** An embedding is a list of numbers that represents the *meaning* of a piece of text. Similar meanings produce similar lists of numbers.

**Analogy:** Imagine meaning as a location in a huge multidimensional city. "How do I return a product?" and "What's the refund process?" are two different addresses but in the same neighborhood. "What's the weather today?" is in a completely different borough. Embeddings give everything an address — and searching by meaning means finding nearby addresses.

```
"How do I return a product?"    → [0.12, -0.45, 0.78, 0.33, ...]  (1536 numbers)
"What's the refund process?"    → [0.14, -0.42, 0.75, 0.31, ...]  ← very similar!
"What's the weather today?"     → [-0.67, 0.23, -0.11, 0.89, ...] ← very different
```

```javascript
// Embeddings are like a hash function for meaning

// Regular hash: same input → same hash (exact match)
hash("hello") === hash("hello")  // true
hash("hello") !== hash("hi")     // true (different strings)

// Embedding: similar meaning → similar vectors (semantic match)
similarity(embed("return a product"), embed("refund process"))  // 0.95 (very close)
similarity(embed("return a product"), embed("weather today"))   // 0.12 (very far)

// You can search by MEANING, not keywords — this is the magic.
```

**Popular embedding models:**
```
Model                           Dimensions   Cost          Quality
────────────────────────────────────────────────────────────────
OpenAI text-embedding-3-small   1536         ~$0.02/M tok  Good
OpenAI text-embedding-3-large   3072         ~$0.13/M tok  Better
Cohere embed-v3                 1024         Free tier     Great
BGE / E5 (open source)          768–1024     Free          Good
Voyage AI voyage-3              1024         ~$0.06/M tok  Great for code
```

**Common misconception:** Better embeddings always mean better RAG. In practice, chunking strategy and retrieval quality matter as much as the embedding model. Start simple.

---

### 2. Vector Databases — Searching by Meaning

**Plain English:** A vector database stores embeddings and finds the most similar ones to a query — fast, even with millions of documents.

**Analogy:** A regular database is like a filing cabinet with alphabetical labels — great for exact matches. A vector database is like a library with a brilliant librarian who can say "you want this book, and these 4 other books are conceptually similar, even though they're in different sections."

```javascript
// Regular database (keyword search):
SELECT * FROM docs WHERE content LIKE '%refund%'
// → Only finds docs containing the word "refund"

// Vector database (semantic search):
SELECT * FROM docs ORDER BY cosine_similarity(embedding, query_embedding) LIMIT 5
// → Finds docs about "returns", "money back", "reimbursement" — same concept, different words
```

**Popular vector databases:**
```
Database     Type              Best For
──────────────────────────────────────────────────────
Pinecone     Managed cloud     Production, zero ops overhead
ChromaDB     Local/embedded    Prototyping, small projects
pgvector     PostgreSQL ext    Already using Postgres
Weaviate     Self-hosted       Full-featured, hybrid search
Qdrant       Self-hosted       Performance-critical, Rust-based
FAISS        Library (Meta)    Research, in-memory, no server
```

**Common misconception:** You need a dedicated vector database. For many projects, `pgvector` (adding vector search to your existing Postgres) is the right move — less infrastructure to manage.

---

### 3. Chunking — Splitting Documents for Search

**Plain English:** You can't embed an entire 50-page PDF as one vector — it would be too vague. You split documents into smaller pieces (chunks) so each chunk is specific and searchable.

**Analogy:** Index cards in a library. Instead of one card per book, you write one card per key concept in the book. When someone asks a specific question, they find the right card — not the whole book.

```
Raw document (50 pages)
       │
       ▼  chunk
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ "Returns within 30   │   │ "Digital products are │   │ "Contact support for  │
│  days with receipt"  │   │  non-refundable..."   │   │  policy exceptions..."│
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
       │                          │                          │
       ▼  embed                   ▼  embed                   ▼  embed
  [0.12, -0.45, ...]         [-0.22, 0.91, ...]         [0.55, -0.31, ...]
```

**Chunking strategies:**
```python
# 1. Fixed-size chunks (simplest — start here)
chunks = split_text(document, chunk_size=500, overlap=50)

# 2. Sentence-based (more natural)
chunks = split_by_sentences(document, max_sentences=5)

# 3. Recursive character splitting (most common in practice)
# Tries to split at: paragraphs → sentences → words → characters
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(document)
```

**Chunk size tradeoffs:**
```
Size            Precision   Context     Use for
──────────────────────────────────────────────────────
100–200 tokens  Very high   Very low    FAQ, specific facts
300–500 tokens  High        Good        Most use cases (start here)
500–1000 tokens Medium      High        Complex topics needing context
1000+ tokens    Low         Very high   Legal/technical docs
```

**Why overlap matters:**
```
Without overlap (chunk boundary):
  Chunk 1: "...returns within 30 days"     ← answer is split
  Chunk 2: "with receipt. Digital..."      ← across two chunks

With 50-token overlap:
  Chunk 1: "...returns within 30 days with receipt. Digital..."
  Chunk 2: "...30 days with receipt. Digital products are..."
  → Both chunks capture the boundary, neither loses context
```

**Common misconception:** Smaller chunks are always better for precision. Too-small chunks lose the surrounding context needed to answer correctly. Start with 300–500 tokens.

---

### 4. Retrieval — Finding What Matters

**Plain English:** At query time, embed the user's question and find the most similar chunks. Return the top K results.

**Analogy:** Like Spotify's recommendation engine — you describe what you want, it finds the most similar items in its database. Except instead of songs, you're finding document chunks.

```python
query = "Can I return a digital product?"
query_vector = embed(query)  # same embedding model as documents

results = vector_db.search(query_vector, top_k=3)
# Returns:
# [
#   { text: "Digital products are non-refundable...", score: 0.92 },
#   { text: "Returns within 30 days with receipt...", score: 0.78 },
#   { text: "Contact support for exceptions...", score: 0.71 },
# ]
```

**How "similar" is measured — cosine similarity:**

```
Two vectors' similarity = cosine of the angle between them
                        = (A · B) / (|A| × |B|)    ← normalized dot product

Range: -1 (opposite) ... 0 (unrelated) ... 1 (identical direction)

Why cosine, not Euclidean distance?
  Cosine ignores magnitude — only direction matters.
  An embedding's "direction" encodes meaning; its length is mostly noise.
  Two texts of different lengths about the same topic → same direction, different magnitudes.
  Cosine treats them as similar. Euclidean wouldn't.

Most embedding models output L2-normalized vectors, so cosine ≡ dot product.
```

**Choosing top_k — the precision/recall tradeoff:**

```
top_k = 1   → highest precision, but one wrong retrieval = wrong answer
top_k = 3   → good default; the LLM can synthesize across a few chunks
top_k = 10  → better recall, but adds noise and cost (each chunk = input tokens)
top_k = 50+ → "lost in the middle" kicks in; LLM ignores most of it

Rule of thumb: start at k=3–5. Increase only if evals show answers
missing information that DID exist in the index.
```

**Measuring retrieval quality (before you blame the LLM):**

```python
# Build a small eval set: (question, chunk_id_that_should_be_retrieved)
eval_set = [
    ("Can I return a digital product?", "chunk_digital_refunds"),
    ("How long does a refund take?",    "chunk_refund_timelines"),
    # ... 20–50 of these
]

# Recall@k: fraction of queries where the right chunk appears in top-k
def recall_at_k(k):
    hits = 0
    for question, correct_id in eval_set:
        top = vector_db.search(embed(question), top_k=k)
        if correct_id in [r.id for r in top]:
            hits += 1
    return hits / len(eval_set)

print("Recall@3:", recall_at_k(3))   # e.g., 0.85
print("Recall@10:", recall_at_k(10)) # e.g., 0.95
# If recall@10 is low, the problem is chunking/embeddings, not the LLM.
# If recall@3 is low but recall@10 is high, try re-ranking.
```

**Groundedness — did the LLM actually use the retrieved context?**

Retrieval can be perfect and the LLM can still hallucinate by ignoring the context. Measure this separately:

```python
# LLM-as-judge groundedness check
GROUND_PROMPT = """
Given the CONTEXT and ANSWER below, is every factual claim in the ANSWER
supported by the CONTEXT?

CONTEXT: {context}
ANSWER:  {answer}

Reply with JSON: {{"grounded": bool, "unsupported_claims": [str]}}
"""
# Run this on every eval example. Track groundedness% as a first-class metric.
```

**Common misconception:** Retrieval is a solved problem. In practice, retrieval quality is the #1 cause of bad RAG performance. If you retrieve the wrong chunks, even the best LLM can't give good answers. Measure retrieval recall@k AND groundedness separately before blaming the LLM.

---

## How the Full RAG Pipeline Works (Step-by-Step)

```
                        ┌─────────────────────────────────┐
INDEXING                │ 1. Load documents                │
(happens once)          │ 2. Split into chunks             │
                        │ 3. Embed each chunk              │
                        │ 4. Store in vector DB            │
                        └─────────────────────────────────┘

                        ┌─────────────────────────────────┐
QUERYING                │ 5. User asks a question          │
(per request)           │ 6. Embed the question            │
                        │ 7. Search DB for top-K chunks    │
                        │ 8. Build prompt with chunks      │
                        │ 9. Send to LLM                   │
                        │ 10. Return answer + citations    │
                        └─────────────────────────────────┘
```

```
User Query
    │
    ▼
Embed query ──► Vector DB ──► Top-K chunks
                    ▲               │
                    │               ▼
              Pre-embedded    Build prompt:
              document        [System][Context: chunk1, chunk2...][Question]
              store                   │
                                      ▼
                                   LLM API
                                      │
                                      ▼
                              Answer + source citations
```

---

## Code in Practice

### Minimal RAG in TypeScript

```typescript
import Anthropic from '@anthropic-ai/sdk';
import { ChromaClient } from 'chromadb';

const anthropic = new Anthropic();
const chroma = new ChromaClient();
const collection = await chroma.getOrCreateCollection({ name: 'docs' });

// Step 1: Index documents (run once)
async function indexDocs(docs: string[]) {
  for (let i = 0; i < docs.length; i++) {
    const chunks = chunkText(docs[i], 500);  // your chunking function
    await collection.add({
      ids: chunks.map((_, j) => `doc-${i}-chunk-${j}`),
      documents: chunks,  // ChromaDB auto-embeds with a default model
    });
  }
}

// Step 2: Query with RAG
async function ask(question: string): Promise<string> {
  // Retrieve relevant chunks
  const results = await collection.query({
    queryTexts: [question],
    nResults: 3,
  });

  const context = results.documents[0].join('\n\n---\n\n');

  // Generate with context
  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 1024,
    system: `Answer based only on the provided context.
             If the context doesn't contain the answer, say "I don't have that information."
             Always cite which part of the context you're using.`,
    messages: [{
      role: 'user',
      content: `Context:\n${context}\n\nQuestion: ${question}`
    }]
  });

  return response.content[0].text;
}
```

### Practical: RAG with metadata filtering

```typescript
// Index with metadata
await collection.add({
  ids: ['chunk-1'],
  documents: ['Returns within 30 days with receipt...'],
  metadatas: [{
    source: 'refund-policy.md',
    department: 'customer-support',
    last_updated: '2024-01-15',
  }]
});

// Query with metadata filter
const results = await collection.query({
  queryTexts: ['refund policy'],
  where: { department: 'customer-support' },  // Only search support docs
  nResults: 5,
});
```

### Production: Full RAG pipeline with Python

```python
from anthropic import Anthropic
import chromadb

client = Anthropic()
chroma = chromadb.Client()
collection = chroma.get_or_create_collection("knowledge_base")

def query_rag(question: str, top_k: int = 3) -> dict:
    # 1. Retrieve relevant chunks
    results = collection.query(query_texts=[question], n_results=top_k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    # 2. Build grounded prompt
    context = "\n\n".join(
        f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, chunks)
    )

    # 3. Generate answer
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="""Answer based ONLY on the provided context sources.
                  If unsure, say so. Always cite the source filename.""",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )

    return {
        "answer": response.content[0].text,
        "sources": list(set(sources)),  # deduplicate
    }
```

---

## Advanced RAG Techniques

### Hybrid Search (Semantic + Keyword)

Best of both worlds: semantic search finds conceptually related docs, keyword search finds exact terms.

```
Query: "error code 404 in auth service"

Keyword search (BM25):                 Semantic search:
1. "HTTP 404 in auth module"           1. "Authentication failure handling"
2. "Auth service error codes"          2. "Auth service error codes"
3. "404 page configuration"            3. "Debugging service errors"

Hybrid (re-ranked by both scores):
1. "Auth service error codes"     ← appears in both ✅
2. "HTTP 404 in auth module"      ← strong keyword match
3. "Authentication failure handling" ← strong semantic match
```

### HyDE (Hypothetical Document Embeddings)

When a user query is short/vague, generate a hypothetical answer and search for documents similar to *that*.

```python
# User query: "it's not working"  ← too vague for good retrieval

# Step 1: Generate a hypothetical answer
hypo = llm("Write a short technical answer to: 'it's not working'")
# → "If your application is not working, check: 1. Logs for error messages..."

# Step 2: Search for docs similar to the hypothetical answer
results = vector_db.search(embed(hypo))
# → Finds troubleshooting guides (way better than searching "it's not working")
```

### Re-ranking

Initial retrieval is fast but imperfect. A re-ranker model does a second, more expensive pass to improve ordering.

```
Initial retrieval (fast vector search):
  1. Refund timelines chunk    (score: 0.89)
  2. Digital refunds chunk     (score: 0.85)  ← actually most relevant
  3. Shipping policy chunk     (score: 0.83)

After re-ranking (cross-encoder model — slower, more accurate):
  1. Digital refunds chunk     (score: 0.95)  ← promoted to #1
  2. Refund timelines chunk    (score: 0.72)
  3. Shipping policy chunk     (score: 0.31)  ← demoted
```

---

## RAG vs Fine-tuning — When to Use Which

```
┌─────────────────┬─────────────────────┬─────────────────────┐
│                 │ RAG                 │ Fine-tuning         │
├─────────────────┼─────────────────────┼─────────────────────┤
│ Updates data    │ ✅ Instant           │ ❌ Retrain model    │
│ Cost            │ $ (API + DB)         │ $$$$ (GPU compute) │
│ Hallucination   │ Low (grounded)       │ Medium              │
│ Private data    │ ✅ Stays in your DB  │ ⚠️ Baked in model   │
│ Cites sources   │ ✅ Easy              │ ❌ Hard             │
│ Best for        │ Knowing facts        │ Doing a behavior    │
│ Example         │ "Answer from docs"   │ "Write like us"    │
└─────────────────┴─────────────────────┴─────────────────────┘

Rule of thumb:
  Model needs to KNOW something?  → RAG
  Model needs to DO something?    → Fine-tuning
  Needs both?                     → RAG + Fine-tuning
```

---

## Gotchas & Pitfalls

```
❌ Chunks too small → ✅ Use 300–500 tokens
   Loses surrounding context, retrieves unhelpful fragments

❌ Chunks too large → ✅ Start smaller, measure
   Retrieves too much noise, dilutes relevant content

❌ No chunk overlap → ✅ Use 50–100 token overlap
   Loses information at chunk boundaries

❌ Wrong embedding model for your domain → ✅ Test with your actual data
   General embeddings may miss technical jargon or domain-specific meaning

❌ Retrieving too many chunks → ✅ Start with top-3, measure
   Flooding the prompt with noise confuses the LLM

❌ No fallback for missing info → ✅ Add "If you don't know, say so" to system prompt
   Without this, the LLM hallucinates when context doesn't contain the answer

❌ No source citations → ✅ Always return sources
   Users can't verify answers and trust erodes

❌ Measuring LLM quality, not retrieval quality → ✅ Measure both separately
   Bad retrieval + good LLM = bad answers. You'll blame the model when the real issue is retrieval.
```

---

## When to Use / When NOT to Use RAG

**Use RAG when:**
- Your app needs to answer questions about private/internal documents
- Information changes frequently (news, policies, prices)
- You need source attribution ("according to document X...")
- The LLM's training data doesn't cover your domain
- You need to reduce hallucinations on factual queries

**Don't use RAG when:**
- The base LLM already knows the answer reliably (don't add complexity for free knowledge)
- You need the model to change its *behavior* or *style* (use fine-tuning)
- Your documents are tiny and fit in the context window directly (just include them all)
- Latency is critical and the extra retrieval step is unacceptable (consider pre-warming or caching)

---

## Production Notes

### Cost breakdown (per query)

| Stage | Typical cost | Notes |
|-------|-------------|-------|
| Query embedding | $0.00002 – $0.0001 | One embedding call per query |
| Vector search | ~free (per-query) | Cost is storage + infra, not per-query |
| Reranker (optional) | $0.001 – $0.005 | Cross-encoder over top-50 → top-5 |
| Generation (LLM) | $0.005 – $0.05 | Retrieved chunks inflate input tokens |
| **Total per query** | **~$0.01 – $0.06** | Generation dominates |

**Ingestion** (one-time): embedding 1M chunks ≈ $20–$100; vector DB storage ≈ $70/mo per 1M 1536-d vectors on managed services (Pinecone/Turbopuffer ballpark).

**Biggest cost lever:** chunk size. Smaller chunks = more retrieved context = more input tokens. Tune `top_k` + chunk size together against an eval set.

### Latency budget (p50 / p95)

| Stage | p50 | p95 |
|-------|-----|-----|
| Query embedding | 30 ms | 100 ms |
| Vector search (managed, 1M vectors) | 20–50 ms | 100–200 ms |
| Reranker (top-50) | 100–300 ms | 500 ms–1 s |
| LLM generation | 1–3 s | 5–10 s |
| **End-to-end** | **1.5–4 s** | **6–12 s** |

Stream the generation — users forgive slow starts but not frozen UIs.

### Failure modes

- **No-hit retrieval** — query is out-of-distribution; vector search returns semantically unrelated chunks. Mitigation: set a similarity-score floor and fall back to "I don't know" or hybrid keyword search.
- **Chunk boundary cutoff** — answer spans two chunks, neither contains the full context. Mitigation: overlap chunks by 10–20%.
- **Stale index** — source docs updated, index not re-embedded. Mitigation: track `source_doc_version` in metadata; delta-re-embed on change.
- **Embedding model swap** — if you change the embedder, you must re-embed *everything*. Pin the embedding model version.
- **Prompt injection via retrieved content** — malicious content in your corpus hijacks the model. Treat retrieved text as untrusted input; sanitize and wrap in clear delimiters.
- **"Lost in the middle"** — top-ranked chunk buried between filler chunks gets ignored. Put the highest-ranked chunk first *and* last.

### What to monitor

- **Recall@k** and **MRR** on a golden query set (offline, run in CI).
- **Retrieval similarity score distribution** — a leftward shift signals drift or index rot.
- **End-to-end p50/p95** broken down by stage (retrieval vs generation).
- **No-hit rate** (queries where top score < threshold) — early signal of corpus gaps.
- **Cost per query** and **$/day** — watch output-token spikes on long retrieved contexts.
- **Answer quality** via LLM-as-judge on a sampled 1–5% of traffic ([evals.md](evals.md)).

See [../ml-ops/vector-databases.md](../ml-ops/vector-databases.md) for storage choices and [../ml-ops/llm-observability.md](../ml-ops/llm-observability.md) for tracing the full pipeline.

---

## Related Concepts (The Map)

| If you know... | RAG concept is like... |
|----------------|----------------------|
| Server-side rendering (SSR) | RAG is like SSR for LLMs — fetch fresh data per request |
| Search engines (Elasticsearch) | Vector DB is semantic Elasticsearch |
| Database joins | RAG = JOIN between your query and your knowledge base |
| CDN/caching | Embedding docs is like pre-rendering — expensive once, fast to serve |
| React Suspense + data fetching | RAG pipeline = async data fetch before rendering the answer |

**Connected topics:**
- **LLM Fundamentals** → why LLMs hallucinate (what RAG solves)
- **Prompt Engineering** → RAG prompts need careful context injection design
- **LLM APIs & SDKs** → how to call the LLM with the retrieved context
- **Fine-tuning** → the alternative to RAG when behavior, not knowledge, needs to change

---

## Cheat Sheet

| Term | One-line definition |
|------|---------------------|
| Embedding | List of numbers representing text meaning; similar text → similar numbers |
| Vector DB | Database for storing and similarity-searching embeddings |
| Chunking | Splitting documents into searchable pieces (aim for 300–500 tokens) |
| Top-K | Number of chunks to retrieve (start with 3–5) |
| Overlap | Repeated tokens between adjacent chunks — prevents boundary info loss |
| Hybrid search | Semantic + keyword search combined for better precision |
| Re-ranking | Second-pass model that re-orders retrieved results by relevance |
| HyDE | Search using a hypothetical answer instead of the raw query |
| Grounding | Answers based on provided context, not model memory |

**The minimal RAG prompt:**
```
Answer based ONLY on the following context.
If the context doesn't contain the answer, say "I don't know."
Always cite the source.

Context:
{retrieved_chunks}

Question: {user_question}
```

**Remember these 3 things:**
1. Retrieval quality is more important than LLM quality for RAG performance
2. Start with 300–500 token chunks, 50-token overlap, top-3 retrieval
3. Always include a "say I don't know" fallback in the system prompt

---

## Self-Check Questions

1. **Why can't you just put your entire company knowledge base in the system prompt?**

<details>
<summary>Answer</summary>
Three reasons: (1) Cost — you pay per input token, and 10,000 docs would be millions of tokens per request. (2) Context window limits — even 200K tokens won't fit a large knowledge base. (3) "Lost in the middle" — models lose focus on information buried deep in huge contexts. RAG solves this by retrieving only the 3–5 most relevant chunks.
</details>

2. **What's the difference between semantic search and keyword search, and when does each fail?**

<details>
<summary>Answer</summary>
Keyword search finds exact word matches (good for specific terms, codes, names). Semantic search finds meaning matches (good for paraphrased queries, synonyms). Keyword fails when users paraphrase; semantic fails when users use exact technical terms (like "HTTP 404") that might not map to semantically similar docs. Hybrid search combines both.
</details>

3. **A user asks "it's not working" and your RAG returns irrelevant docs. What's the fix?**

<details>
<summary>Answer</summary>
Use HyDE: first ask the LLM to generate a hypothetical detailed answer to "it's not working," then search for documents similar to that hypothetical answer. The hypothetical answer contains rich technical language ("check logs," "verify configuration," etc.) that retrieves much better results than the vague original query.
</details>

4. **Why do chunks need overlap? Give a concrete example of what goes wrong without it.**

<details>
<summary>Answer</summary>
Without overlap, information spanning a chunk boundary is split. Example: chunk 1 ends with "returns within 30 days" and chunk 2 starts with "with receipt required." Neither chunk answers "What do I need for a return?" correctly. With 50-token overlap, both chunks contain the complete sentence, so retrieval finds the full answer.
</details>

5. **Your RAG system gives correct answers 70% of the time. How do you debug it?**

<details>
<summary>Answer</summary>
Separate retrieval from generation: (1) First, test retrieval alone — for the failing 30%, are the right chunks being retrieved? If not, the problem is chunking, embedding, or retrieval, not the LLM. (2) If the right chunks ARE retrieved but the LLM still gives wrong answers, the problem is prompt design or the LLM not following grounding instructions. Fix retrieval first — it's the most common bottleneck.
</details>

---

## Go Deeper

1. **[LlamaIndex Documentation](https://docs.llamaindex.ai/)** — The most comprehensive RAG framework for Python. The docs include excellent conceptual explanations of every component. Start with "Understanding" section. (2 hours)

2. **[Pinecone Learning Center](https://www.pinecone.io/learn/)** — Provider-agnostic guides on embeddings, vector search, and RAG architecture. Best hands-on explainers with code examples. (2 hours)

3. **[BEIR Benchmark](https://github.com/beir-cellar/beir)** — Standard benchmark for retrieval evaluation. Understanding how retrieval is measured helps you build better RAG systems. (30 min to understand, ongoing as reference)

4. **[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)** — The original RAG paper by Lewis et al. (Meta AI). Read the abstract and introduction to understand the original formulation vs. what practitioners do today. (20 min)

5. **[ChromaDB Quickstart](https://docs.trychroma.com/getting-started)** — Build your first RAG system in 20 minutes with ChromaDB + Python. The fastest path from theory to working code. (20 min)

---

**What's next?** You know how to give LLMs knowledge. Now learn how to build full applications: [LLM APIs & SDKs →](llm-apis-sdks.md)
