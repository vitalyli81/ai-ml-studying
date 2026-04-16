# Vector Databases

## TL;DR

A vector database stores data as **lists of numbers (vectors)** and lets you search by **meaning** rather than exact keyword match. When you ask "find documents similar to this question," it computes distances between vectors and returns the closest matches in milliseconds — even across millions of items. It's the engine that powers RAG, semantic search, recommendation systems, and duplicate detection.

> 💡 **Key Insight:** Traditional databases answer "does this row contain the word 'dog'?" Vector databases answer "find me everything that's conceptually related to dogs" — even if the word "dog" never appears.

---

## The Mental Model

Think of a **map of a city**, where every document, image, or piece of data is plotted at a specific location.

- Similar things are physically close together on the map
- "Poodle" and "Labrador" are in the same neighborhood (dogs)
- "Jazz" and "Blues" are in the same neighborhood (music genres)
- "Poodle" and "Jazz" are far apart (unrelated)

Mapping:
- City map → The vector space (high-dimensional)
- Location coordinates → The vector (list of 1,536 numbers)
- Distance between locations → Similarity between meanings
- Neighborhood search → "Find me the top 5 most similar items"

When you search, you drop a pin at your query's location and ask: "what's in the 5 nearest buildings?"

---

## Why It Exists

### The Problem

Traditional databases are great at exact matches. But AI needs fuzzy, semantic search.

```sql
-- Traditional SQL: keyword match
SELECT * FROM articles WHERE content LIKE '%neural network%';
-- Misses: "deep learning", "artificial neurons", "backprop"
-- Finds:  only exact phrase matches

-- What you actually want:
-- "Find articles that are ABOUT the same topic as my query,
--  regardless of exact wording"
```

### The Solution

Represent meaning as numbers. Things with similar meanings get similar numbers. Then search by numeric proximity.

```
Before vector DBs: 
  Search = keyword matching (LIKE, full-text search)
  Good for: finding exact terms
  Bad for: semantic understanding

After vector DBs:
  Search = vector similarity
  Good for: semantic understanding, natural language queries
  Works even when vocabulary is completely different
```

### What Changed

```
AI applications that became possible:
  ✅ "Find documents relevant to this question" (RAG)
  ✅ "Find products similar to this one" (recommendations)
  ✅ "Is this email a duplicate of a previous one?" (deduplication)
  ✅ "Find images that look like this photo" (visual search)
  ✅ "Cluster these support tickets by topic" (grouping)
```

---

## Core Concepts

### 1. Vectors & Embeddings

**One-line definition:** A vector is a list of numbers that represents the meaning of a piece of data.

**Analogy:** Imagine rating every movie on 1,536 different scales: how much action, how much romance, how futuristic, how funny, etc. Two movies with similar ratings are similar movies. The ratings list is the vector.

```
"The quick brown fox" → [0.12, -0.45, 0.78, 0.33, ..., 0.91]
                                                          ↑ 1,536 numbers total

"A fast auburn canine" → [0.14, -0.42, 0.75, 0.31, ..., 0.89]
                          nearly identical numbers! → semantically similar

"Quantum entanglement" → [-0.67, 0.23, 0.55, -0.88, ..., 0.12]
                          completely different numbers → semantically different
```

**Technical explanation:** Embedding models (like `text-embedding-3-small`) are neural networks trained to map text to vectors such that similar text lands close together in the vector space. The training objective is: "similar sentences should have high cosine similarity."

**Common misconception:** People think the individual numbers in a vector mean something specific (number 5 = "how romantic it is"). They don't. The numbers only have meaning in relation to each other. The pattern across all 1,536 dimensions encodes meaning.

---

### 2. Similarity Metrics

**One-line definition:** A formula that measures how "close" two vectors are — i.e., how similar two pieces of data are.

**Analogy:** You can measure distance between cities in different ways: straight-line distance (as the crow flies) or driving distance. Different similarity metrics are like different ways of measuring "closeness" in the vector space.

```
The main similarity metrics:

1. Cosine Similarity (most common for text)
   Measures the ANGLE between vectors — ignores magnitude
   Range: -1 (opposite) to 1 (identical)
   Formula: cos(θ) = (A·B) / (|A| × |B|)

   "dog running fast" and "quickly sprinting canine"
   → cosine similarity: 0.92 (very similar!)

2. Euclidean Distance (L2)
   Measures the straight-line distance between vectors
   Range: 0 (identical) to ∞ (far apart)
   Good for: image similarity, numeric data

3. Dot Product
   Similar to cosine but affected by vector magnitude
   Used when magnitude matters (e.g., popularity + relevance)
```

```
         Vector A                Vector B
              ↗                       ↗
             ↗                       ↗
            ↗   θ (small angle)     ↗
           ↗──────────────────────→↗
         origin

Small angle θ → high cosine similarity → vectors are "pointing the same direction"
= the texts have similar meaning
```

**Common misconception:** You need to use Euclidean distance. For text, cosine similarity almost always works better because it measures directional similarity (meaning), not magnitude (how long the text is).

---

### 3. HNSW — How Fast Search Works

**One-line definition:** HNSW (Hierarchical Navigable Small World) is the algorithm that makes searching millions of vectors take milliseconds instead of hours.

**Analogy:** Imagine finding your way in a city using a multi-layer map:
- Layer 1 (top): World map — jump between continents quickly
- Layer 2: Country map — navigate to the right region
- Layer 3: City map — find the right neighborhood
- Layer 4 (bottom): Street map — find the exact address

You don't check every street on earth to find your destination. You navigate hierarchically. HNSW does the same with vectors.

```
Naive search (brute force):
  Compare your query to ALL 10M vectors
  10M × 1,536 multiplications
  Time: 30-60 seconds ❌

HNSW search:
  Navigate the graph layer by layer
  Check only ~200-500 vectors total
  Time: <10 milliseconds ✅

Trade-off: HNSW is approximate (might miss the absolute closest vector),
but in practice accuracy is >99% for most use cases.
```

**Common misconception:** Vector search is always slow. Brute-force is slow. HNSW (used by every major vector DB) is extremely fast even at scale.

---

### 4. Metadata Filtering

**One-line definition:** Filtering search results by structured fields (date, category, user ID) alongside the vector similarity search.

**Analogy:** It's like searching Google Maps for "coffee shop" but filtering by "open now" + "within 1 mile". The vector search finds semantically similar items; the filter narrows to the ones that also match structured criteria.

```python
# Without metadata filtering — too broad
results = db.search("refund policy", top_k=5)
# Returns docs from all departments, all languages, all dates

# With metadata filtering — targeted
results = db.search(
    "refund policy",
    top_k=5,
    filter={
        "department": "customer-support",
        "language": "en",
        "date": { "$gte": "2024-01-01" }  # Only recent docs
    }
)
# Returns only relevant, recent, English customer-support docs
```

---

### 5. Index Types

```
Flat (brute force):
  - Exact results (100% accurate)
  - Scales poorly — O(n) search time
  - Use for: small datasets (<100K vectors), when accuracy is critical

IVF (Inverted File Index):
  - Clusters vectors into groups, searches only relevant clusters
  - Faster than flat, slightly less accurate
  - Use for: medium datasets (100K–10M vectors)

HNSW (Hierarchical Navigable Small World):
  - Graph-based navigation, very fast
  - ~99% accuracy
  - Use for: most production use cases ✅

Product Quantization (PQ):
  - Compresses vectors to save memory
  - Use for: huge datasets where memory is the bottleneck
```

---

## The Major Vector Databases

```
Database      Type          Best For                    Cost
──────────────────────────────────────────────────────────────
Pinecone      Managed       Production, zero-ops        $70+/mo
Weaviate      Self-hosted   Full-featured, hybrid       Free (self-host)
Qdrant        Self-hosted   Performance, Rust           Free (self-host)
ChromaDB      Embedded      Local dev, prototyping      Free
pgvector      PostgreSQL    Already using Postgres      Free (extension)
FAISS         Library       Research, in-memory         Free
Redis VSS     Redis         Low-latency, real-time      $$$
```

### Decision Guide

```
Just prototyping / learning?
  → ChromaDB (runs in-process, zero setup)

Already using PostgreSQL?
  → pgvector (one SQL extension, no new infra)

Need managed cloud (no DevOps)?
  → Pinecone (fully managed, easy to start)

Need self-hosted with full control?
  → Qdrant (best performance) or Weaviate (best features)

Research / offline processing?
  → FAISS (Meta's library, battle-tested, in-memory)
```

---

## How It Actually Works (Step-by-Step)

Let's build a semantic document search from scratch:

```
Step 1: Prepare documents
        ["Our refund policy allows 30-day returns...",
         "To reset your password, go to Settings...",
         "Shipping takes 3-5 business days..."]

Step 2: Generate embeddings
        Call embedding API for each document
        doc1 → [0.12, -0.45, 0.78, ...] (1,536 numbers)
        doc2 → [-0.33, 0.67, -0.21, ...] (1,536 numbers)
        doc3 → [0.55, -0.12, 0.44, ...] (1,536 numbers)

Step 3: Store in vector database
        collection.upsert(ids=["doc1","doc2","doc3"],
                          embeddings=[...],
                          documents=[...],
                          metadatas=[{type:"policy"},{type:"help"},{type:"shipping"}])

Step 4: User asks a question
        "How do I get my money back?"

Step 5: Embed the query
        "How do I get my money back?" → [0.11, -0.43, 0.76, ...]
        (Very similar to doc1's vector!)

Step 6: Search for similar vectors
        Compare query vector to all stored vectors
        doc1 similarity: 0.94 ← closest!
        doc3 similarity: 0.61
        doc2 similarity: 0.38

Step 7: Return top results
        [{ doc: "Our refund policy allows 30-day returns...", score: 0.94 }]

Step 8: Feed to LLM as context (RAG!)
        "Based on this document: [doc1 text], answer: How do I get my money back?"
```

---

## Code in Practice

### 1. ChromaDB — Local Prototype

```python
import chromadb
from chromadb.utils import embedding_functions

# ChromaDB runs in-memory or on disk — no server needed
client = chromadb.PersistentClient(path="./chroma_db")

# Use OpenAI embeddings (or any embedding model)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="support_docs",
    embedding_function=openai_ef
)

# Add documents — ChromaDB auto-embeds them
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "Our refund policy allows 30-day returns with receipt.",
        "To reset password, go to Settings > Security > Change Password.",
        "Shipping takes 3-5 business days for standard delivery."
    ],
    metadatas=[
        {"category": "policy"},
        {"category": "account"},
        {"category": "shipping"}
    ]
)

# Search semantically
results = collection.query(
    query_texts=["How do I get my money back?"],
    n_results=2,
    where={"category": "policy"}  # Optional metadata filter
)

print(results["documents"][0])
# [['Our refund policy allows 30-day returns with receipt.']]
print(results["distances"][0])
# [[0.08]]  ← very small distance = very similar (ChromaDB uses L2)
```

### 2. Pinecone — Production

```python
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

pc = Pinecone(api_key="your-pinecone-key")
openai_client = OpenAI(api_key="your-openai-key")

# Create index (one-time setup)
pc.create_index(
    name="support-docs",
    dimension=1536,  # text-embedding-3-small output size
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("support-docs")

def embed(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Upsert vectors
vectors = [
    {
        "id": "doc1",
        "values": embed("Our refund policy allows 30-day returns."),
        "metadata": {"text": "Our refund policy...", "category": "policy"}
    }
]
index.upsert(vectors=vectors)

# Query
query_vector = embed("How do I get my money back?")
results = index.query(
    vector=query_vector,
    top_k=3,
    filter={"category": "policy"},
    include_metadata=True
)

for match in results.matches:
    print(f"Score: {match.score:.3f} | {match.metadata['text'][:60]}")
```

### 3. pgvector — PostgreSQL Extension

```sql
-- Enable the extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table with a vector column
CREATE TABLE documents (
    id       SERIAL PRIMARY KEY,
    content  TEXT,
    category TEXT,
    embedding vector(1536)  -- 1536-dimensional vector
);

-- Insert a document with its embedding (from your app code)
INSERT INTO documents (content, category, embedding)
VALUES ('Our refund policy allows 30-day returns.', 'policy', '[0.12, -0.45, ...]');

-- Create an HNSW index for fast search
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- Semantic search — find the 5 most similar documents
SELECT content, 1 - (embedding <=> '[0.11, -0.43, ...]'::vector) AS similarity
FROM documents
WHERE category = 'policy'
ORDER BY embedding <=> '[0.11, -0.43, ...]'::vector
LIMIT 5;
```

```python
# From Python using psycopg2
import psycopg2

conn = psycopg2.connect("postgresql://user:pass@localhost/mydb")
cur = conn.cursor()

query_embedding = embed("How do I get my money back?")

cur.execute("""
    SELECT content, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 5
""", (query_embedding, query_embedding))

results = cur.fetchall()
for content, similarity in results:
    print(f"{similarity:.3f}: {content[:60]}")
```

---

## Gotchas & Pitfalls

```
❌ Using the wrong embedding model for search and indexing
   You MUST use the same model for both embedding documents and queries
   Mixing models = completely wrong results (different vector spaces)
✅ Pick one model and use it everywhere

❌ Not normalizing vectors when using cosine similarity
   Some libraries expect normalized vectors (length = 1)
✅ Check your library's docs — most handle this automatically

❌ Embedding entire documents as one vector
   1,000-word document → one vector → loses fine-grained detail
✅ Chunk documents before embedding (300-500 tokens each)

❌ Searching without metadata filters when you should
   "Find docs about refunds" returns docs from all languages and dates
✅ Add metadata (language, date, department) and filter at search time

❌ Choosing a vector DB before knowing your scale
   ChromaDB is great for 100K vectors, terrible for 100M
✅ Prototype with ChromaDB, switch to Pinecone/Qdrant at scale

❌ Not having a re-indexing strategy
   Your data changes — old embeddings become stale
✅ Plan for incremental updates: upsert (insert or update), not just insert
```

---

## When to Use / When NOT to Use

### Use Vector Databases When:
- Building RAG systems (finding relevant docs for LLM context)
- Adding semantic search to your app ("find similar items")
- Deduplication at scale ("is this a near-duplicate?")
- Recommendation systems ("users who liked X also liked Y")

### Don't Use Vector Databases When:
- Simple keyword search is sufficient → use Elasticsearch or PostgreSQL full-text search
- Your dataset is tiny (<10,000 items) → just use FAISS in-memory or a simple similarity loop
- You need exact matches (usernames, IDs, exact phrases) → use a regular database
- Real-time data that changes every second → vector indexes don't update that fast

---

## Related Concepts

| Concept | Connection |
|---------|------------|
| RAG | Vector DBs are RAG's retrieval layer — they're inseparable |
| Embeddings | The format that goes into vector DBs — you need an embedding model first |
| Model Serving | Your embedding model needs to be served to generate vectors at query time |
| Monitoring | Track retrieval quality: are the top results actually relevant? |

---

## Cheat Sheet

```
Key operations:
  embed(text) → vector       Use same model for index + queries
  upsert(id, vector, meta)   Insert or update a vector
  query(vector, top_k=5)     Find top-K most similar vectors
  filter(metadata={...})     Narrow search with structured fields

Similarity metrics:
  cosine   → text similarity (most common)
  euclidean → numeric/spatial data
  dot product → when magnitude matters

Quick picks:
  Learning?      → ChromaDB (pip install chromadb, no server)
  Have Postgres? → pgvector (1 SQL extension)
  Going to prod? → Pinecone (managed) or Qdrant (self-hosted)

Remember:
  1. Same embedding model for index AND queries — always
  2. Chunk text before embedding (300-500 tokens)
  3. Add metadata to enable filtering — you'll need it later
```

---

## Self-Check Questions

<details>
<summary>Click to reveal answers</summary>

**Q1: Why can't you use a regular SQL database for semantic search?**
SQL databases match exact values or patterns (LIKE '%dog%'). They can't measure meaning similarity between text — "canine" and "dog" would be completely unrelated in SQL, but semantically identical in a vector space.

**Q2: What's the most critical rule when using embedding models?**
The SAME model must be used for embedding documents at index time AND for embedding the query at search time. Different models produce incompatible vector spaces — mixing them gives random, meaningless results.

**Q3: Why do we chunk documents before embedding?**
One vector represents one average meaning. A 10,000-word document has many topics — one vector can't represent all of them precisely. Smaller chunks give more precise vectors that map to specific pieces of information.

**Q4: What's the difference between cosine similarity and Euclidean distance?**
Cosine measures the angle between vectors (directional similarity — same meaning regardless of length). Euclidean measures the straight-line distance (affected by vector magnitude). For text, cosine is almost always better.

**Q5: When would you choose pgvector over Pinecone?**
When you're already running PostgreSQL and want to avoid adding new infrastructure. pgvector adds vector search to your existing Postgres DB. Pinecone is better when you need a fully managed solution with no DB administration overhead.

</details>

---

## Go Deeper

| Resource | Why It's Worth Your Time |
|----------|--------------------------|
| [pgvector GitHub](https://github.com/pgvector/pgvector) | If you already know SQL, this is the fastest path to production vector search. The README is excellent. |
| [Pinecone Learning Center](https://www.pinecone.io/learn/) | Best written tutorials on vector databases — covers theory and practice together. Free to read. |
| [ChromaDB Docs](https://docs.trychroma.com) | Best for getting your first semantic search working in under 30 minutes. |
| [Qdrant Documentation](https://qdrant.tech/documentation/) | If you go self-hosted, Qdrant's docs are exceptionally clear on indexing, filtering, and performance. |
| "Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality" | The foundational paper on ANN search — worth skimming to understand why HNSW works. |
