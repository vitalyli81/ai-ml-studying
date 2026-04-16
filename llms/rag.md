# RAG (Retrieval-Augmented Generation)

## What Is It?

RAG is a technique that gives LLMs access to **external knowledge** by retrieving relevant documents and including them in the prompt. Instead of relying only on what the model memorized during training, you **fetch the right information at query time** and hand it to the model.

```
Without RAG:
  User: "What's our refund policy?"
  LLM:  "I don't know your specific refund policy." (or worse, makes one up)

With RAG:
  1. Search your docs for "refund policy" → finds refund-policy.md
  2. Insert that document into the prompt
  3. LLM: "Your refund policy allows returns within 30 days with receipt..."
```

## Frontend Analogy

```javascript
// RAG is like server-side rendering with data fetching

// WITHOUT RAG (pure LLM) — like a static page with hardcoded data:
function Page() {
  return <div>Data from build time only (training data)</div>;
}

// WITH RAG — like getServerSideProps fetching fresh data:
async function getServerSideProps(context) {
  const query = context.params.question;
  const relevantDocs = await vectorDB.search(query);  // ← retrieval
  return { props: { docs: relevantDocs } };
}

function Page({ docs }) {
  // LLM now has current, relevant data to work with
  return <Answer context={docs} />;
}

// The LLM is the renderer, RAG is the data fetching layer
```

## Why RAG Matters

| Problem | Without RAG | With RAG |
|---------|------------|----------|
| Knowledge cutoff | "I only know things up to my training date" | Fetches current information |
| Hallucinations | Makes up plausible-sounding but wrong answers | Answers grounded in real documents |
| Private data | Can't access your company's docs | Searches your internal knowledge base |
| Source attribution | "I think..." (no source) | "According to refund-policy.md..." |
| Cost | Fine-tuning is expensive | Just update your document store |

## The RAG Pipeline

```
User Query → Embed Query → Search Vector DB → Retrieve Top-K Docs → Build Prompt → LLM → Response
                                    ↑
                            ┌───────┴────────┐
                            │  Vector Store   │
                            │  (your docs,    │
                            │   pre-embedded) │
                            └─────────────────┘
```

### Step-by-Step Breakdown

### Step 1: Prepare Your Documents (Indexing)

Before you can search, you need to process and store your documents.

```
Raw Documents                    Chunks                        Vectors
┌──────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│ refund.pdf   │     │ "Returns within 30   │     │ [0.12, -0.45,    │
│ (50 pages)   │ ──► │  days with receipt"  │ ──► │  0.78, 0.33, ...]│
│              │     ├──────────────────────┤     ├──────────────────┤
│              │     │ "Digital products    │     │ [-0.22, 0.91,    │
│              │     │  are non-refundable" │     │  0.15, -0.67,...]│
└──────────────┘     └──────────────────────┘     └──────────────────┘
    Document              Chunking                   Embedding
```

### Step 2: Chunk Your Documents

You can't embed a 50-page PDF as one vector — it would lose detail. You split it into meaningful chunks.

```python
# Common chunking strategies

# 1. Fixed-size chunks (simplest)
chunks = split_text(document, chunk_size=500, overlap=50)
# Each chunk is ~500 tokens with 50-token overlap between chunks

# 2. Sentence-based chunking
chunks = split_by_sentences(document, max_sentences=5)

# 3. Semantic chunking (smarter)
# Split at natural boundaries: paragraphs, sections, headers
chunks = split_by_headers(document)  # Each section = one chunk
```

**Chunk size tradeoffs:**
```
Small chunks (100-200 tokens):
  ✅ More precise retrieval
  ❌ Might miss broader context
  Use for: FAQ, short answers

Medium chunks (300-500 tokens):
  ✅ Good balance of precision and context
  Use for: Most use cases (start here)

Large chunks (500-1000 tokens):
  ✅ More context per result
  ❌ Less precise, more noise
  Use for: Complex topics needing full paragraphs
```

**Overlap matters:**
```
Without overlap:
  Chunk 1: "...returns within 30 days"  |  Chunk 2: "with receipt. Digital..."
  → If someone asks about "30-day receipt policy", neither chunk has the full answer

With overlap (50 tokens):
  Chunk 1: "...returns within 30 days with receipt. Digital..."
  Chunk 2: "...30 days with receipt. Digital products are..."
  → Both chunks capture the boundary content
```

### Step 3: Create Embeddings

An embedding is a **vector (list of numbers)** that represents the meaning of text. Similar meanings → similar vectors.

```
Text: "How do I return a product?"
Vector: [0.12, -0.45, 0.78, 0.33, -0.91, 0.55, ...]  (1536 numbers)

Text: "What's the refund process?"
Vector: [0.14, -0.42, 0.75, 0.31, -0.88, 0.52, ...]  (very similar!)

Text: "What's the weather today?"
Vector: [-0.67, 0.23, -0.11, 0.89, 0.45, -0.33, ...] (very different)
```

```javascript
// Frontend analogy: Embeddings are like a hash function for meaning

// Hash function: same input → same hash (exact match)
// hash("hello") === hash("hello")  ✅
// hash("hello") !== hash("hi")     ✅ (different strings)

// Embedding: similar meaning → similar vectors (semantic match)
// embed("How do I return a product?") ≈ embed("What's the refund process?")
// Even though the WORDS are completely different!

// This is the magic — search by MEANING, not keywords
```

**Popular embedding models:**
```
Model                    Dimensions   Speed    Quality
───────────────────────────────────────────────────────
OpenAI text-embedding-3-small  1536  Fast     Good
OpenAI text-embedding-3-large  3072  Medium   Better
Cohere embed-v3               1024  Fast     Great
Voyage AI voyage-3            1024  Fast     Great (for code)
BGE / E5 (open source)       768-1024  Fast  Good (free!)
```

### Step 4: Store in a Vector Database

A vector database is optimized for storing and searching vectors by similarity.

```javascript
// It's like a regular database, but instead of:
//   SELECT * FROM docs WHERE title LIKE '%refund%'    (keyword match)

// You do:
//   SELECT * FROM docs ORDER BY similarity(embedding, query_embedding) LIMIT 5
//   (meaning match)
```

**Popular vector databases:**
```
Database        Type            Best For
─────────────────────────────────────────────────────
Pinecone        Managed cloud   Production, zero maintenance
ChromaDB        Local/embedded  Prototyping, small projects
pgvector        PostgreSQL ext  Already using Postgres
Weaviate        Self-hosted     Full-featured, hybrid search
Qdrant          Self-hosted     Performance, Rust-based
FAISS           Library (Meta)  Research, in-memory
```

### Step 5: Retrieve and Generate

At query time, embed the user's question and find the most similar document chunks.

```python
# Pseudocode for the full RAG pipeline

# 1. User asks a question
query = "Can I return a digital product?"

# 2. Embed the query (same model used for documents)
query_vector = embed(query)  # → [0.14, -0.42, ...]

# 3. Search vector DB for similar chunks
results = vector_db.search(query_vector, top_k=3)
# Returns:
# [
#   { text: "Digital products are non-refundable...", score: 0.92 },
#   { text: "Returns within 30 days with receipt...", score: 0.78 },
#   { text: "Contact support for exceptions...", score: 0.71 },
# ]

# 4. Build the prompt with retrieved context
prompt = f"""Answer the user's question based ONLY on the following context.
If the context doesn't contain the answer, say "I don't know."

Context:
{results[0].text}
{results[1].text}
{results[2].text}

Question: {query}
"""

# 5. Send to LLM
answer = llm.generate(prompt)
# → "No, digital products are non-refundable according to our policy.
#    However, you can contact support for exceptions."
```

## Building a RAG System in JavaScript

Here's how it looks with real code you'd write as an AI engineer:

```typescript
import { Anthropic } from '@anthropic-ai/sdk';
import { ChromaClient } from 'chromadb';

// 1. Initialize
const anthropic = new Anthropic();
const chroma = new ChromaClient();
const collection = await chroma.getOrCreateCollection({ name: "docs" });

// 2. Index documents (do this once)
async function indexDocuments(docs: string[]) {
  for (const doc of docs) {
    const chunks = chunkText(doc, { size: 500, overlap: 50 });
    
    await collection.add({
      ids: chunks.map((_, i) => `doc-${i}`),
      documents: chunks,  // ChromaDB auto-embeds with a default model
    });
  }
}

// 3. Query with RAG
async function askQuestion(question: string): Promise<string> {
  // Retrieve relevant chunks
  const results = await collection.query({
    queryTexts: [question],
    nResults: 5,
  });

  const context = results.documents[0].join('\n\n');

  // Generate answer with context
  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-5-20241022',
    max_tokens: 1024,
    system: `Answer questions based on the provided context. 
             If the context doesn't contain the answer, say so.
             Always cite which part of the context you're using.`,
    messages: [{
      role: 'user',
      content: `Context:\n${context}\n\nQuestion: ${question}`
    }]
  });

  return response.content[0].text;
}
```

## Advanced RAG Techniques

### Hybrid Search (Keywords + Semantic)

Combine traditional keyword search (BM25) with vector similarity for better results.

```
User: "error code 404 in auth service"

Keyword search (BM25): Great at finding "404" and "auth service" exactly
Semantic search: Great at finding docs about "authentication failures"

Hybrid = both combined → best of both worlds
```

```
Query: "error code 404 in auth service"

BM25 Results:                    Vector Results:
1. "HTTP 404 in auth module"     1. "Authentication failure handling"
2. "Auth service error codes"    2. "Auth service error codes"
3. "404 page configuration"      3. "Debugging service errors"

Hybrid (re-ranked):
1. "Auth service error codes"        ← appears in both
2. "HTTP 404 in auth module"         ← strong keyword match
3. "Authentication failure handling"  ← strong semantic match
```

### Query Transformation

Sometimes the user's query isn't great for retrieval. Transform it first.

```python
# Original query: "it's not working"  ← too vague for search

# Technique 1: Query expansion
# Ask the LLM to generate a better search query
better_query = llm("Rewrite this user question as a detailed search query: 'it's not working'")
# → "troubleshooting common errors and issues"

# Technique 2: HyDE (Hypothetical Document Embeddings)
# Ask the LLM to write a hypothetical answer, then search for documents similar to that
hypothetical_answer = llm("Write a short paragraph answering: 'it's not working'")
# → "If your application is not working, check the following: 1. Verify..."
# Now search for docs similar to this hypothetical answer
results = vector_db.search(embed(hypothetical_answer))
```

### Re-ranking

After initial retrieval, use a more powerful model to re-rank results by relevance.

```
Initial retrieval (fast, might be noisy):
  1. Chunk about refund timelines    (score: 0.89)
  2. Chunk about digital refunds     (score: 0.85)  ← most relevant
  3. Chunk about shipping policy     (score: 0.83)
  4. Chunk about refund exceptions   (score: 0.80)

After re-ranking (slower, more accurate):
  1. Chunk about digital refunds     (score: 0.95)  ← promoted!
  2. Chunk about refund exceptions   (score: 0.88)
  3. Chunk about refund timelines    (score: 0.72)
  4. Chunk about shipping policy     (score: 0.31)  ← demoted
```

### Metadata Filtering

Add metadata to chunks for more targeted search.

```javascript
// When indexing, add metadata
await collection.add({
  ids: ["doc-1"],
  documents: ["Returns within 30 days..."],
  metadatas: [{
    source: "refund-policy.md",
    department: "customer-support",
    last_updated: "2024-01-15",
    audience: "customer"
  }]
});

// When querying, filter by metadata
const results = await collection.query({
  queryTexts: ["refund policy"],
  where: { department: "customer-support" },  // Only search support docs
  nResults: 5,
});
```

## RAG vs Fine-tuning — When to Use Which

```
┌─────────────────┬──────────────────────┬──────────────────────┐
│                 │ RAG                  │ Fine-tuning          │
├─────────────────┼──────────────────────┼──────────────────────┤
│ Updates data    │ ✅ Instant (update DB)│ ❌ Retrain model     │
│ Cost            │ $ (API + DB hosting)  │ $$$$ (GPU compute)  │
│ Setup time      │ Hours                 │ Days/weeks          │
│ Accuracy        │ High (with good docs) │ High (with good data)│
│ Best for        │ Knowledge Q&A         │ Behavior/style change│
│ Private data    │ ✅ Data stays in DB   │ ⚠️ Data baked in model│
│ Hallucination   │ Low (grounded)        │ Medium               │
│ Example         │ "Answer from our docs"│ "Write like our brand"│
└─────────────────┴──────────────────────┴──────────────────────┘

Rule of thumb:
- Need the model to KNOW something?      → RAG
- Need the model to DO something?         → Fine-tuning
- Need both?                              → RAG + Fine-tuning
```

## Common RAG Pitfalls

```
❌ Chunks too small → loses context, retrieves fragments
❌ Chunks too large → retrieves irrelevant noise
❌ No overlap      → information lost at chunk boundaries
❌ Wrong embedding model → poor similarity matching
❌ Too few results  → might miss the answer
❌ Too many results → floods context window with noise
❌ No source citation → user can't verify answers
❌ No fallback      → model hallucinates when context lacks answer

✅ Start with 300-500 token chunks, 50-token overlap
✅ Use hybrid search (semantic + keyword)
✅ Retrieve 3-5 chunks, re-rank, then use top 3
✅ Always include "If you don't know, say so" in system prompt
✅ Return source citations with every answer
```

## Key Takeaways

| Concept | What to Remember |
|---------|-----------------|
| Embedding | Vector (list of numbers) representing meaning |
| Vector DB | Database optimized for similarity search |
| Chunking | Splitting docs into searchable pieces |
| Top-K | Number of results to retrieve (start with 3-5) |
| Hybrid search | Keywords + semantic = best results |
| Re-ranking | Second pass to improve result quality |

## What's Next?

Now you know how to give LLMs knowledge. Next, let's learn how to **build applications** with them using [LLM APIs & SDKs](llm-apis-sdks.md).
