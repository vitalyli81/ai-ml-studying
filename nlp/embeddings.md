# Embeddings

## What Is It?

An embedding is a way to represent words, sentences, or documents as **dense vectors of numbers** where **similar meanings are close together** in the vector space. It's how computers understand that "king" and "queen" are related, even though the letters are completely different.

Think of it as GPS coordinates for meaning — similar concepts are at nearby locations.

## Frontend Analogy

```javascript
// Without embeddings: words are arbitrary IDs (like CSS class names)
const words = { "cat": 0, "dog": 1, "king": 2, "queen": 3 };
// No relationship between numbers. "cat" (0) isn't "closer" to "dog" (1).

// With embeddings: words are positioned in meaning-space
const embeddings = {
  "cat":   [0.2, 0.8, 0.1],   // furry, pet, small
  "dog":   [0.3, 0.7, 0.2],   // furry, pet, medium — close to cat!
  "king":  [0.9, 0.1, 0.8],   // royal, human, male
  "queen": [0.9, 0.1, 0.2],   // royal, human, female — close to king!
};

// Distance between cat and dog: small (similar meanings)
// Distance between cat and king: large (different meanings)
```

It's like the difference between using `z-index: 1, 2, 3` (arbitrary) vs actual `position: {x, y}` coordinates that encode spatial relationships.

## Why Embeddings Matter

### The Problem: Computers Don't Understand Words

```
One-hot encoding (the naive approach):
  "cat"   = [1, 0, 0, 0, 0, ...]   (10,000-dimensional vector, mostly zeros)
  "dog"   = [0, 1, 0, 0, 0, ...]
  "kitten" = [0, 0, 1, 0, 0, ...]

Problems:
  1. cat and kitten look equally different as cat and rocket
  2. Vectors are HUGE (vocabulary size = dimension count)
  3. No semantic information whatsoever
```

### The Solution: Dense Embeddings

```
Embedding (learned representation):
  "cat"    = [0.2, 0.8, 0.1, -0.3, ...]   (256-dimensional, dense)
  "dog"    = [0.3, 0.7, 0.2, -0.2, ...]    ← close to cat!
  "kitten" = [0.2, 0.9, 0.1, -0.4, ...]    ← very close to cat!
  "rocket" = [-0.8, 0.1, 0.9, 0.7, ...]    ← far from cat

Dimensions don't have human-readable names, but they encode
concepts like: [animal-ness, size, royalty, gender, ...]
```

## Word2Vec — Where It All Started (2013)

The breakthrough idea: **words that appear in similar contexts have similar meanings**.

### The Training Idea

```
"The cat sat on the mat"
"The dog sat on the rug"

"cat" and "dog" appear in the same context ("The ___ sat on the ___")
→ Their embeddings should be similar
```

### The Famous Result

Word2Vec discovered that **vector arithmetic = meaning arithmetic**:

```
king - man + woman ≈ queen

vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

```
                 woman
  queen ←─────────── king
    |                  |
    | female→male      | female→male
    |                  |
  woman ←─────────── man
                 woman
```

More examples:
```
Paris - France + Italy ≈ Rome        (capital relationships)
bigger - big + small ≈ smaller       (comparative forms)
walking - walk + swim ≈ swimming     (tense patterns)
```

## GloVe — The Other Classic (2014)

GloVe (Global Vectors) uses a different approach — analyzes **word co-occurrence statistics** across the entire corpus. Results are similar to Word2Vec.

**You don't need to choose between them.** Both are mostly historical now — modern Transformer-based embeddings are better.

## The Big Problem: Context

Word2Vec/GloVe give each word **one fixed vector**, regardless of context:

```
"I went to the bank to deposit money"     ← bank = financial institution
"I sat on the bank of the river"           ← bank = riverside

Word2Vec: "bank" = [0.3, 0.5, -0.1, ...]  ← SAME vector for both!
```

This is a major limitation. Enter **contextual embeddings**.

## Contextual Embeddings (BERT, GPT) — The Modern Approach

Transformer models produce **different vectors for the same word** depending on context:

```
"I went to the bank to deposit money"
  "bank" = [0.8, 0.1, 0.3, ...]     ← financial meaning

"I sat on the bank of the river"
  "bank" = [0.1, 0.7, -0.2, ...]    ← riverside meaning

Different vectors! The model understands context.
```

This is why BERT and GPT are so much better than Word2Vec — they don't just know what a word means, they know what it means **right now, in this sentence**.

## Sentence Embeddings — Comparing Whole Texts

For many tasks (search, similarity, RAG), you need to embed **entire sentences or paragraphs**, not just individual words.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "What's the weather today?",
]

embeddings = model.encode(sentences)
# Each sentence → one vector (384 dimensions)
# embeddings.shape = (3, 384)

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity

sims = cosine_similarity(embeddings)
# sentences[0] vs sentences[1]: 0.82 — high! (similar meaning)
# sentences[0] vs sentences[2]: 0.11 — low (different topics)
```

## Cosine Similarity — Measuring Closeness

How do you measure if two vectors are "similar"? **Cosine similarity** — measure the angle between them:

```
Cosine similarity = dot(A, B) / (|A| × |B|)

Range: -1 to 1
  1.0  → identical meaning
  0.0  → unrelated
 -1.0  → opposite meaning
```

```
                    "king"
                   ↗  angle = small → similar!
    origin ──────→ "queen"


                   "king"
                  ↗
    origin ──────────────→ "banana"
              angle = large → not similar
```

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Or use sklearn:
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([vec_a], [vec_b])[0][0]
```

## Embeddings for RAG (Why This Matters for AI Engineers)

RAG (Retrieval-Augmented Generation) is the most common AI engineering task, and it's **entirely built on embeddings**:

```
1. EMBED your documents:
   "How to reset password" → [0.2, 0.8, -0.1, ...]
   "Billing FAQ"           → [0.7, 0.1, 0.3, ...]
   "Return policy"         → [0.5, 0.3, 0.6, ...]
   Store in vector database (Pinecone, ChromaDB, etc.)

2. User asks a question:
   "I can't log in" → [0.3, 0.7, -0.2, ...]  ← embed the query

3. Find closest documents:
   cosine_similarity("I can't log in", "How to reset password") = 0.89 ← match!
   cosine_similarity("I can't log in", "Billing FAQ") = 0.12

4. Send the matched document + question to the LLM:
   "Based on this doc: [How to reset password...], answer: I can't log in"
```

## Popular Embedding Models

| Model | Dimensions | Use Case |
|-------|-----------|----------|
| **all-MiniLM-L6-v2** | 384 | Fast, good quality. Great starting point |
| **text-embedding-3-small** (OpenAI) | 1536 | API-based, good quality |
| **text-embedding-3-large** (OpenAI) | 3072 | Best quality from OpenAI |
| **voyage-3** (Voyage AI) | 1024 | Strong for code and text |
| **nomic-embed-text** | 768 | Open source, good quality |

## Python Example — Full Workflow

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Your knowledge base
docs = [
    "To reset your password, click 'Forgot Password' on the login page",
    "Refunds are processed within 5-7 business days",
    "Our office hours are Monday to Friday, 9am to 5pm",
    "To upgrade your plan, go to Settings > Billing > Change Plan",
]

# 3. Embed all documents
doc_embeddings = model.encode(docs)
print(f"Each doc is a {doc_embeddings.shape[1]}-dimensional vector")

# 4. User asks a question
query = "How do I change my password?"
query_embedding = model.encode([query])

# 5. Find the most relevant document
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

for doc, score in sorted(zip(docs, similarities), key=lambda x: -x[1]):
    print(f"  {score:.3f}: {doc[:60]}...")

# Output:
#   0.812: To reset your password, click 'Forgot Password' on the...
#   0.203: To upgrade your plan, go to Settings > Billing > Chang...
#   0.105: Our office hours are Monday to Friday, 9am to 5pm...
#   0.067: Refunds are processed within 5-7 business days...
```

## Key Takeaway

Embeddings convert text into **vectors where meaning is encoded as position** — similar texts are close together, different texts are far apart. Classic embeddings (Word2Vec) give one vector per word. Modern embeddings (BERT-based) are **contextual** — the same word gets different vectors based on context. For AI engineering, **sentence embeddings** are the key skill — they power semantic search, RAG systems, and similarity matching. This is one of the most practically important concepts for your career.
