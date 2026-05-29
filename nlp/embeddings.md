# Embeddings

## 1. TL;DR

An embedding converts text (words, sentences, documents) into a **list of numbers (a vector)** where **similar meanings are close together**. It's how machines understand that "dog" and "puppy" are related without anyone telling them. Classic embeddings (Word2Vec) give every word one fixed vector. Modern embeddings (BERT, sentence-transformers) are **contextual** — the same word gets different vectors in different sentences. For AI engineering, sentence embeddings power search, RAG, and similarity matching.

---

## 2. The Mental Model

> 💡 **Think of it like this:** Embeddings are **GPS coordinates for meaning**.

On a map, nearby coordinates = nearby places. In embedding space, nearby vectors = similar meanings. "cat" and "kitten" live in the same neighborhood; "cat" and "rocket" live on different continents. The twist: real embeddings use hundreds of coordinates (not just lat/lng), because meaning has more than two axes.

| Real world | Technical concept |
|---|---|
| GPS coordinates (lat, lng) | Embedding vector (list of numbers) |
| Two locations close on a map | Two words with similar meaning |
| Distance between locations | Cosine distance between embeddings |
| City neighborhoods (finance district, arts district) | Clusters of semantically related words |
| Adding more axes beyond lat/lng (altitude, terrain, climate) | Higher embedding dimensions capturing richer meaning |

---

## Build the Intuition From Zero

The deep question: **where do the numbers come from?** Nobody sits down and assigns "cat = [0.2, −0.5, …]". How does a model learn that dog and puppy should be near each other? The answer is one famous idea.

### Idea 1: "You shall know a word by the company it keeps"

The trick that makes embeddings work: **a word's meaning is defined by the words that appear around it.** Look at the contexts:

```
"the ___ barked and wagged its tail"   → dog, puppy   (both fit)
"the ___ purred on the windowsill"     → cat, kitten  (both fit)
"the ___ launched into orbit"          → rocket       (cat/dog do NOT fit)
```

Words that show up in the *same kinds of slots* must mean similar things. So if a model is trained to **predict a word from its neighbors** (or neighbors from the word), it's forced to give "dog" and "puppy" similar internal numbers — because they need to make the same predictions. The embeddings are a *byproduct* of that prediction task: meaning falls out of "what context does this word live in."

### Idea 2: The vector is just a learned lookup table, nudged by training

Concretely, every word starts with a **random** vector in a big lookup table. During training, every time two words appear in similar contexts, gradient descent ([../ml/linear-regression.md](../ml/linear-regression.md)) nudges their vectors closer; words in different contexts drift apart:

```
start:  dog=[random]   puppy=[random]   rocket=[random]   (meaningless)
        ... train on billions of sentences, nudging vectors by context ...
end:    dog=[0.21,-0.4,...]  puppy=[0.19,-0.38,...]  rocket=[-0.7,0.6,...]
              └── dog & puppy ended up close; rocket far away ──┘
```

The famous result of this: directions in the space encode *relationships*, so vector arithmetic works — `king − man + woman ≈ queen`. The model never learned that rule; it emerged from contexts.

### Idea 3: Why modern embeddings are "contextual"

Old embeddings (Word2Vec) gave each word **one fixed** vector — so "bank" had a single blurry vector averaging riverbank and money-bank. Modern embeddings (BERT, sentence-transformers) compute the vector **using the whole sentence** via attention ([../deep-learning/transformers.md](../deep-learning/transformers.md)):

```
"river bank"   → "bank" gets a vector near {water, shore}
"bank account" → "bank" gets a DIFFERENT vector near {money, finance}
   → same word, different numbers, because context now shapes the vector
```

> 💡 **One line:** embeddings are learned numbers where similarity-in-meaning becomes closeness-in-space, trained purely from the company words keep — and modern ones recompute that vector per sentence so meaning bends to context. This is the engine behind semantic search and [RAG](../llms/rag.md): embed the query, find the nearest vectors.

---

## 3. Why It Exists

**The problem:** Neural networks need numbers. "cat" means nothing to a matrix multiply.

**What came before:** **One-hot encoding** — each word gets a vector of all zeros except a single 1 at its index. "cat" = `[1,0,0,0,...]`, "dog" = `[0,1,0,0,...]`. Problems:
- Huge vectors (vocabulary size = vector size → 50,000+ dimensions)
- "cat" and "dog" look equally different as "cat" and "spaceship" — no semantic relationship
- Completely ignores meaning

**What changed:** Word2Vec (2013) proved that you could **learn** vectors where similar words cluster together, just by training on text context. The famous result: `king - man + woman ≈ queen`. Meaning became arithmetic.

**What changed further:** Transformer models (2017+) made embeddings **contextual** — "bank" by a river gets a different vector than "bank" for money. Same word, different vectors, depending on sentence context.

---

## 4. Core Concepts

### Word Embedding

**One-line definition:** A fixed-size vector of floats that represents a word's meaning.

**Analogy:** Imagine plotting words on a 2D chart where X = "animal-ness" and Y = "size." Cat would be at (high, small), whale at (high, huge), car at (low, medium). Real embeddings have 300-1536 dimensions instead of 2, encoding hundreds of semantic features simultaneously.

**Technical explanation:** An embedding is a row in a matrix of shape `[vocab_size, embedding_dim]`. During training, these rows are updated so that words appearing in similar contexts get similar vectors.

```python
import numpy as np

# Pretend embeddings (simplified, 3D instead of 300D)
embeddings = {
    "cat":    np.array([0.2,  0.8,  0.1]),  # furry, pet, small
    "dog":    np.array([0.3,  0.7,  0.2]),  # furry, pet, medium
    "king":   np.array([0.9,  0.1,  0.8]),  # royal, human, male
    "queen":  np.array([0.9,  0.1,  0.2]),  # royal, human, female
}

# Cat and dog are closer than cat and king
```

**Common misconception:** ❌ "Embedding dimensions have human-readable meanings" → ✅ No single dimension means "animal-ness." The meanings emerge distributed across all dimensions. You can't read off what dimension 47 represents.

---

### Cosine Similarity

**One-line definition:** A measure of how similar two vectors are, based on the angle between them (not their length).

**Analogy:** Two arrows pointing in almost the same direction are "similar," even if one is longer. Cosine similarity measures the angle between directions.

```
Range: -1 to 1
  1.0  = same direction = same meaning
  0.0  = perpendicular = unrelated
 -1.0  = opposite directions = opposite meanings
```

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Or with sklearn:
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
score = cos_sim([vec_a], [vec_b])[0][0]
```

**Common misconception:** ❌ "Euclidean distance works just as well" → ✅ Cosine similarity ignores vector magnitude and only measures direction. This matters because two documents expressing the same idea at different lengths can end up with vectors of different magnitudes — cosine treats them as equally similar, Euclidean doesn't. (Footnote: if vectors are L2-normalized to unit length, cosine and Euclidean rank neighbors identically — some vector DBs exploit this and actually store normalized vectors under the hood.)

---

### Word2Vec

**One-line definition:** A 2013 algorithm that learns word vectors by predicting context words.

**Analogy:** You can guess someone's personality by the company they keep. Word2Vec figures out word meaning by the words it tends to hang out with.

**Technical explanation:** Two approaches:
- **CBOW** (Continuous Bag of Words): Predict a word from its context
- **Skip-gram**: Predict context words given a center word

```
"The cat sat on the mat"

Skip-gram training:
  Center: "cat" → predict "The", "sat"
  Center: "sat" → predict "cat", "on"

After training on millions of sentences:
  "cat" and "dog" appear in similar contexts
  → their vectors land close together
```

**The famous result:**
```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
walking - walk + swim ≈ swimming
```

**Common misconception:** ❌ "Word2Vec is what modern models use" → ✅ Word2Vec is historical context. Modern models use contextual embeddings from Transformers. Word2Vec is still useful for understanding the concept.

---

### Contextual Embeddings

**One-line definition:** Embeddings where the same word gets a different vector depending on the sentence it's in.

**Analogy:** The word "cool" means different things in "a cool breeze," "that's so cool," and "cool it down." A human understands context — contextual embeddings do the same.

```
Word2Vec (context-free):
  "bank" = [0.3, 0.5, -0.1, ...]  ← SAME vector always

BERT (contextual):
  "I went to the bank to deposit money"
    "bank" = [0.8, 0.1, 0.3, ...]   ← financial institution vector

  "I sat on the bank of the river"
    "bank" = [0.1, 0.7, -0.2, ...]  ← riverside vector
```

BERT produces different representations by attending to all surrounding tokens before computing a word's vector.

**Common misconception:** ❌ "Contextual embeddings are just bigger Word2Vec" → ✅ They're architecturally different. Word2Vec trains each word independently. BERT processes the full sentence at once; every word's vector is influenced by every other word.

---

### Sentence Embeddings

**One-line definition:** A single vector representing the meaning of an entire sentence or paragraph.

**Analogy:** Like a movie trailer — one short clip that captures the essence of the whole movie. A sentence embedding is one vector that captures the meaning of all the words together.

**Why this matters for AI engineers:** Sentence embeddings are the foundation of:
- Semantic search (find documents by meaning, not keywords)
- RAG (find relevant context for LLMs)
- Duplicate detection, clustering, recommendation

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials",    # semantically similar to above
    "What's the weather today?",         # unrelated
]

embeddings = model.encode(sentences)
# Shape: (3, 384) — each sentence → 384-dimensional vector
```

**Common misconception:** ❌ "Just average all word vectors to get a sentence embedding" → ✅ Simple averaging loses word order and context. Sentence-Transformers are trained specifically to produce good sentence-level representations.

---

## 5. How It Actually Works (Step-by-Step)

Let's build a mini semantic search in 5 steps:

```
GOAL: User asks "can't log in" → find most relevant FAQ

Step 1: Embed your knowledge base
  "How to reset password" → model.encode() → [0.2, 0.8, -0.1, ...]  (384 floats)
  "Billing questions FAQ" → model.encode() → [0.7, 0.1,  0.3, ...]
  "Contact support"       → model.encode() → [0.5, 0.3,  0.6, ...]
  Store all vectors.

Step 2: User submits query
  "can't log in" → model.encode() → [0.3, 0.7, -0.2, ...]

Step 3: Compare query vector to all stored vectors
  cosine_similarity(query, "reset password") = 0.89  ← HIGH (related!)
  cosine_similarity(query, "billing FAQ")    = 0.12  ← low (unrelated)
  cosine_similarity(query, "contact support")= 0.41  ← medium

Step 4: Return top result
  "How to reset password" wins

Step 5: Combine with LLM (RAG)
  Prompt: "Based on this FAQ: [reset password text], answer: can't log in"
  LLM generates a tailored answer
```

> 💡 **Key Insight:** The query "can't log in" never exactly matched "reset password" — but the embeddings captured their shared semantic territory. This is the power of semantic similarity over keyword matching.

---

## 6. Code in Practice

### Minimal: Generate and compare embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

a = model.encode("The dog ran quickly")
b = model.encode("A puppy sprinted fast")  # similar meaning
c = model.encode("The stock market crashed")  # unrelated

def cosine_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

print(cosine_sim(a, b))  # ~0.85 — similar
print(cosine_sim(a, c))  # ~0.10 — unrelated
```

### Practical: Semantic search over a document set

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

docs = [
    "To reset your password, click 'Forgot Password' on the login page",
    "Refunds are processed within 5-7 business days",
    "Our office hours are Monday to Friday, 9am to 5pm",
    "To upgrade your plan, go to Settings > Billing > Change Plan",
]

doc_embeddings = model.encode(docs)  # shape: (4, 384)

def search(query: str, top_k: int = 2):
    query_emb = model.encode([query])
    scores = cosine_similarity(query_emb, doc_embeddings)[0]
    ranked = sorted(zip(scores, docs), reverse=True)
    return ranked[:top_k]

results = search("I forgot my login")
for score, doc in results:
    print(f"{score:.3f}: {doc}")
# 0.812: To reset your password, click 'Forgot Password'...
# 0.203: To upgrade your plan, go to Settings > Billing...
```

### Real-world pattern: Using OpenAI embeddings API

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

vec = embed("What is machine learning?")
print(f"Vector dimensions: {len(vec)}")  # 1536
```

---

## 7. Gotchas & Pitfalls

❌ **Mixing embedding models in the same vector store** → ✅ All vectors must come from the same model. OpenAI's embeddings and Sentence-Transformers' embeddings live in completely different spaces.

❌ **Assuming cosine similarity is always the right metric** → ✅ Some vector databases (Pinecone, Weaviate) use dot product or Euclidean distance. Match the metric to how your model was trained.

❌ **Embedding single words when you need sentences** → ✅ Word-level models (Word2Vec) work on words. Sentence-Transformers work on whole sentences. Using a sentence model on individual words gives poor results.

❌ **Not normalizing vectors before using dot product** → ✅ Cosine similarity = dot product only when vectors are unit-length. Use `cosine_similarity()` from sklearn, or normalize first: `vec / np.linalg.norm(vec)`.

❌ **Assuming more dimensions = better embeddings** → ✅ OpenAI's `text-embedding-3-small` (1536 dims) often outperforms `all-MiniLM-L6-v2` (384 dims) on retrieval tasks — but the latter is free and fast. Benchmark for your use case.

❌ **Embedding at query time with the wrong model** → ✅ Use the same model for indexing documents AND for encoding queries. Switching models means re-embedding your entire database.

❌ **Confusing embedding similarity with factual correctness** → ✅ "The earth is flat" and "The earth is round" will have HIGH cosine similarity (same topic), not low. Embeddings measure topic proximity, not truth.

---

## 8. When to Use / When NOT to Use

### Use embeddings when:
- Building **semantic search** (find by meaning, not keyword)
- Building **RAG** (retrieve relevant context for an LLM)
- **Clustering** documents into topics
- Finding **duplicate or near-duplicate** content
- Building **recommendation systems** ("similar items")

### Don't use embeddings when:
- You need **exact keyword matching** — use BM25 or a search index instead
- Your task is **generation** (writing, chatting) — use a language model directly
- You're comparing **very short strings** where exact match makes more sense
- You need **real-time** responses with no precomputed index — embedding is slow for streaming

---

## 9. Related Concepts (The Map)

- **Tokenization** — the step before embedding. Text → token IDs → embeddings. You can't embed without first tokenizing.
- **BERT** — produces contextual word embeddings. When you use `[CLS]` token output for classification, you're using an embedding.
- **RAG** — entirely built on sentence embeddings. Embed documents → store in vector DB → embed query → find closest docs.
- **Vector databases** (Pinecone, ChromaDB, Weaviate) — databases optimized for storing and searching embedding vectors.
- **Fine-tuning** — can adapt embedding models to your domain. A general model might not know that "LTV" and "lifetime value" are the same in your company's context; fine-tuning fixes that.

---

## 10. Cheat Sheet

| Term | Definition |
|---|---|
| **Embedding** | Vector of floats representing meaning |
| **Word2Vec** | 2013 algorithm; learns static word vectors from context |
| **GloVe** | 2014 algorithm; uses co-occurrence statistics |
| **Contextual embedding** | Same word → different vector depending on context |
| **Sentence embedding** | One vector for an entire sentence |
| **Cosine similarity** | Angle-based similarity: 1=same, 0=unrelated, -1=opposite |
| **Embedding dimension** | Length of the vector (e.g., 384, 768, 1536) |
| **all-MiniLM-L6-v2** | Best free starting model for sentence embeddings |

**Core pattern:**
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(list_of_texts)           # (N, 384)
scores = cosine_similarity(query_emb, doc_embs)    # (1, N)
```

**Remember this:**
1. Similar meanings → close vectors (small cosine distance)
2. Always use the same model for both indexing and querying
3. Sentence embeddings (not word embeddings) power RAG and search

---

## 11. Self-Check Questions

1. Why is one-hot encoding a poor way to represent words?
2. What does `king - man + woman ≈ queen` demonstrate about embeddings?
3. Why do contextual embeddings outperform Word2Vec on most tasks?
4. You built a semantic search engine using `text-embedding-3-small`. Now you want to switch to `all-MiniLM-L6-v2` to save money. What must you do?
5. A user queries "affordable laptop" and your system returns "budget notebook computer." How is this possible without keyword overlap?

<details>
<summary>Answers</summary>

1. One-hot vectors are enormous (one dimension per vocabulary word), contain no semantic information (all words are equally distant from each other), and don't generalize — there's no way to learn that "cat" and "kitten" are related from the vectors alone.

2. It demonstrates that **meaning is arithmetic in embedding space**. Semantic relationships (gender, geography, tense) are encoded as consistent directions in the vector space. You can navigate meaning by adding and subtracting vectors.

3. Word2Vec gives every word one fixed vector regardless of context. "bank" (financial) and "bank" (river) get the same vector. Contextual embeddings process the full sentence and produce different vectors for the same word based on what surrounds it — much closer to how humans understand language.

4. You must **re-embed your entire document database** with the new model. Vectors from different models live in different spaces and are incompatible. You cannot mix models in a single vector store.

5. The embedding model maps "affordable laptop" and "budget notebook computer" to vectors that are close together in embedding space — they share the same semantic region (cheap portable computers). The model learned this from training on text where these phrases appear in similar contexts. Embeddings capture meaning, not surface-level word overlap.

</details>

---

## 12. Go Deeper

- **[Sentence-Transformers documentation](https://www.sbert.net/)** — the go-to library for sentence embeddings. Start with the "Getting Started" guide. Best for understanding what pre-trained sentence models are available.
- **["Efficient Estimation of Word Representations in Vector Space" (Mikolov 2013)](https://arxiv.org/abs/1301.3781)** — the original Word2Vec paper. Only 9 pages, very readable. Understand the idea that kicked off the embeddings revolution.
- **[Jay Alammar's "The Illustrated Word2Vec"](https://jalammar.github.io/illustrated-word2vec/)** — the best visual explanation of how Word2Vec training works. Build a real intuition with animated diagrams.
- **[OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)** — practical guide for using the API. Good for understanding dimensions, distance functions, and pricing.
- **[Building a Semantic Search Engine with LangChain + ChromaDB](https://python.langchain.com/docs/use_cases/question_answering/)** — hands-on tutorial for the full RAG pipeline. Apply embeddings to a real project in under an hour.
