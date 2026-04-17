# Sequence-to-Sequence & Summarization

## 1. TL;DR

Seq2Seq models take **text in and produce different text out** — different length, possibly different language. The encoder reads and understands the input; the decoder generates the output word by word. Use **BART for summarization**, **MarianMT for translation**, **T5 for any text-to-text task**. Control output length with `max_length`/`min_length` and quality with `num_beams`. This architecture is the direct ancestor of GPT — which is essentially a decoder-only seq2seq.

---

## 2. The Mental Model

> 💡 **Think of it like this:** Seq2Seq is a **translator at the UN** — they hear the full speech first, understand it completely, then speak a new version in a different language.

A bad translator translates word-by-word as they hear it. A good translator listens to the whole sentence, builds a mental model of meaning, then speaks a natural translation. Seq2Seq does exactly that: encode the meaning, then decode a new expression of it.

| Real world | Technical concept |
|---|---|
| UN translator listening to a speech | Encoder reading the full input |
| Mental note capturing the meaning | Context vector (or encoder hidden states) |
| Speaking the translated version word by word | Decoder generating output token by token |
| Translator looking back at their notes for complex phrases | Attention mechanism consulting encoder states |
| English → French | Input language → output language |

---

## 3. Why It Exists

**The problem:** Some NLP tasks can't be solved by classification — the output isn't a label, it's new text. Summarizing an article can't output "sports" or "positive" — it must generate actual sentences.

**What came before:** RNN-based models (2014, Sutskever et al.) introduced encoder-decoder architectures for translation. They worked but had a critical flaw: the encoder compressed the entire input into a *single fixed-size vector* — a bottleneck that made long sequences hard.

**What changed:** The attention mechanism (Bahdanau 2015) let the decoder look back at specific parts of the encoder's output at each generation step. Then the full Transformer architecture (2017) replaced RNNs entirely, letting both encoder and decoder process all positions in parallel. This gave us T5, BART, and every modern translation/summarization model.

---

## 4. Core Concepts

### Encoder-Decoder Architecture

**One-line definition:** Two separate Transformer stacks — encoder reads input, decoder generates output, connected via cross-attention.

**Analogy:** A relay race with a baton. The encoder (first runner) processes the input and passes a rich representation (the baton) to the decoder (second runner) who generates the output.

```
Architecture overview:

ENCODER (bidirectional, like BERT):
  Input: "The cat sat on the mat"
  Process: all tokens attend to all tokens
  Output: hidden states for every input token

CROSS-ATTENTION (the connection):
  Decoder looks at encoder's hidden states
  At each decoder step: "which encoder positions matter most right now?"

DECODER (autoregressive, like GPT):
  Input: [start] token
  Process: generate one token, add to sequence, repeat
  Output: "Le chat s'est assis sur le tapis"
```

**Common misconception:** ❌ "Encoder-decoder and decoder-only are the same" → ✅ A decoder-only model (GPT) uses only its own previous outputs as context. An encoder-decoder model uses a separate encoder to produce a dedicated input representation — better for tasks where input and output are very different (translation, summarization).

---

### Cross-Attention

**One-line definition:** The mechanism that lets each decoder step look at any part of the encoder's output.

**Analogy:** A translator reading back through their notes at each word they speak — they focus on the source phrase that's most relevant to the word they're currently generating.

```
Generating "chat" (French for "cat"):
  Decoder asks: "which encoder positions are relevant?"
  Attention weights:
    "The":  0.02
    "cat":  0.87   ← high attention
    "sat":  0.05
    "on":   0.02
    "the":  0.01
    "mat":  0.03

  Decoder uses mostly "cat"'s encoder representation to generate "chat"
```

The decoder doesn't just get one vector — it gets a dynamic blend of all encoder positions, weighted by relevance at each generation step.

**In Q/K/V terms (the vocabulary you'll see in papers and interviews):**
- **Query (Q)** = "what am I looking for right now?" → comes from the **decoder's** current position
- **Key (K)** = "what am I about?" → comes from **encoder** outputs (one per input token)
- **Value (V)** = "what content do I carry?" → also comes from **encoder** outputs

The decoder's Query is matched against every encoder Key (dot product → softmax = attention weights), then those weights blend the encoder Values into one context vector. That's the whole mechanism.

**Common misconception:** ❌ "Cross-attention is the same as self-attention" → ✅ Self-attention: Q, K, V all come from the same sequence. Cross-attention: Q comes from the decoder, K and V come from the encoder. That's the only architectural difference — but it's what lets the decoder "look at" the input while generating.

---

### Extractive vs. Abstractive Summarization

**One-line definition:** Extractive copies sentences from the source; abstractive generates new sentences with the same meaning.

**Analogy:** Extractive = highlighting sentences in a textbook. Abstractive = reading the chapter and explaining it in your own words.

```
Original article:
  "Amazon reported record quarterly earnings of $143 billion.
   The company attributed growth to AWS cloud services.
   CEO Andy Jassy announced expansion into healthcare markets.
   Office renovations were completed at the Seattle headquarters."

Extractive summary (copied sentences):
  "Amazon reported record quarterly earnings of $143 billion.
   CEO Andy Jassy announced expansion into healthcare markets."

Abstractive summary (generated new text):
  "Amazon achieved record $143B quarterly earnings driven by AWS growth,
   with CEO Jassy announcing a new push into healthcare."
```

Modern models (BART, T5, GPT) do abstractive summarization. It's harder — the model must understand and paraphrase — but produces more natural results.

**Common misconception:** ❌ "Abstractive is always better" → ✅ For legal/medical summaries requiring exact wording, extractive is safer. Abstractive models can hallucinate facts that weren't in the original. Use abstractive when fluency matters; use extractive when fidelity matters.

---

### Decoding Strategies

**One-line definition:** Algorithms that decide which token to pick at each step of generation.

**Analogy:** Choosing your next word when writing a sentence — do you always pick the most obvious word (greedy), or consider multiple possible continuations (beam search), or pick something creative (sampling)?

```
"The cat sat on the ___"
Next token probabilities: mat(0.40), floor(0.30), rug(0.20), chair(0.10)

Greedy:        Always pick highest → "mat" every time, can be repetitive

Beam search:   Keep top-K sequences at each step:
  beam 1: "mat" → "." (prob: 0.40 × 0.70 = 0.28)
  beam 2: "floor" → "." (prob: 0.30 × 0.60 = 0.18)
  Pick best overall sequence → usually better quality

Sampling:      Draw randomly from distribution
  Sometimes "mat", sometimes "floor", sometimes "rug" → diverse outputs

Top-p (nucleus): Sample only from tokens summing to p=0.9 probability mass
  [mat(0.4) + floor(0.3) + rug(0.2)] = 0.9 → only sample from these 3
```

```python
# Deterministic (best for summarization)
output = model.generate(input_ids, num_beams=4, do_sample=False)

# Creative (best for stories/brainstorming)
output = model.generate(input_ids, do_sample=True, top_p=0.9, temperature=0.8)
```

---

### ROUGE Metric

**One-line definition:** A family of metrics measuring overlap between a generated summary and human reference summaries.

**Analogy:** Grading an essay by counting how many words and phrases from the answer key appear in the student's essay.

```
Reference: "The cat sat on the mat"
Generated: "The cat is on the mat"

ROUGE-1 (unigram overlap):
  Matches: The, cat, on, the, mat  → 5/6 = 0.83

ROUGE-2 (bigram overlap):
  Reference bigrams: [the cat], [cat sat], [sat on], [on the], [the mat]
  Generated bigrams: [the cat], [cat is], [is on], [on the], [the mat]
  Matches: [the cat], [on the], [the mat] → 3/5 = 0.60

ROUGE-L (longest common subsequence):
  LCS = "The cat * on the mat" → 5 tokens → 5/6 = 0.83
```

**Common misconception:** ❌ "High ROUGE = good summary" → ✅ ROUGE measures overlap, not correctness. A summary can score poorly on ROUGE while being more accurate and readable than the reference. ROUGE is a proxy, not ground truth. Always complement with human evaluation.

---

## 5. How It Actually Works (Step-by-Step)

Let's trace summarizing a paragraph with BART:

```
INPUT: "Amazon rainforest produces 20% of world's oxygen, spans 5.5M km².
        Deforestation accelerating. Scientists warn of tipping point that
        could convert it from carbon sink to carbon source."

Step 1: Tokenize input
  ~ 45 tokens from the input text

Step 2: Encoder forward pass (bidirectional)
  All 45 input tokens attend to each other simultaneously
  Each token gets a rich contextual vector
  "tipping point" attends to "carbon" and "deforestation" → understands the connection

Step 3: Start decoding with [BOS] token
  Decoder input: [BOS]
  Cross-attention: looks at all encoder outputs → which parts matter for the first word?
  Attention focuses on most important concepts: "Amazon", "oxygen", "deforestation"
  Generate: "The"

Step 4: Autoregressive generation continues
  Decoder input: [BOS, "The"]
  Generate: "Amazon"

  Decoder input: [BOS, "The", "Amazon"]
  Generate: "rainforest"
  ...

Step 5: Beam search tracks top-4 candidates at each step
  Candidate 1: "The Amazon rainforest produces..."
  Candidate 2: "The Amazon produces..."
  Candidate 3: "Deforestation of the Amazon..."
  Candidate 4: "Scientists warn the Amazon..."
  → Keeps whichever sequence has highest cumulative probability

Step 6: Stop when [EOS] is generated or max_length reached

OUTPUT: "The Amazon rainforest produces 20% of world's oxygen and faces
         accelerating deforestation that could trigger a climate tipping point."
```

> 💡 **Key Insight:** The summary "trigger a climate tipping point" never appeared word-for-word in the input. BART generated a paraphrase by understanding the meaning, not copying the text. That's what makes it abstractive.

---

## 6. Code in Practice

### Minimal: Summarize text

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
The Amazon rainforest, often called the lungs of the Earth, produces about 20%
of the world's oxygen. Spanning 5.5 million square kilometers across nine countries,
it is home to 10% of all species on Earth. However, deforestation has been
accelerating, with scientists warning that the forest could pass a tipping point,
transforming from a carbon sink into a carbon source and dramatically accelerating
climate change.
"""

result = summarizer(article, max_length=80, min_length=30, do_sample=False)
print(result[0]['summary_text'])
```

### Practical: Translation with MarianMT

```python
from transformers import pipeline

translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

texts = [
    "Machine learning is transforming every industry.",
    "The model predicts the next token based on context.",
]

results = translator(texts)
for r in results:
    print(r['translation_text'])
# L'apprentissage automatique transforme chaque industrie.
# Le modèle prédit le jeton suivant en fonction du contexte.
```

### Real-world pattern: T5 for multiple tasks with one model

```python
from transformers import pipeline

t5 = pipeline("text2text-generation", model="t5-base")

# Summarization — prefix changes the task
summary = t5("summarize: " + long_article, max_length=60)
print(summary[0]['generated_text'])

# Translation
translation = t5("translate English to German: Hello, how are you?")
print(translation[0]['generated_text'])  # "Hallo, wie geht es Ihnen?"

# Question answering
answer = t5("question: Who is CEO of Apple? context: Tim Cook is the CEO of Apple Inc.")
print(answer[0]['generated_text'])  # "Tim Cook"

# All three use the SAME model — just change the prefix!
```

---

## 7. Gotchas & Pitfalls

❌ **Summary cuts off mid-sentence** → ✅ `max_length` is too short. Increase it. Also set `min_length` to prevent degenerate 1-word summaries.

❌ **Repetitive output** → ✅ Greedy decoding falls into loops ("the the the..."). Use `no_repeat_ngram_size=3` to prevent repeating 3-grams, or use beam search.

❌ **Hallucinated facts in summary** → ✅ Abstractive models can generate plausible-sounding facts not in the original. Always verify claims in summaries for high-stakes use cases.

❌ **Summarizing text longer than the model's context window** → ✅ Most seq2seq models have ~1024 token input limits. For longer documents: chunk into sections → summarize each → summarize the summaries (hierarchical summarization).

❌ **Using the wrong model for translation** → ✅ Check the model card for supported language pairs. `Helsinki-NLP/opus-mt-en-fr` only does English→French. For multilingual, use mBART or a modern LLM.

❌ **Expecting ROUGE to tell you if your summary is good** → ✅ ROUGE measures lexical overlap with a reference summary. It can't detect hallucinations, incorrect facts, or missing key information. Treat ROUGE as a rough proxy.

❌ **Using `do_sample=True` for factual summarization** → ✅ Sampling introduces randomness — each call gives a different summary. For factual tasks, use `do_sample=False` with `num_beams=4` for deterministic, high-quality output.

---

## 8. When to Use / When NOT to Use

### Use Seq2Seq when:
- **Summarization** — condense long documents into short ones
- **Translation** — convert between languages
- **Paraphrasing** — rewrite text while preserving meaning
- **Question generation** — create questions from text (exam preparation)
- **Data augmentation** — paraphrase training examples to create more data

### Don't use Seq2Seq when:
- **Text classification** — use BERT (simpler, faster, cheaper)
- **Named entity extraction** — use NER pipeline or LLM prompting
- **Conversational AI** — use decoder-only LLMs (GPT, Claude) with chat formatting
- **Very long inputs** (>50K tokens) — use specialized long-document models or chunking + LLM APIs
- **Exact wording required** — Seq2Seq is generative and may change phrasing; use extractive methods instead

---

## 9. Related Concepts (The Map)

- **BERT** — the encoder stack in seq2seq models. BART's encoder is essentially a BERT-style bidirectional encoder. Understanding BERT = understanding the first half of seq2seq.
- **GPT** — the decoder-only special case. GPT is a seq2seq model where the encoder is removed and the decoder learns to both understand and generate from the same context.
- **Attention mechanism** — the key innovation that made seq2seq practical. Before attention, all encoder state was compressed to one vector. Attention lets the decoder look at every encoder position.
- **Fine-tuning** — you can fine-tune BART or T5 on domain-specific data. Domain-specific summarization (medical, legal) often benefits from task-specific fine-tuning.
- **LLM summarization** — modern practice: just send the text to Claude/GPT-4 and ask for a summary. Much simpler, often better quality, but costs money per call and has context limits.

---

## 10. Cheat Sheet

| Model | Creator | Best For | Input Limit |
|---|---|---|---|
| **BART-large-CNN** | Meta | News summarization | 1024 tokens |
| **Pegasus** | Google | Multi-domain summarization | 1024 tokens |
| **T5-base/large** | Google | Any text-to-text task | 512 tokens |
| **MarianMT** | Helsinki NLP | Translation (200+ pairs) | 512 tokens |
| **mBART** | Meta | Multilingual translation | 1024 tokens |

**Summarization control parameters:**
```python
summarizer(
    text,
    max_length=130,       # output token limit
    min_length=30,        # output minimum length
    num_beams=4,          # beam search (quality vs speed)
    no_repeat_ngram_size=3,  # prevent repetition
    do_sample=False,      # deterministic
)
```

**ROUGE quick reference:**
```
ROUGE-1: unigram overlap (word level)
ROUGE-2: bigram overlap (phrase level)
ROUGE-L: longest common subsequence
All range 0-1, higher = more overlap with reference
```

**Remember this:**
1. Encoder = understand input; Decoder = generate output; Cross-attention = connection
2. Use `num_beams=4, do_sample=False` for quality summarization
3. Abstractive can hallucinate — always verify important facts

---

## 11. Self-Check Questions

1. What problem did the attention mechanism solve in early seq2seq models?
2. Why does BART outperform GPT for summarization despite both being Transformer-based?
3. What's the difference between extractive and abstractive summarization, and when would you choose each?
4. Your summarization pipeline keeps generating repetitive text like "the company the company the company." What parameter fixes this?
5. Why is ROUGE an imperfect metric for summarization quality?

<details>
<summary>Answers</summary>

1. Early seq2seq models compressed the entire input into a single fixed-size vector — a severe bottleneck for long inputs. The model had to forget details as sequences got longer. Attention allows the decoder to look back at any encoder position at each generation step, creating a dynamic, content-based access to the full input regardless of length.

2. BART was specifically designed and trained for text generation tasks using a denoising pre-training objective — it learns to reconstruct corrupted documents, which is closely related to summarization. GPT was trained for next-token prediction, making it a decoder-only model with no dedicated encoder. BART's encoder-decoder architecture gives it a dedicated "understand the input" component (encoder) and a "generate output" component (decoder) — better suited for transforming one text into another.

3. **Extractive**: copies sentences verbatim from the source. Safe for legal/medical contexts where exact wording matters; guaranteed factual accuracy but may be choppy. **Abstractive**: generates new sentences paraphrasing the key ideas. More fluent and concise; better reading experience, but risks hallucinating details not in the source. Choose extractive for safety-critical applications; choose abstractive for user-facing summaries where readability matters.

4. Set `no_repeat_ngram_size=3`. This prevents the model from generating any 3-gram (sequence of 3 tokens) that has already appeared in the output. It's a simple but effective heuristic for stopping repetition loops.

5. ROUGE measures lexical overlap with a human-written reference summary. It has several failure modes: (1) a factually incorrect summary can score high ROUGE if it copies the same words; (2) a brilliant paraphrase can score low ROUGE even if it's more accurate than the reference; (3) ROUGE can't detect hallucinations — invented facts will match the reference on other words and still score well. Human evaluation or factuality-specific metrics (like FactCC) are needed for reliable summarization evaluation.

</details>

---

## 12. Go Deeper

- **["Attention Is All You Need" (Vaswani 2017)](https://arxiv.org/abs/1706.03762)** — the paper that introduced the full Transformer architecture. The encoder-decoder design in seq2seq comes directly from here. Read Sections 1-3 and Figure 1.
- **["BART: Denoising Sequence-to-Sequence Pre-training" (Lewis 2019)](https://arxiv.org/abs/1910.13461)** — explains how BART was trained and why denoising pre-training is ideal for summarization. Short, 9 pages, very accessible.
- **[Jay Alammar's "Visualizing Neural Machine Translation"](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)** — the best animated explanation of encoder-decoder + attention. Watch the GIFs to build real intuition.
- **[Hugging Face Summarization Guide](https://huggingface.co/docs/transformers/tasks/summarization)** — step-by-step fine-tuning of BART on custom summarization data. Covers ROUGE evaluation and training setup.
- **["Exploring the Limits of Transfer Learning with T5" (Raffel 2020)](https://arxiv.org/abs/1910.10683)** — explains T5's "everything is text-to-text" framing. Fascinating read for understanding why T5 can handle any NLP task with the same model/format.
