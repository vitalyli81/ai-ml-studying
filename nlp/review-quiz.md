# NLP Mixed Review Quiz — 25 Scenario Questions

> **Why this file exists:** each doc tests only itself, but interviews and real work test whether you can *choose the right model/approach* and *debug NLP pipelines* from symptoms. Do 5 at a time, out loud, *before* opening the answer.
>
> Score yourself: ≥20/25 → you're ready for the LLM/AI-engineering phase. Misses → re-read that doc's "Build the Intuition From Zero" section.

---

## Round 1 — Choosing the Right Approach

**1.** Classify 200K support tickets/day into 6 fixed categories. The PM suggests calling GPT-4 for each. What do you propose, and what's the cost argument?

<details><summary>Answer</summary>

Fine-tune DistilBERT (`AutoModelForSequenceClassification`, num_labels=6, lr=2e-5, 3 epochs). At 200K calls/day, LLM API costs and 500ms+ latency dominate; a fine-tuned encoder serves in milliseconds, 10-100× cheaper per prediction, with higher accuracy on a fixed label set. Use the LLM for prototyping and label design, not the production hot path. ([text-classification.md](text-classification.md), [fine-tuning-nlp.md](fine-tuning-nlp.md))
</details>

**2.** "Find documents about login problems even when they say 'credentials' or 'sign-in failure'" — keyword search keeps missing them. Name the technique and the 3-step pipeline.

<details><summary>Answer</summary>

Semantic search with sentence embeddings: (1) embed every document with a sentence-transformer (e.g., all-MiniLM-L6-v2) and store the vectors, (2) embed the query with the *same* model, (3) rank by cosine similarity and return top-k. Meaning proximity replaces keyword overlap — that's also the retrieval half of RAG. ([embeddings.md](embeddings.md))
</details>

**3.** Extract drug names and dosages from clinical notes. The standard `pipeline("ner")` performs terribly. Why, and what are your two realistic options?

<details><summary>Answer</summary>

Domain shift: stock NER models are trained on news/web text with PER/ORG/LOC labels — they've never seen clinical vocabulary or your entity types (DRUG, DOSAGE). Options: (1) LLM prompting with a JSON schema (zero training data, slower/costlier per call), (2) fine-tune a domain model (BioBERT/clinical BERT) on labeled medical NER data for high-volume production. ([ner.md](ner.md))
</details>

**4.** Summarize legal contracts for lawyers who must quote exact clauses. Abstractive BART or extractive? Why does the choice matter here specifically?

<details><summary>Answer</summary>

Extractive. Abstractive models paraphrase — and can hallucinate plausible-sounding clauses that aren't in the contract, which is disqualifying in legal contexts. Extractive copies real sentences verbatim: less fluent, but every quoted clause is guaranteed to exist. Fidelity beats fluency when wording is load-bearing. ([seq2seq-summarization.md](seq2seq-summarization.md))
</details>

**5.** Your team needs: (a) a chatbot, (b) ticket classification, (c) English→German translation. Map each to encoder-only / decoder-only / encoder-decoder, with one sentence of justification each.

<details><summary>Answer</summary>

(a) Decoder-only (GPT/Claude) — generation, token-by-token, causal attention. (b) Encoder-only (BERT-family) — understanding task with full text upfront; bidirectional attention + [CLS] head. (c) Encoder-decoder (MarianMT/T5) — read one sequence fully, emit a different one; cross-attention bridges them. Architecture follows task shape. ([README.md](README.md), [bert.md](bert.md), [gpt-decoder-models.md](gpt-decoder-models.md))
</details>

---

## Round 2 — Debugging Pipelines

**6.** A teammate loads `bert-base-uncased`'s tokenizer with a RoBERTa model. Nothing crashes, but predictions are garbage. Explain the silent failure.

<details><summary>Answer</summary>

Tokenizers and models are paired by vocabulary: token-to-ID mappings, special tokens (`[CLS]`/`[SEP]` vs `<s>`/`</s>`), and subword algorithms (WordPiece vs byte-level BPE) all differ. The IDs produced by BERT's tokenizer point at completely different entries in RoBERTa's embedding table — every input is effectively scrambled, and nothing raises an error. Always `AutoTokenizer.from_pretrained(<same checkpoint>)`. ([huggingface.md](huggingface.md), [tokenization.md](tokenization.md))
</details>

**7.** Batched inference returns nonsense for short sentences but works for the longest one in each batch. What's missing?

<details><summary>Answer</summary>

The attention mask. Shorter sequences get padded to the batch max; without `attention_mask`, the model attends to [PAD] tokens as real content — corrupting exactly the padded (short) examples. Use `tokenizer(texts, padding=True, ...)`, which builds the mask, and pass it through. ([tokenization.md](tokenization.md))
</details>

**8.** Your NER output is `[{'word': 'New', 'entity': 'B-LOC'}, {'word': '##York', 'entity': 'I-LOC'}]` — unusable fragments. Two fixes, one for inference and one for training?

<details><summary>Answer</summary>

Inference: `pipeline("ner", grouped_entities=True)` merges B-/I- subword runs into spans ("New York", LOC). Training: align word-level labels to subword tokens — real tag on each word's first subword, `-100` on continuations (via `word_ids()`), so the loss ignores fragments. ([ner.md](ner.md))
</details>

**9.** Fine-tuned sentiment model: 97% train accuracy, 64% validation. List the diagnosis and the three highest-impact fixes for an NLP fine-tune specifically.

<details><summary>Answer</summary>

Overfitting — memorized the training reviews. Fixes, in order: (1) fewer epochs — fine-tuning converges in 2-3; use `load_best_model_at_end=True` with per-epoch eval, (2) more/augmented data (paraphrase examples), (3) regularization — `weight_decay=0.01`, possibly a smaller model (DistilBERT over BERT-large). Also verify the LR is ~2e-5, not higher. ([fine-tuning-nlp.md](fine-tuning-nlp.md))
</details>

**10.** A summarizer outputs "the company said the company said the company..." and another run cuts off mid-sentence. Name the two parameters to fix each symptom.

<details><summary>Answer</summary>

Repetition: `no_repeat_ngram_size=3` (and `num_beams=4` instead of greedy). Mid-sentence cutoff: `max_length` too small — raise it (and set `min_length` to avoid degenerate one-liners). For factual summaries also keep `do_sample=False`. ([seq2seq-summarization.md](seq2seq-summarization.md))
</details>

---

## Round 3 — Architecture Internals

**11.** Why can BERT not power a chatbot, and why can GPT not match a fine-tuned BERT on bulk classification? One mechanism each.

<details><summary>Answer</summary>

BERT: encoder-only, bidirectional attention, trained on masked-word prediction — there's no autoregressive decoder to emit the next token, so it cannot generate replies. GPT: it *can* classify via prompting, but per-call it runs a billion-parameter generator where a 66M-parameter encoder with a dedicated head is faster, cheaper, and usually more accurate on a fixed label set. Understanding ≠ generation; pick the pole that matches the task. ([bert.md](bert.md), [gpt-decoder-models.md](gpt-decoder-models.md))
</details>

**12.** Explain masked language modeling and why next-word prediction couldn't train a bidirectional model.

<details><summary>Answer</summary>

MLM hides ~15% of tokens and trains the model to predict them from the full surrounding context — both sides. If you trained *next-word* prediction with bidirectional attention, the model could simply look ahead and read the answer; the task would collapse. Masking makes "use both directions" non-cheatable, forcing genuine contextual understanding. ([bert.md](bert.md))
</details>

**13.** In a translation model generating French token-by-token: where do the Query, Key, and Value in cross-attention come from, and what question does each answer?

<details><summary>Answer</summary>

Q from the decoder's current position ("what do I need from the source right now?"). K and V from the encoder's outputs — one per source token (K: "what am I about?", V: "the content I carry"). Q·K scores → softmax weights → blend of V's = the source context for emitting this output token. Self-attention is the same math with Q, K, V all from one sequence. ([seq2seq-summarization.md](seq2seq-summarization.md))
</details>

**14.** Why is the first token of a long-prompt LLM response slow, but every token after fast? Name the mechanism and one way providers let you exploit it for cost.

<details><summary>Answer</summary>

Prefill builds the KV cache for the whole prompt — O(N²) in prompt length — before the first token can emerge. After that, each new token only computes attention against cached K/Vs (one step's work). Providers expose prompt caching: pay full price once to build the KV for a large fixed system prompt, then a fraction on reuse — a major lever for RAG/agent systems. ([gpt-decoder-models.md](gpt-decoder-models.md))
</details>

**15.** Zero-shot classification gives you scores for labels the model never saw. What's actually happening under the hood?

<details><summary>Answer</summary>

It's an NLI model: classification is reframed as entailment — "premise: <your text>; hypothesis: this text is about <label>." The model scores entailment for each candidate label, and the entailment skill (learned from NLI training data) generalizes to arbitrary label names. ([text-classification.md](text-classification.md))
</details>

---

## Round 4 — Embeddings & Retrieval

**16.** Your vector search returns nonsense after a teammate "upgraded" the query encoder from MiniLM to OpenAI embeddings, keeping the existing index. What law did they break?

<details><summary>Answer</summary>

Vectors from different models live in different, incompatible spaces — a MiniLM document vector and an OpenAI query vector can't be meaningfully compared (they don't even share dimensions: 384 vs 1536). Same model for indexing AND querying, always; switching models means re-embedding the whole corpus. ([embeddings.md](embeddings.md))
</details>

**17.** Why does Word2Vec fundamentally fail on "I deposited cash at the bank" vs "I sat on the bank of the river," and what fixed it?

<details><summary>Answer</summary>

Word2Vec assigns each word ONE static vector — "bank" gets a single blurry average of riverbank and money-bank. Contextual embeddings (BERT-style) compute the vector per sentence via attention, so the same word lands near {money, finance} in one sentence and {water, shore} in the other. ([embeddings.md](embeddings.md))
</details>

**18.** A teammate computes sentence embeddings by averaging Word2Vec word vectors and wonders why search quality is mediocre. Two reasons, and the fix?

<details><summary>Answer</summary>

Averaging (1) destroys word order and negation ("not good" ≈ "good"), and (2) uses context-free word vectors to begin with. Fix: a sentence-transformer — a BERT-family model trained contrastively to produce sentence-level vectors whose cosine geometry actually reflects sentence meaning. ([embeddings.md](embeddings.md), [bert.md](bert.md))
</details>

**19.** "The earth is flat" and "The earth is round" — high or low cosine similarity? What does this imply for RAG systems?

<details><summary>Answer</summary>

High — same topic, same vocabulary territory. Embeddings measure topical proximity, not truth. Implication: retrieval will happily surface wrong/contradictory passages that are on-topic; RAG needs the LLM (and ideally source curation) to handle factuality — the retriever can't. ([embeddings.md](embeddings.md))
</details>

**20.** Estimate: a 300-page book is roughly how many tokens, and will it fit in a 200K context window? Show the arithmetic rule you used.

<details><summary>Answer</summary>

Rule: 1 token ≈ 0.75 words. A 300-page book ≈ 75-100K words ≈ 100-130K tokens — fits in 200K with room for the response. (Equivalently ~4 chars/token. Code is denser in tokens per character than prose.) Token math = cost math; always estimate before sending. ([tokenization.md](tokenization.md))
</details>

---

## Round 5 — Fine-Tuning & Production

**21.** You have 8K labeled examples and a fixed task. Sketch the full Hugging Face fine-tuning recipe — model class, the 4 critical TrainingArguments, and what you save at the end.

<details><summary>Answer</summary>

`AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=N)`; tokenize with the matching tokenizer (`batched=True`). Critical args: `learning_rate=2e-5`, `num_train_epochs=3`, `eval_strategy="epoch"` + `save_strategy="epoch"`, `load_best_model_at_end=True`. Save BOTH `trainer.save_model()` and `tokenizer.save_pretrained()` to the same directory. ([huggingface.md](huggingface.md), [fine-tuning-nlp.md](fine-tuning-nlp.md))
</details>

**22.** Your startup wants Llama-3-8B to follow its house style, on one RTX 3090 (24 GB). Walk the memory math: why full fine-tuning is impossible and what configuration works.

<details><summary>Answer</summary>

Full fine-tuning: ~16 GB weights (fp16) + gradients + Adam states ≈ 60-80 GB — far beyond 24 GB. LoRA freezes all base weights (no gradients/optimizer states for them) and trains rank-8 adapters (~0.05% of params) → ~16 GB. QLoRA additionally quantizes the frozen base to 4-bit → ~6 GB, comfortably on the 3090. `LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"])` + `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`. ([fine-tuning-nlp.md](fine-tuning-nlp.md))
</details>

**23.** Product asks the model to "know our 2026 pricing and answer in our support voice." Which half is a fine-tuning problem and which is a RAG problem?

<details><summary>Answer</summary>

Support *voice* (format, tone, behavior) → fine-tuning — it changes HOW the model responds. 2026 *pricing* (facts that change) → RAG — retrieval injects WHAT it should know, and updating a doc beats retraining every price change. The classic split: fine-tuning is for form, RAG is for facts. ([fine-tuning-nlp.md](fine-tuning-nlp.md))
</details>

**24.** Walk "I love NLP!" from raw string to a sentiment label — every representation it passes through, naming the component responsible for each hop.

<details><summary>Answer</summary>

String → tokens (`["[CLS]","i","love","nl","##p","!","[SEP]"]` — tokenizer) → token IDs (vocabulary lookup) → embedding vectors (embedding table) → contextual vectors (encoder layers, bidirectional attention) → one [CLS] summary vector → logits (classification head, 768→2) → softmax probabilities → label. Every NLP pipeline is some variation of this chain. ([tokenization.md](tokenization.md), [bert.md](bert.md), [text-classification.md](text-classification.md))
</details>

**25.** Design call: new feature needs to label incoming emails with *changing* category definitions (product team edits them weekly), low volume (~500/day). Fine-tune or prompt? Defend with three factors.

<details><summary>Answer</summary>

Prompt an LLM (zero/few-shot). (1) Categories change weekly — prompts update in minutes; fine-tuned models need relabeling + retraining per change. (2) 500/day is trivial volume — API cost is negligible, so BERT's per-call savings never amortize. (3) No labeled dataset exists for the new definitions, and <100 examples per class makes fine-tuning overfit anyway. Revisit fine-tuning only if categories stabilize AND volume grows 100×. ([text-classification.md](text-classification.md), [fine-tuning-nlp.md](fine-tuning-nlp.md))
</details>

---

## Scoring yourself

| Score | Verdict |
|---|---|
| 23–25 | Ready for the LLM/AI-engineering phase — this folder's concepts are the vocabulary of that work. Keep [flashcards.md](flashcards.md) on a weekly loop. |
| 18–22 | Solid. Re-read the "Build the Intuition From Zero" sections you missed; retake in 3 days. |
| < 18 | Foundations still forming — redo the per-doc self-checks first, then return here. |
