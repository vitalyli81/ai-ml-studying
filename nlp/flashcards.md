# NLP Flashcards — Spaced Repetition Deck

> **How to use this file:** Don't read it — *quiz yourself with it.* Cover the **A:** line, answer out loud, then check. Review misses the same day, after 2 days, and after a week (see [README.md](README.md) → How to Study This Folder). Each `Q:`/`A:` pair is one Anki card.

---

## NLP Fundamentals

**Q:** Every NLP system reduces to which three moves?
**A:** Text → numbers (tokenization + embeddings), numbers → numbers (a model transforms them), numbers → text or label (decode/classify back out).

**Q:** The three Transformer paradigms and their task shapes?
**A:** Encoder-only (BERT): input → label, understanding. Decoder-only (GPT): input → more text, generation. Encoder-decoder (T5/BART): input → different text, transform (translate, summarize). Match the architecture to the task shape first.

**Q:** The modern NLP workflow in three steps?
**A:** (1) Pick a pretrained model from Hugging Face, (2) fine-tune on your data or use it zero-shot, (3) done. You almost never build from scratch.

**Q:** Word2Vec era vs Transformer era — the one-word difference in embeddings?
**A:** Context. Word2Vec: one fixed vector per word ("bank" is always the same). Transformers: contextual — the vector for "bank" depends on the whole sentence.

---

## Tokenization

**Q:** Why do word-level and character-level tokenization both fail?
**A:** Word-level: vocabulary explodes and any unseen word becomes `[UNK]` (information lost). Character-level: tiny vocab but enormous sequences with meaningless pieces. Subword keeps common words whole and breaks rare words into known pieces — no unknowns, reasonable length.

**Q:** How does BPE learn its vocabulary?
**A:** Start from characters; repeatedly merge the most frequent adjacent pair into a new token; stop at the target vocab size (~50K). Frequent strings "graduate" into single tokens; splits happen where the learned merges run out.

**Q:** What is the attention mask and what happens without it?
**A:** A 1/0 array marking real tokens vs padding. Without it the model attends to [PAD] tokens as if they were content and produces garbage on batched, padded input.

**Q:** Rough token math for English?
**A:** 1 token ≈ 4 characters ≈ 0.75 words. Token count drives both API cost and context-window usage — measure before sending.

**Q:** Name three special tokens and their jobs.
**A:** `[CLS]` — sequence start, carries the classification summary (BERT). `[SEP]` — separates segments. `[PAD]` — filler for equal-length batches. (GPT uses `<|endoftext|>`.) The model was trained expecting them — let `tokenizer()` add them.

**Q:** The #1 tokenizer rule?
**A:** Always load the tokenizer that matches your model: `AutoTokenizer.from_pretrained("same-model-name")`. BERT (WordPiece) and GPT (BPE) vocabularies are incompatible — mixing them silently produces garbage.

---

## Embeddings

**Q:** What is an embedding, in one sentence?
**A:** A learned vector of floats where similarity in meaning becomes closeness in space — "dog" and "puppy" end up neighbors because they appear in the same kinds of contexts.

**Q:** How does training produce meaningful vectors from random ones?
**A:** Train the model to predict a word from its neighbors (or vice versa). Words used in the same contexts must make the same predictions, so gradient descent pushes their vectors together. Meaning is a byproduct of the prediction task.

**Q:** What does `king − man + woman ≈ queen` show?
**A:** Relationships are encoded as consistent *directions* in the space — meaning is arithmetic. Nobody programmed it; it emerged from context statistics.

**Q:** Why cosine similarity for text vectors?
**A:** It measures angle (direction), ignoring magnitude — so a short and a long document about the same topic compare as similar. (If vectors are L2-normalized, cosine and Euclidean rank identically.)

**Q:** The two embedding-pipeline rules that prevent silent disasters?
**A:** (1) Index and query with the SAME model — different models' vectors live in incompatible spaces. (2) Switching models means re-embedding the entire database.

**Q:** Do embeddings measure truth?
**A:** No — topic proximity only. "The earth is flat" and "the earth is round" embed close together (same topic). Retrieval relevance ≠ factual correctness.

---

## Text Classification

**Q:** The three approaches, in order of when to try them?
**A:** (1) Zero-shot pipeline — instant, no training, ~70-85%. (2) Fine-tuned DistilBERT — best for fixed labels at volume, ~90-95%. (3) LLM prompting — most flexible, costs per call. Start zero-shot; upgrade only when accuracy demands.

**Q:** Multi-class vs multi-label — and the activation each needs?
**A:** Multi-class: exactly one label → softmax (probabilities sum to 1). Multi-label: zero-to-N labels → sigmoid per class with a threshold. Wrong choice = wrong loss = broken training.

**Q:** Why does TF-IDF fail on "not bad"?
**A:** It only counts words — "bad" scores as negative evidence regardless of the "not." Transformers read context, so negation flips the meaning correctly.

**Q:** How can zero-shot classify labels it never trained on?
**A:** It's an NLI (natural language inference) model under the hood: it asks "does this text *entail* the label 'technology'?" for each candidate label. Entailment reasoning generalizes to any label you can name.

**Q:** Your classifier says NEGATIVE at 62% confidence — what do you do in production?
**A:** Treat it as uncertain — it's barely above the coin-flip line. Set a confidence threshold (e.g., 80%) and route low-confidence predictions to a fallback or human review.

---

## NER

**Q:** What ambiguity does BIO tagging solve?
**A:** Where one entity ends and the next begins. [PER, PER, O, PER, PER] can't distinguish one 4-word person from two 2-word people; [B-PER, I-PER, O, B-PER, I-PER] can — B marks a new entity, I continues it.

**Q:** NER vs text classification, architecturally?
**A:** Same BERT backbone, different head level: classification reads one [CLS] vector per sentence; NER runs the classification head on *every token's* output vector. (`AutoModelForTokenClassification` vs `...SequenceClassification`.)

**Q:** Why is `bert-base-uncased` wrong for NER?
**A:** Capitalization is a core signal for proper nouns — "Apple" (company) vs "apple" (fruit). Uncased models lowercase everything, destroying it. Use cased models.

**Q:** The subword-alignment trick when fine-tuning NER?
**A:** Labels are per word but the model sees subword tokens. Standard pattern: give the real tag to the first subword of each word and `-100` (ignored by cross-entropy) to continuations — via `word_ids()`.

**Q:** Custom entity types, no training data — what's the move?
**A:** LLM prompting: describe the entities and ask for JSON. Fine-tune BERT only when you need high throughput / low cost per prediction and have labeled data.

---

## Seq2Seq & Summarization

**Q:** Encoder vs decoder — division of labor?
**A:** Encoder reads the FULL input bidirectionally and produces one contextual vector per input token (not a single bottleneck vector). Decoder generates the output token-by-token, consulting those encoder states via cross-attention.

**Q:** In cross-attention, where do Q, K, V come from?
**A:** Q from the decoder's current position ("what am I looking for right now?"); K and V from the encoder outputs. That's the only difference from self-attention, and it's what lets the decoder look at the input while writing.

**Q:** What failure does beam search prevent?
**A:** Greedy decoding locks in the locally-best token and can never reconsider — a better overall sentence starting with a slightly-less-likely word is unreachable. Beam search keeps the top-K partial sentences alive and returns the best complete one.

**Q:** Extractive vs abstractive — which when?
**A:** Extractive copies source sentences — choose for fidelity (legal, medical; can't hallucinate). Abstractive generates new phrasing — choose for fluency (user-facing summaries), but verify facts: it can hallucinate.

**Q:** Repetitive generation loop ("the company the company...") — the fix?
**A:** `no_repeat_ngram_size=3` (blocks repeating any 3-gram), plus beam search instead of greedy.

**Q:** What can ROUGE not tell you?
**A:** Whether the summary is *true*. It measures lexical overlap with a reference — hallucinated facts and brilliant paraphrases both get mis-scored. It's a proxy; pair with human/factuality evaluation.

---

## BERT

**Q:** What does bidirectionality buy, concretely?
**A:** Each word is understood using context on BOTH sides. "I sat on the bank **of the river**" — the disambiguating words come *after* "bank"; a left-to-right model hasn't seen them yet.

**Q:** How does masked language modeling train bidirectionality without cheating?
**A:** Hide ~15% of tokens and predict them from the full surrounding context. You can't "peek at the answer" of a blank — filling it requires genuine understanding from both directions, at internet scale.

**Q:** What happens to BERT's pretraining head when you fine-tune for classification?
**A:** The MLM output layer (768 → vocab_size) is thrown away; a fresh head (768 → num_classes) is bolted on. The 12 encoder layers — the language understanding — are reused; that's the entire point.

**Q:** Why can't BERT generate text?
**A:** It's encoder-only — no autoregressive decoder, never trained to predict the next token. It produces representations of existing text; generation needs GPT-family models.

**Q:** Vanilla BERT [CLS] for sentence similarity — good idea?
**A:** No — its geometry is poor for cosine comparison. Use sentence-transformers (BERT fine-tuned with contrastive learning specifically for similarity).

**Q:** Default encoder model to start any project with?
**A:** DistilBERT — 60% faster, 40% smaller, ~97% of BERT's accuracy. Upgrade to RoBERTa/DeBERTa only when you hit an accuracy wall.

---

## GPT & Decoder Models

**Q:** The autoregressive loop in one sentence?
**A:** Given the text so far, output a probability for every possible next token, pick one, append it, feed the whole thing back in — repeat until an end token or max_tokens.

**Q:** Why does "predict the next word" produce reasoning?
**A:** Predicting the next token *well* across all human text requires modeling whatever generates that text — facts, logic, syntax, code semantics. Capability is a side effect of the prediction game on a hard enough dataset.

**Q:** The 3-stage pipeline from text-completer to assistant?
**A:** (1) Pre-training: next-token prediction on ~1T tokens → raw knowledge. (2) Instruction fine-tuning: (prompt, response) pairs → follows instructions. (3) RLHF: human preference rankings → helpful, harmless behavior.

**Q:** Does few-shot prompting update the model's weights?
**A:** No — examples are just context the model pattern-matches against at inference. Weights only change during training. That's why "in-context learning" needs no GPU.

**Q:** What does the KV cache store, and which two costs does it explain?
**A:** The attention Keys/Values of all previous tokens, so each new token doesn't recompute them. Explains (1) why streaming is fast after a slow prefill (prefill is O(N²) in prompt length), and (2) why prompt caching (Anthropic/OpenAI) saves money — reuse the KV of a big system prompt across requests.

**Q:** Temperature 0 / 0.7 / 1.5 — when?
**A:** 0: deterministic — classification, extraction, factual answers. 0.7: balanced chat/writing. 1.5+: flattened distribution, creative but often incoherent.

---

## Hugging Face

**Q:** The three API levels?
**A:** `pipeline()` — zero-config, task in, results out (prototyping). `AutoTokenizer`/`AutoModel*` — manual control over preprocessing and outputs. `Trainer` — the full fine-tuning loop with checkpointing, eval, and logging.

**Q:** `AutoModel` vs `AutoModelForSequenceClassification`?
**A:** `AutoModel` = bare backbone, outputs hidden states only. The `For*` variants add the task head: `ForSequenceClassification` (labels), `ForTokenClassification` (NER), `ForCausalLM` (generation), `ForSeq2SeqLM` (T5/BART).

**Q:** What must be saved together after fine-tuning, and why?
**A:** Model AND tokenizer (`trainer.save_model()` + `tokenizer.save_pretrained()`). Inference must preprocess exactly as training did; without the tokenizer's vocab and rules, inputs won't match.

**Q:** `dataset.map(tokenize)` vs `dataset.map(tokenize, batched=True)`?
**A:** `batched=True` passes batches (default 1000) to the fast Rust tokenizer instead of one example per Python call — typically 10–100× faster preprocessing.

**Q:** When should you NOT reach for Hugging Face?
**A:** Calling hosted LLM APIs (use the provider SDK), extreme-scale serving (vLLM/ONNX Runtime/TorchServe), or edge devices (llama.cpp / ONNX).

---

## Fine-Tuning NLP

**Q:** The decision flow before any fine-tuning?
**A:** Try zero-shot prompting → try few-shot prompting → only if accuracy still isn't enough AND you have 100+ labeled examples per class, fine-tune. Don't fix what prompting already solves.

**Q:** Fine-tuning learning rate, and the disaster at 100× larger?
**A:** 2e-5 to 5e-5. At ~1e-3 you get catastrophic forgetting — the gradients from your small dataset overwrite the pretrained knowledge within steps, and more epochs make it worse.

**Q:** LoRA in two sentences?
**A:** Freeze all base weights; at targeted attention matrices add two tiny trainable matrices A [d×r] and B [r×d] whose product is the weight update (rank r≈8). You train ~0.05% of parameters, fit in 8–16 GB instead of 60–80 GB, and quality matches full fine-tuning on most tasks.

**Q:** What does QLoRA add, and what does it enable?
**A:** 4-bit quantization of the frozen base model under the LoRA adapters — a 7-8B model fine-tunes in ~6 GB (consumer GPU / free Colab) with near-identical quality.

**Q:** Why does training 0.05% of parameters match full fine-tuning?
**A:** Fine-tuning's weight updates have low intrinsic rank — the adaptation lives in a low-dimensional subspace that rank-8 adapter matrices can capture. You're not learning language; you're steering it.

**Q:** Fine-tuning vs RAG — which for which problem?
**A:** Fine-tuning changes *how* the model responds (format, style, task behavior). RAG changes *what facts* it has access to. New knowledge → RAG; new behavior → fine-tune.
