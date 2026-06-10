# Deep Learning Mixed Review Quiz — 25 Scenario Questions

> **Why this file exists:** each doc tests only itself, but interviews and real debugging test whether you can *choose between* architectures and *diagnose* training failures from symptoms. Do 5 at a time, out loud, *before* opening the answer.
>
> Score yourself: ≥20/25 → ready for the LLM/AI-engineering phase. Misses → re-read that doc's "Build the Intuition From Zero" section.

---

## Round 1 — Architecture Choice

**1.** Classify 50K product photos into 30 categories, 2 weeks deadline, one GPU. Walk through your approach — and what you would NOT do.

<details><summary>Answer</summary>

Load a pretrained ResNet/EfficientNet from torchvision, replace the final layer with a 30-class head, freeze the backbone, warm up the head (lr=1e-3, a few epochs), then unfreeze the last blocks with lr=1e-5 and early stopping. Add standard augmentation (flips, crops, color jitter). Do NOT build a CNN from scratch — days of compute for a worse result. ([transfer-learning.md](transfer-learning.md), [cnn.md](cnn.md))
</details>

**2.** Real-time anomaly detection on streaming sensor data, predictions every 100ms on an edge device. Transformer or GRU — and why?

<details><summary>Answer</summary>

GRU (or LSTM). Streaming means inputs arrive one at a time — recurrent models carry a hidden state forward and process each new reading in O(1). A Transformer needs the sequence in memory and re-attends over the window each step, with a much larger memory footprint — wrong shape for low-latency edge streaming. ([rnn-lstm.md](rnn-lstm.md))
</details>

**3.** Your team wants to classify support tickets (5 classes) by prompting a giant decoder-only LLM. You have 10K labeled tickets. What's the stronger classical option and why?

<details><summary>Answer</summary>

Fine-tune an encoder-only model (BERT-style) with a classification head. Classification is an *understanding* task — bidirectional attention builds a representation of the full ticket, and 10K labels is plenty for fine-tuning. A decoder-only generator can do it via prompting but typically underperforms a fine-tuned encoder at this size, at higher cost per request. ([transformers.md](transformers.md))
</details>

**4.** Tabular churn data: 80K rows, 40 columns. Your colleague proposes a 6-layer neural network. What do you say?

<details><summary>Answer</summary>

Try XGBoost/LightGBM first — gradient-boosted trees still beat neural nets on most tabular data, train in seconds on CPU, and need no scaling or architecture tuning. A neural net is the fallback, not the default, for spreadsheets. ([README.md](README.md), [../ml/gradient-boosting.md](../ml/gradient-boosting.md))
</details>

**5.** Why did Transformers replace LSTMs for NLP? Give the two specific mechanisms (not just "they're better").

<details><summary>Answer</summary>

(1) Direct connections: self-attention lets any token attend to any other token in one hop — no information bottleneck through a fixed-size hidden state, so no forgetting at distance. (2) Parallelism: RNNs must process step t after t−1; attention is matrix multiplies over the whole sequence at once, which GPUs eat — training is orders of magnitude faster, which is what made scaling laws practical. ([transformers.md](transformers.md), [rnn-lstm.md](rnn-lstm.md))
</details>

---

## Round 2 — Debugging Training Failures

**6.** Loss goes NaN at epoch 3. List your three moves, in order.

<details><summary>Answer</summary>

Exploding gradients, most likely. (1) Lower the learning rate (often 10×). (2) Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` before `optimizer.step()`. (3) Check inputs for huge/unnormalized values. ([backpropagation.md](backpropagation.md))
</details>

**7.** Training loss decreases for 2 epochs then plateaus far above zero; validation tracks it closely. Overfitting or underfitting? Three fixes.

<details><summary>Answer</summary>

Underfitting (both losses high and close — high bias). Fixes: bigger model (more layers/width), train longer, better features or less aggressive regularization, and check the learning rate isn't too low. Regularization would make this *worse* — that's the overfitting medicine. ([regularization.md](regularization.md))
</details>

**8.** Your model performs great in the notebook but produces *different predictions on every call* in production. The classic cause?

<details><summary>Answer</summary>

You forgot `model.eval()` — dropout is still active at inference, randomly zeroing neurons per call. Fix: `model.eval()` plus `with torch.no_grad():` for memory/speed. ([regularization.md](regularization.md), [pytorch-basics.md](pytorch-basics.md))
</details>

**9.** A 10-layer MLP with sigmoid activations trains, but the first layers' weights barely change while the last layers learn fine. Explain the mechanism and two fixes.

<details><summary>Answer</summary>

Vanishing gradients: sigmoid's derivative is ≤ 0.25, and gradients multiply through every layer — 0.25¹⁰ ≈ 1e-6 reaches layer 1. Fixes: switch hidden activations to ReLU (gradient = 1 when active), add batch norm, or add residual/skip connections. ([backpropagation.md](backpropagation.md), [activation-functions.md](activation-functions.md))
</details>

**10.** Loss decreases beautifully, but every batch seems to "remember" the previous one and updates feel wrong. You inspect the loop:
```python
pred = model(x); loss = loss_fn(pred, y)
loss.backward(); optimizer.step()
```
What's missing and what exactly goes wrong without it?

<details><summary>Answer</summary>

`optimizer.zero_grad()`. PyTorch *accumulates* gradients into `.grad` — without zeroing, batch N's update applies the sum of gradients from batches 1..N. The effective step grows and the optimization direction is corrupted. ([pytorch-basics.md](pytorch-basics.md), [backpropagation.md](backpropagation.md))
</details>

---

## Round 3 — Losses, Activations, Optimizers

**11.** Your classifier ends with `nn.Softmax(dim=1)` and you train with `nn.CrossEntropyLoss`. Accuracy is stuck near random. Why?

<details><summary>Answer</summary>

Double softmax. CrossEntropyLoss applies log-softmax internally; feeding it softmaxed outputs computes log(softmax(softmax(x))) — the distribution gets squashed toward uniform and gradients go nearly flat. Remove the Softmax; pass raw logits. ([activation-functions.md](activation-functions.md), [loss-functions-optimizers.md](loss-functions-optimizers.md))
</details>

**12.** Predicting house prices; a few mansions are 50× the median. MSE or MAE, and what happens with the wrong choice?

<details><summary>Answer</summary>

MAE (L1). Under MSE the mansions' *squared* errors dominate the total loss, dragging the model toward fitting outliers at the expense of typical homes. MAE penalizes linearly — robust to the skew. ([loss-functions-optimizers.md](loss-functions-optimizers.md))
</details>

**13.** Why does cross-entropy use a logarithm? What behavior does it create that MSE can't?

<details><summary>Answer</summary>

−log(p) → ∞ as the probability of the correct class → 0: confidently-wrong predictions get unbounded punishment, and the gradient stays large exactly when correction matters most. MSE on probabilities caps at ~1 and its gradient *shrinks* near confident-wrong answers — the model never gets the strong signal to fix them. ([loss-functions-optimizers.md](loss-functions-optimizers.md))
</details>

**14.** A regression network ends with `nn.ReLU()` on the output. House prices look fine, but profit/loss predictions (which can be negative) are broken. Why?

<details><summary>Answer</summary>

ReLU clamps all negatives to 0 — the model literally cannot output a negative number. Regression outputs take no activation (linear). ([activation-functions.md](activation-functions.md))
</details>

**15.** Explain to a junior: what do Adam's momentum and variance terms each do, and why does AdamW exist?

<details><summary>Answer</summary>

Momentum = running average of gradients: smooths jittery batch gradients into a consistent direction (the heavy ball). Variance = running average of squared gradients: gives each weight its own step size — big-gradient weights step smaller, tiny-gradient ones bigger. AdamW exists because Adam folded weight decay into that adaptive scaling, weakening it; AdamW applies decay separately, as intended — use AdamW. ([loss-functions-optimizers.md](loss-functions-optimizers.md))
</details>

---

## Round 4 — Attention & Architecture Internals

**16.** Walk through how "it" finds "cat" in "The cat sat because it was tired" — all four steps, naming Q, K, V.

<details><summary>Answer</summary>

(1) "it" builds a Query ("pronoun seeking its noun"). (2) The Query is dot-producted against every word's Key → match scores; "cat" (advertising "noun, animal, subject") scores highest. (3) Softmax turns scores into weights summing to 1 — say 0.81 on "cat". (4) "it"'s new representation = weighted sum of all Values, so it now mostly carries "cat"'s content. One matrix line: `softmax(QKᵀ/√d)·V`. ([transformers.md](transformers.md))
</details>

**17.** Remove positional encoding from a Transformer. What specific capability breaks, and why?

<details><summary>Answer</summary>

Word-order understanding. Attention is permutation-invariant — a set operation — so "Dog bites man" and "Man bites dog" produce identical representations. Position signals (sinusoidal, learned, or RoPE in modern LLMs) are the only thing telling the model who came first. ([transformers.md](transformers.md))
</details>

**18.** Why can BERT not write a paragraph, and why can GPT not use future context when classifying a word mid-sentence?

<details><summary>Answer</summary>

BERT trains with bidirectional attention — every token sees the whole sentence — so it has no notion of generating left-to-right (it would be "cheating" itself). GPT uses a causal mask that zeroes attention to future positions so it can generate token-by-token — which also means at any position it has only seen the past. The mask is the whole difference. ([transformers.md](transformers.md))
</details>

**19.** Explain the LSTM cell state as a fix to a specific mathematical problem — not just "it remembers more."

<details><summary>Answer</summary>

A vanilla RNN recomputes its hidden state through a squashing function every step, so backprop multiplies a < 1 factor per step → gradients vanish exponentially with distance. The LSTM cell state updates *additively* (`c_t = f⊙c_{t−1} + i⊙c̃`): with forget gate ≈ 1, both information and gradient pass through nearly unchanged — a highway across time. That's why it learns 100+ step dependencies. ([rnn-lstm.md](rnn-lstm.md))
</details>

**20.** A 3×3 conv filter has 9 weights and finds its pattern anywhere in a 224×224 image. Name the two properties this exemplifies and why a dense layer has neither.

<details><summary>Answer</summary>

Parameter sharing (one filter reused at every position — 9 weights instead of a weight per pixel-neuron pair) and translation invariance (it's literally the same detector everywhere, so position doesn't matter). A dense layer learns a separate weight per input position: ~154M parameters for one layer, and a cat learned in the top-left teaches it nothing about the bottom-right. ([cnn.md](cnn.md))
</details>

---

## Round 5 — Transfer Learning & Practice

**21.** 300 X-ray images, 2 classes. Your colleague says "ImageNet is photos of dogs, not X-rays — pretraining won't transfer; train from scratch." Two-part rebuttal?

<details><summary>Answer</summary>

(1) What transfers is the *early layers* — edges, textures, gradients — which are universal to all images, X-rays included (Yosinski et al. 2014). (2) With 300 images, training from scratch is hopeless anyway; freeze the pretrained backbone, train only a small head — empirically this beats scratch even across large domain gaps. ([transfer-learning.md](transfer-learning.md))
</details>

**22.** You fine-tune BERT with lr=1e-2 "to converge faster." Within 100 steps it's worse than random. Name the phenomenon and the fix.

<details><summary>Answer</summary>

Catastrophic forgetting — large updates overwrote the pretrained weights, destroying the language knowledge. Fix: lr=2e-5-ish (10–100× smaller), optionally warm up by training only the new head first, and use differential LRs (smallest for early layers). More epochs can't undo it; restart from the pretrained checkpoint. ([transfer-learning.md](transfer-learning.md))
</details>

**23.** Your startup wants 5 task-specific versions of a 7B LLM but has one 24 GB GPU and limited storage. What's the play, and why does it work?

<details><summary>Answer</summary>

LoRA (or QLoRA): freeze the base model, train low-rank adapter matrices (~0.1% of params) per task. Fits on the 24 GB GPU, each adapter is ~20 MB (ship one base model + 5 adapters, swap at inference), and the frozen base means no catastrophic forgetting. Works because fine-tuning weight deltas have low intrinsic rank — rank 8–16 captures them. ([transfer-learning.md](transfer-learning.md))
</details>

**24.** Write the validation block of a training loop from memory — the two context switches that must happen, and what each one actually controls.

<details><summary>Answer</summary>

```python
model.eval()                      # dropout off; batchnorm → running stats (behavior)
with torch.no_grad():             # no gradient tape (memory + ~2× speed)
    val_loss = loss_fn(model(X_val), y_val)
model.train()                     # ← back on, or the next epoch trains without dropout
```
`eval()` changes layer *behavior*; `no_grad()` changes gradient *bookkeeping* — they're independent and you want both. Forgetting `model.train()` afterward silently disables regularization for the rest of training. ([pytorch-basics.md](pytorch-basics.md), [regularization.md](regularization.md))
</details>

**25.** Pick the full recipe: 2,000 labeled product reviews, 3 sentiment classes, must ship this week and run cheaply. Walk the stack from model choice to the two most likely bugs.

<details><summary>Answer</summary>

Fine-tune a small pretrained encoder (BERT-base or DistilBERT) via Hugging Face: `AutoModelForSequenceClassification(num_labels=3)`, lr=2e-5, 3–5 epochs, early stopping on validation loss, `load_best_model_at_end=True`. With 2K samples, also consider freezing most layers. Likely bugs: (1) too-high LR → catastrophic forgetting; (2) training too many epochs on 2K samples → overfitting (watch the train/val gap). Honorable mention: a TF-IDF + logistic-regression baseline first — it's an hour of work and gives you the bar to beat. ([transfer-learning.md](transfer-learning.md), [../ml/logistic-regression.md](../ml/logistic-regression.md))
</details>

---

## Scoring yourself

| Score | Verdict |
|---|---|
| 23–25 | Ready for the LLM / AI-engineering phase. Keep [flashcards.md](flashcards.md) on a weekly loop. |
| 18–22 | Solid. Re-read the "Build the Intuition From Zero" sections you missed; retake in 3 days. |
| < 18 | Foundations still forming — redo the per-doc self-checks first, then return here. |
