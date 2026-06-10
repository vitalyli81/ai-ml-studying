# Deep Learning Flashcards — Spaced Repetition Deck

> **How to use this file:** Don't read it — *quiz yourself with it.* Cover the **A:** line, answer out loud, then check. Review misses again the same day, then after 2 days, then after a week (see [README.md](README.md) → How to Study This Folder). Each `Q:`/`A:` pair is one Anki card.

---

## Fundamentals

**Q:** What does "deep" in deep learning refer to, and what do the layers learn?
**A:** Many stacked layers. Each learns increasingly abstract features automatically: edges → textures → parts → objects (or characters → words → meaning). You design the architecture; the network designs the features.

**Q:** Deep learning vs classical ML — when does each win?
**A:** DL: unstructured data (images, text, audio), large datasets, GPU available. Classical ML (XGBoost): tabular data, small datasets, explainability, fast CPU training. XGBoost still beats neural nets on most tabular problems.

**Q:** The three things that made deep learning take off after 2012?
**A:** Data (internet-scale datasets like ImageNet), compute (GPUs), and algorithms (ReLU, batch norm, Adam, skip connections, attention).

**Q:** Name the big-three architectures and their data types.
**A:** CNN → images/spatial grids. RNN/LSTM → sequences processed step-by-step (time series, streaming). Transformer → sequences processed in parallel via attention (text, and increasingly everything).

---

## Neural Network Basics

**Q:** What does one neuron compute?
**A:** A weighted sum of its inputs plus a bias, passed through an activation function: `output = activation(w·x + b)`. It's literally logistic regression as a building block.

**Q:** Why is a 100-layer network with no activation functions pointless?
**A:** A composition of linear functions is still one linear function — it all collapses to a single matrix multiply. Activations insert the non-linear "bends" that make depth add expressive power.

**Q:** The 5-line PyTorch training loop, in order?
**A:** `optimizer.zero_grad()` → `pred = model(x)` → `loss = loss_fn(pred, y)` → `loss.backward()` → `optimizer.step()`. (Clear stale gradients, forward, measure, compute gradients, update.)

**Q:** Epoch vs batch vs iteration?
**A:** Epoch = one full pass through all training data. Batch = the chunk processed per gradient update. Iteration = one batch. 1,000 samples at batch size 100 → 10 iterations per epoch.

**Q:** What does the Universal Approximation Theorem promise — and not promise?
**A:** Promises: a big-enough network CAN represent any continuous function. Doesn't promise: that training will actually FIND it — optimization is the hard part.

**Q:** Train loss 0.01, validation loss 2.5 — what's happening?
**A:** Overfitting — the model memorized the training set. The train/val gap is the red flag. Fix: regularization (dropout, weight decay), more data, smaller model, early stopping.

---

## Backpropagation

**Q:** What does a weight's gradient tell you — and what doesn't it?
**A:** Tells: the direction and relative magnitude to change the weight to reduce loss ("blame"). Doesn't tell: the right final value — you take a small step (learning rate) and recompute.

**Q:** The chain rule in plain words?
**A:** A weight's influence on the final loss = its local effect × the effects of everything downstream, multiplied along the path. Backprop computes each layer's local gradient once and reuses it for everything upstream — two passes total, not millions.

**Q:** Why do gradients vanish, mechanically?
**A:** Gradients multiply backward through every layer. If each layer's local gradient is < 1 (sigmoid's max is 0.25), twenty layers give 0.25²⁰ ≈ 0 — early layers receive no signal and stop learning. Fixes: ReLU, batch norm, skip connections.

**Q:** Loss suddenly becomes NaN — diagnosis and fixes?
**A:** Exploding gradients. Lower the learning rate, add gradient clipping (`clip_grad_norm_(params, 1.0)` before `optimizer.step()`), check for huge input values.

**Q:** `loss.backward()` vs `optimizer.step()`?
**A:** `backward()` only computes gradients and stores them in each parameter's `.grad`. `step()` reads those and actually updates the weights (`w -= lr × grad`). Compute, then apply.

**Q:** Why does `optimizer.zero_grad()` exist?
**A:** PyTorch accumulates gradients (adds to `.grad`) by default. Skip the zeroing and every batch's update is contaminated by previous batches — the #1 beginner bug.

---

## Activation Functions

**Q:** ReLU in one formula, and its failure mode?
**A:** `max(0, x)`. Failure: dying ReLU — a neuron whose input is always negative has zero gradient forever and stops learning. Fix: Leaky ReLU (`max(0.01x, x)`).

**Q:** Why is sigmoid wrong for hidden layers but right for binary output?
**A:** Hidden: its gradient is at most 0.25 and saturates near 0/1 → vanishing gradients through depth. Output: squashing to (0,1) is exactly what a binary probability needs.

**Q:** Softmax does what to a vector of logits?
**A:** Converts raw scores into probabilities that sum to 1.0 — each class's share. `[2.0, 1.0, 0.1] → [0.66, 0.24, 0.10]`.

**Q:** Why must you NOT add softmax before `nn.CrossEntropyLoss`?
**A:** It applies log-softmax internally (numerically stable). Adding your own gives `log(softmax(softmax(x)))` — squashed distribution, flat gradients, stalled training. Pass raw logits.

**Q:** Output activation for: regression / binary / multi-class?
**A:** Regression: none (linear). Binary: sigmoid (or better: no activation + BCEWithLogitsLoss). Multi-class: none + CrossEntropyLoss (softmax happens inside the loss).

**Q:** Where does GELU show up, and is it better than ReLU everywhere?
**A:** Transformers (GPT, BERT, LLaMA) — a smooth ReLU that lets small negatives partially through. For CNNs/MLPs, plain ReLU is usually equivalent and faster.

---

## Loss Functions & Optimizers

**Q:** MSE vs cross-entropy — which task gets which, and why does MSE fail at classification?
**A:** MSE: regression. Cross-entropy: classification. MSE's penalty on a probability caps at ~1 and its gradient shrinks on confident-wrong answers; cross-entropy's −log(p) penalty grows to infinity exactly when the model is confidently wrong.

**Q:** The one rule every optimizer repeats?
**A:** `new_weight = old_weight − learning_rate × gradient` — step downhill, opposite the slope. Everything fancier (momentum, Adam) is a smarter version of this step.

**Q:** What two running statistics does Adam keep per weight?
**A:** Momentum (average of past gradients → smooth, fast direction, like a heavy ball) and variance (average of squared gradients → per-weight adaptive step size). Big-gradient weights step smaller; tiny-gradient weights step bigger.

**Q:** Why AdamW instead of Adam?
**A:** Adam incorrectly mixes weight decay into the adaptive scaling, weakening it. AdamW decouples weight decay (applies it separately after the update) — strictly better; what GPT/BERT/LLaMA use.

**Q:** Default learning rates: training from scratch with AdamW? Fine-tuning pretrained?
**A:** From scratch: 1e-3. Fine-tuning: 1e-5 to 5e-5 (NLP) / ~1e-4 (vision) — 10–100× smaller, or you destroy the pretrained knowledge.

**Q:** MSE or MAE when the data has extreme outliers?
**A:** MAE (L1) — it penalizes linearly, so a few extreme values don't dominate the loss the way their *squared* errors would under MSE.

---

## Regularization

**Q:** Diagnose: train 99%/val 72%; train 85%/val 83%; train 60%/val 59%?
**A:** Overfit (big gap → regularize, more data); healthy; underfit (both low → bigger model, more features, train longer).

**Q:** What does dropout do at train time, and at eval time?
**A:** Train: randomly zeroes fraction p of neurons each forward pass (survivors scaled by 1/(1−p)), forcing redundant pathways. Eval (`model.eval()`): completely disabled — all neurons active, no scaling, no speed cost.

**Q:** Why do small weights generalize better?
**A:** A weight is how violently the output reacts to an input. Huge weights = a spiky curve threading through every noisy point; small weights = the smoothest curve that still fits the trend. Weight decay penalizes Σw², buying smoothness.

**Q:** Early stopping — what do you monitor, and which weights do you keep?
**A:** Monitor validation loss (training loss always decreases). Keep the checkpoint with the lowest val loss — the final epoch's weights are already overfit.

**Q:** BatchNorm or LayerNorm in a Transformer?
**A:** LayerNorm. BatchNorm is for CNNs/MLPs and breaks down at small batch sizes; Transformers normalize per token across features.

**Q:** When is data augmentation dangerous?
**A:** When it changes the label — horizontally flipping a "6" makes a "9". Augmentations must preserve meaning.

---

## CNNs

**Q:** What does a convolution filter compute at each position?
**A:** Element-wise multiply the small filter against the patch under it and sum — one number per position. High value = the filter's pattern is present there. The full grid of results is a feature map.

**Q:** The two giant wins of parameter sharing (same filter everywhere)?
**A:** (1) Vastly fewer parameters — a 3×3 filter is 9 weights vs millions in a dense layer. (2) Translation invariance — one edge detector works in every corner of the image.

**Q:** What does max pooling keep and discard?
**A:** Keeps whether the pattern fired (the max per region); discards exactly where. Halves spatial size, cuts compute, adds shift tolerance.

**Q:** `[1, 3, 224, 224]` through `Conv2d(3, 64, kernel_size=3, padding=1)` → shape?
**A:** `[1, 64, 224, 224]` — channels become 64 (one per filter); padding=1 with a 3×3 kernel preserves spatial size.

**Q:** Why do early conv layers detect edges and deep layers detect objects?
**A:** Composition: layer 1 sees raw pixels (simplest patterns = edges); layer 2 sees edge maps (combines them into corners/textures); each layer builds on the previous abstraction up to whole objects.

**Q:** Starting a new image classification project — first move?
**A:** Don't build a CNN from scratch. Load a pretrained ResNet/EfficientNet, replace the final layer for your classes, fine-tune with a small LR.

---

## RNN & LSTM

**Q:** An RNN is which JavaScript array method, and what maps to what?
**A:** `Array.reduce()` — the accumulator is the hidden state, the current element is the token, the callback (same at every step) is the RNN cell, the final accumulator is the sequence summary.

**Q:** Why does a plain RNN forget after ~10 steps?
**A:** Two reasons: the fixed-size hidden state gets overwritten every step (dilution), and gradients multiply back through every step — 0.7²⁰ ≈ 0.0008, so distant tokens get no learning signal.

**Q:** The three LSTM gates and their jobs?
**A:** Forget gate: what to erase from the cell state. Input gate: what new info to write. Output gate: what part of the cell state to reveal as this step's output. All learned, all 0–1 dials.

**Q:** Why does the cell state fix vanishing gradients?
**A:** Its update is additive — `c_t = f⊙c_{t−1} + i⊙c̃` — so with gates near 1, information (and gradient) rides the "conveyor belt" through time nearly unchanged instead of being re-squashed at every step.

**Q:** When do you still pick an LSTM/GRU over a Transformer?
**A:** Streaming/real-time data (inputs arrive one at a time), time-series forecasting, tight memory/latency budgets. For full-sequence NLP, Transformers win.

**Q:** What does `bidirectional=True` give you and what does it forbid?
**A:** Each position gets context from both past and future (two RNNs, outputs concatenated — dimension doubles). Forbids: generation and streaming — you need the complete sequence upfront.

---

## Transformers

**Q:** Query, Key, Value in one phrase each?
**A:** Query: what I'm looking for. Key: what I advertise about myself. Value: the actual information I hand over if matched. Each is the token embedding times a learned matrix (W_Q, W_K, W_V).

**Q:** The four steps of self-attention?
**A:** (1) Each token forms its Query. (2) Dot-product the Query with every token's Key → match scores. (3) Softmax the scores → weights summing to 1. (4) Weighted sum of all Values → the token's new, context-enriched representation. In one line: `softmax(QKᵀ/√d)·V`.

**Q:** Why does attention need positional encoding?
**A:** Attention is a set operation — permutation-invariant. "Dog bites man" and "Man bites dog" produce identical outputs without position info. Sinusoidal/learned embeddings or RoPE (modern LLMs) inject order.

**Q:** BERT vs GPT — attention type and consequence?
**A:** BERT: bidirectional attention (sees all tokens) → great at understanding, can't generate. GPT: causal/masked attention (sees only past tokens) → generates left-to-right. The mask is the entire difference.

**Q:** Why are Transformers faster to train than RNNs despite doing more work?
**A:** RNNs are sequential — step t needs step t−1, so no parallelism across time. Transformers process all tokens at once (attention is matrix multiplies) — exactly what GPUs are built for.

**Q:** Why is there a context-window limit?
**A:** Attention is O(n²): every token attends to every other. 1K tokens → 1M attention cells; 100K → 10 billion. GPU memory sets the practical ceiling (Flash Attention and sliding windows push it).

**Q:** Where does a Transformer's factual knowledge mostly live?
**A:** The feed-forward (FFN) layers — attention routes context between tokens; FFN stores and transforms knowledge. Both are essential.

---

## Transfer Learning

**Q:** Why do pretrained features transfer to your unrelated task?
**A:** Early layers learn universal building blocks (edges/textures for vision, syntax/word meaning for text) that every task in the domain needs. Only the final layers are task-specific — so keep the bottom, replace the top.

**Q:** Feature extraction vs fine-tuning?
**A:** Feature extraction: freeze ALL pretrained weights, train only a new output head (use with <500 samples/class). Fine-tuning: also unfreeze later layers and update them with a tiny LR (1e-5) so features adapt to your domain.

**Q:** What is catastrophic forgetting and the one hyperparameter that causes it?
**A:** Fine-tuning with too large a learning rate overwrites the pretrained knowledge within a few steps — the model becomes as bad as random init. Fix the LR (10–100× smaller than from-scratch); more epochs make it worse, not better.

**Q:** What does LoRA train, and why is it ~100× cheaper?
**A:** Freeze the base model; train tiny low-rank adapter matrices A×B (rank ~8) alongside the frozen weights — ~0.1% of parameters. Works because fine-tuning weight updates have low intrinsic rank. Bonus: 20 MB adapters, no catastrophic forgetting, swap adapters per task.

**Q:** Differential learning rates — what's the pattern?
**A:** Smallest LR for the earliest (most universal) layers, larger for later layers, largest for the brand-new head: e.g. 1e-6 → 1e-5 → 1e-4 → 1e-3. Touch-up the foundation, redesign the interior.

**Q:** How many epochs does fine-tuning need?
**A:** 2–10. It converges fast and overfits quickly on small datasets — pair with early stopping. 50–100 epochs is a from-scratch habit that hurts here.

---

## PyTorch

**Q:** What makes a tensor different from a NumPy array?
**A:** GPU execution (`.to('cuda')`) and gradient tracking (`requires_grad=True`) — the two things deep learning needs that NumPy can't do.

**Q:** How does `loss.backward()` know every gradient without you writing calculus?
**A:** Autograd records every operation during the forward pass into a computation graph (the "tape"), then replays it in reverse applying the chain rule, filling each tracked tensor's `.grad`. You write only the forward math.

**Q:** `model.eval()` vs `torch.no_grad()` — what does each change?
**A:** `eval()`: switches dropout off and BatchNorm to running statistics — behavior only. `no_grad()`: stops building the gradient tape — memory and speed. Validation uses both together.

**Q:** Why `model(x)` and never `model.forward(x)`?
**A:** `model(x)` goes through `__call__`, which runs registered hooks before/after `forward()`. Calling `forward` directly skips them and breaks things subtly.

**Q:** Why save `state_dict()` instead of the whole model?
**A:** `state_dict()` is pure tensor data keyed by layer name — stable across code refactors and version upgrades. Pickling the whole model object breaks when classes move or Python/PyTorch versions change.

**Q:** Model on GPU, data on CPU — what happens?
**A:** RuntimeError: "Expected all tensors to be on the same device." Always `.to(device)` both the model and every batch.
