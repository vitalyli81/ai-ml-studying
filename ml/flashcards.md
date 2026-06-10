# ML Flashcards — Spaced Repetition Deck

> **How to use this file:** Don't read it — *quiz yourself with it.* Cover the **A:** line, answer out loud, then check. Review cards you missed again the same day, then after 2 days, then after a week (see [README.md](README.md) → How to Study This Folder). To import into Anki: each `Q:`/`A:` pair is one card.

---

## ML Fundamentals

**Q:** Traditional programming vs machine learning — what goes in, what comes out?
**A:** Traditional: data + rules → answers. ML: data + answers → rules (the model learns the rules from examples).

**Q:** Supervised vs unsupervised learning in one line each?
**A:** Supervised: learn from labeled examples (inputs + correct answers). Unsupervised: find hidden structure in unlabeled data.

**Q:** Classification vs regression?
**A:** Classification predicts a category (spam/not spam); regression predicts a number (price, temperature).

**Q:** Why never evaluate on training data?
**A:** The model has seen it and can memorize it — like grading a student on the exact problems they practiced. Only unseen (test) data measures generalization.

**Q:** Which algorithms need feature scaling, and which don't?
**A:** Need: KNN, SVM, K-Means, PCA, linear/logistic regression (with gradient descent or regularization), neural nets. Don't: decision trees, Random Forest, gradient boosting (they compare thresholds, not distances).

**Q:** Tabular data, need best accuracy — neural network or XGBoost?
**A:** XGBoost (gradient boosting). Tree ensembles still beat neural nets on most tabular datasets, and they're faster and more interpretable. Deep learning wins on images, text, and audio.

---

## Linear Regression

**Q:** What does a learned weight of 150 on `sqft` mean?
**A:** Every extra square foot adds $150 to the predicted price — the weight is the change in prediction per unit change of that feature.

**Q:** Why does MSE *square* the errors instead of using absolute values?
**A:** Big misses get disproportionately punished (20K off hurts 4× more than 10K off), pushing the model to avoid large errors — and squared loss is smooth, so gradient descent works cleanly.

**Q:** Gradient descent in one sentence?
**A:** Repeatedly nudge each weight in the direction that reduces the loss (opposite the gradient), with step size = learning rate, until the loss stops shrinking.

**Q:** What goes wrong with a learning rate that's too high? Too low?
**A:** Too high: weights overshoot and bounce, possibly diverging. Too low: converges painfully slowly. Start ~0.01 and tune.

**Q:** Ridge vs Lasso — which one eliminates features?
**A:** Lasso (L1) pushes some weights to exactly 0 (built-in feature selection). Ridge (L2) shrinks all weights but never zeroes them.

**Q:** Can "linear" regression fit a curve?
**A:** Yes — add polynomial features (x²). It's still linear *in the weights*, which is all "linear" means.

---

## Logistic Regression

**Q:** Despite the name, what does logistic regression actually do?
**A:** Classification — it predicts the probability of class 1 (via sigmoid), then thresholds it (default 0.5) into a class label.

**Q:** Why the sigmoid, mathematically?
**A:** The model does linear regression on the *log-odds* (which span −∞ to +∞ like a line does). The sigmoid is just the inverse transform that converts log-odds back into a 0–1 probability.

**Q:** What does log loss punish hardest?
**A:** Confident wrong predictions — saying 99% class 1 when it's class 0 costs ~∞ as p→0. Being uncertain costs little.

**Q:** When should you move the threshold below 0.5?
**A:** When false negatives are expensive (disease screening, fraud) — a lower threshold catches more positives at the cost of more false alarms.

**Q:** What boundary shapes can logistic regression draw?
**A:** Only straight lines (hyperplanes). Non-linear boundaries need feature engineering or tree-based models.

**Q:** What does `class_weight='balanced'` do and when do you need it?
**A:** It up-weights the minority class during training — use it for imbalanced data (e.g., 1% fraud) so the model can't just predict the majority class.

---

## Decision Trees

**Q:** Gini impurity of a group with 10 spam / 0 ham? 5 spam / 5 ham?
**A:** 0 (perfectly pure) and 0.5 (maximally mixed for 2 classes). Lower = purer = better.

**Q:** How does a tree pick its next split?
**A:** It tries every feature and every threshold, computes the drop in (weighted) impurity for each, and keeps the split with the biggest drop.

**Q:** Gini vs entropy — what's the practical difference?
**A:** Almost none — they produce nearly identical trees. Gini is the sklearn default because it's faster (no logarithm).

**Q:** First two hyperparameters to constrain an overfitting tree?
**A:** `max_depth` (3–5 to start) and `min_samples_leaf` (~5). An unconstrained tree memorizes training data (100% train accuracy, poor test).

**Q:** Why don't trees need feature scaling?
**A:** Splits are threshold comparisons within one feature ("is x > v?") — rescaling changes the threshold value but not the split, so scale is irrelevant.

**Q:** When is a single decision tree the right production choice?
**A:** When every prediction must be explainable in plain rules (compliance, medicine). Otherwise use Random Forest / boosting for accuracy.

---

## Random Forest

**Q:** What is bagging?
**A:** Bootstrap aggregating — train each tree on a random sample of rows drawn *with replacement* (some rows repeat, ~37% are left out), then combine the trees' predictions.

**Q:** Why does Random Forest also restrict features at each split?
**A:** Otherwise every tree grabs the same strong feature and they all look alike — correlated trees don't cancel each other's errors. Random feature subsets force diversity.

**Q:** How do the trees combine for classification vs regression?
**A:** Classification: majority vote. Regression: average of the trees' predictions.

**Q:** What is the OOB score?
**A:** Each tree is evaluated on the ~37% of rows it never saw (out-of-bag) — a free validation estimate. Good for iteration, not a replacement for a held-out test set.

**Q:** Why can Random Forest grow deep trees without the overfitting that kills a single tree?
**A:** Each deep tree overfits *different* noise (different rows + features); averaging cancels the noise and keeps the shared signal.

**Q:** Why does feature importance mislead with correlated features?
**A:** Correlated features split the credit (each gets ~half the true importance). Use permutation importance instead.

---

## Gradient Boosting

**Q:** What does each new tree in gradient boosting learn to predict?
**A:** The residuals — the errors left over by all previous trees combined. Not the original target.

**Q:** Why is it called *gradient* boosting?
**A:** For squared loss, the residual IS the negative gradient of the loss w.r.t. the prediction — each tree is one gradient-descent step in function space. Swap the loss, and the same loop handles classification or ranking.

**Q:** Lower learning rate means what for the number of trees?
**A:** More trees needed (each correction is smaller) — but typically better generalization. The two must be tuned together.

**Q:** Why is early stopping essential for boosting but not for Random Forest?
**A:** Boosting keeps fitting the residuals and *will* eventually overfit with more trees; RF just plateaus. Early stopping watches a validation set and stops when it stops improving.

**Q:** XGBoost vs LightGBM vs CatBoost — when each?
**A:** XGBoost: default, battle-tested. LightGBM: large data (>100K rows), 10–100× faster, leaf-wise growth. CatBoost: many categorical features, no encoding needed.

**Q:** Random Forest vs Gradient Boosting in one line?
**A:** RF: independent trees in parallel, averaged (jury voting). GBM: sequential trees, each fixing the previous ensemble's mistakes (editors revising a draft).

---

## SVM

**Q:** What is SVM optimizing for?
**A:** The widest possible margin — of all boundaries that separate the classes, the one with the most empty space on each side generalizes best.

**Q:** What is a support vector?
**A:** A training point sitting on the margin's edge — the points the boundary "leans on." Remove any other point and the boundary doesn't move.

**Q:** The kernel trick in one sentence?
**A:** Compute similarities *as if* the data were mapped into a higher-dimensional space — without ever computing the high-dimensional coordinates — so a flat boundary there becomes a curved boundary here.

**Q:** High C vs low C?
**A:** High C: strict — no training errors tolerated, narrow margin, overfit risk. Low C: relaxed — wide margin, tolerates violations, underfit risk. Start at C=1, tune by CV.

**Q:** High gamma vs low gamma (RBF)?
**A:** High gamma: each point's influence is local → wiggly boundary, overfit risk. Low gamma: influence spreads far → smooth boundary, underfit risk.

**Q:** Why does SVM hit a wall on large datasets?
**A:** Training is O(n²–n³) in samples. Beyond ~50–100K rows, use LinearSVC or switch to gradient boosting.

---

## KNN

**Q:** How does KNN make a prediction?
**A:** Compute distance from the new point to *every* training point, take the K nearest, majority-vote (classification) or average (regression).

**Q:** Why is KNN a "lazy learner"?
**A:** No training phase — it just stores the data. All computation happens at prediction time (O(n×d) per query).

**Q:** Small K vs large K — bias or variance?
**A:** Small K: low bias, high variance (jagged boundary, noise-sensitive). Large K: high bias, low variance (over-smoothed). Tune with cross-validation; use odd K for binary.

**Q:** The curse of dimensionality in one sentence?
**A:** With many features, all points become roughly equidistant — nearest ≈ farthest — so "nearest neighbor" becomes a random pick. Fix: PCA first, or a non-distance model.

**Q:** What happens to KNN if salary ($0–200K) and age (0–100) are unscaled?
**A:** Salary dominates every distance ~2000:1; age is effectively ignored. Scaling is mandatory.

**Q:** What modern systems are basically KNN at scale?
**A:** Vector databases / approximate nearest-neighbor search (FAISS) — the retrieval step in RAG — and collaborative-filtering recommenders.

---

## Naive Bayes

**Q:** Disease is 1% prevalent; test is 90% accurate; you test positive. Roughly what's the chance you're sick?
**A:** Only ~8% — false alarms from the huge healthy group (990 of 9,900) swamp the true positives (90 of 100). The prior (rarity) matters as much as the test.

**Q:** What exactly does "naive" assume?
**A:** That features are conditionally independent given the class — so you can multiply per-feature likelihoods. Wrong in reality, but the class *ranking* usually survives.

**Q:** What breaks without Laplace smoothing?
**A:** One unseen word → P=0 → the whole product collapses to 0, vetoing all other evidence. Smoothing (alpha=1) keeps every probability nonzero.

**Q:** Which variant for TF-IDF text? For continuous measurements?
**A:** MultinomialNB for counts/TF-IDF; GaussianNB for continuous numerics; BernoulliNB for binary presence/absence.

**Q:** Why log probabilities internally?
**A:** Multiplying hundreds of tiny probabilities underflows to 0. log turns products into sums, which never underflow.

**Q:** Should you trust "99% confident" from Naive Bayes?
**A:** No — the independence assumption makes its probabilities overconfident. The ranking is reliable; calibrate (CalibratedClassifierCV) if you need real probabilities.

---

## K-Means

**Q:** The K-Means loop in one line?
**A:** Repeat: assign every point to its nearest centroid; move each centroid to the mean of its points — until nothing changes.

**Q:** Why is convergence guaranteed?
**A:** Both steps can only reduce (or keep) total inertia, which can't go below zero — so it must settle. But only to a *local* minimum, hence `n_init=10`.

**Q:** Why can't you pick K by minimizing inertia?
**A:** Inertia always drops as K grows (K=n gives inertia 0). Use the elbow (where improvement flattens) plus silhouette score.

**Q:** Silhouette score of ~0 for a point means what? Of 1? Of −1?
**A:** 0: on the boundary between two clusters. 1: clearly belongs to its cluster. −1: probably in the wrong cluster.

**Q:** KNN vs K-Means?
**A:** KNN is supervised (K = neighbors consulted to classify a labeled point). K-Means is unsupervised (K = number of groups to discover). Both use distance, different jobs.

**Q:** When is K-Means the wrong clusterer?
**A:** Non-spherical clusters (rings, elongated shapes) or very different densities — use DBSCAN or Gaussian Mixture Models.

---

## PCA

**Q:** What does PCA maximize when choosing PC1?
**A:** Variance — PC1 is the direction along which the projected data is most spread out; PC2 is the most-spread direction perpendicular to it.

**Q:** Does PCA select your best original features?
**A:** No — it creates *new* features (linear combinations of all originals). You lose feature-level interpretability; use feature selection if you need it.

**Q:** "PC1 explains 65% of variance" means what?
**A:** Projecting onto PC1 alone retains 65% of the dataset's total variability — equivalently, PC1's eigenvalue is 65% of the eigenvalue sum.

**Q:** Eigenvector vs eigenvalue, in PCA terms?
**A:** Eigenvector = the direction (the ruler / principal component). Eigenvalue = how much variance lies along it (keep big, drop tiny).

**Q:** What happens if you skip scaling before PCA?
**A:** The highest-variance raw feature (e.g., salary in dollars) hijacks PC1; low-scale features get ignored regardless of importance.

**Q:** When does PCA NOT help?
**A:** Tree-based models (they handle many features natively), non-linear structure (use t-SNE/UMAP), or when you need to name which original features matter.

---

## Feature Engineering

**Q:** Why is encoding red=1, blue=2, green=3 a trap?
**A:** Integers imply an order and arithmetic ("green > red", "avg of red and green = blue") that doesn't exist. Use one-hot for unordered categories; ordinal only when a real order exists (S < M < L).

**Q:** Why is filling missing values with 0 a lie?
**A:** 0 is a real value with meaning (zero salary ≠ unknown salary). Use median/mean imputation plus an `is_missing` indicator column.

**Q:** Which scaler for data with heavy outliers?
**A:** RobustScaler (median + IQR) — outliers don't distort it the way they wreck MinMaxScaler and skew StandardScaler.

**Q:** When should you KEEP outliers?
**A:** When they ARE the target: fraud, defects, churn, anomaly detection. Removing them deletes the signal.

**Q:** What is data leakage and the one pattern that prevents it?
**A:** Test-set information sneaking into training (e.g., scaler fit on all data, naive target encoding). Prevention: do every transform inside a Pipeline — fit on train folds only, transform test with frozen parameters.

**Q:** High-cardinality categorical (500 cities) with predictive signal — how to encode?
**A:** Target encoding with fold-safe CV (or embeddings for deep learning). One-hot would explode into 500 columns.

---

## Model Evaluation

**Q:** Validation set vs test set?
**A:** Validation: looked at repeatedly to tune hyperparameters. Test: touched ONCE at the end for the unbiased final number. Tuning on test silently inflates your score.

**Q:** Precision vs recall, fishing-net version?
**A:** Precision: of everything I pulled up, how much was actually fish (purity of catch)? Recall: of all fish in the lake, how many did I catch (completeness)? They trade off via the threshold.

**Q:** Cancer screening vs spam filter — which metric does each favor?
**A:** Cancer: recall (a missed case is deadly; false alarms just mean more tests). Spam: precision (a real email in the spam folder is costly; some spam getting through is fine).

**Q:** Train 99% / test 68% — diagnosis and fixes?
**A:** High variance (overfitting). Fixes: regularize, simplify the model, get more data, early stopping. (Both low → high bias → add complexity/features.)

**Q:** Why is ROC-AUC misleading at 0.1% positives, and what's the fix?
**A:** TPR/FPR don't penalize poor precision — AUC can look great while nearly every flagged positive is wrong. Use PR-AUC for heavy imbalance.

**Q:** MAE vs RMSE — when each?
**A:** MAE treats all errors equally (robust to outliers). RMSE punishes big misses harder (use when large errors are disproportionately bad). Both are in target units.

**Q:** Cross-validation on time series — what changes?
**A:** Use TimeSeriesSplit, never random K-fold — random folds would train on the future to predict the past.

---

## Scikit-learn

**Q:** `fit` vs `transform` vs `fit_transform`?
**A:** `fit` = learn parameters from data (mean, categories). `transform` = apply the learned rule. `fit_transform` = both in one call — training data only.

**Q:** The golden rule for scalers/encoders on train vs test?
**A:** `fit_transform(X_train)`, then `transform(X_test)` with the SAME frozen parameters. Fitting on test = data leakage.

**Q:** Why are Pipelines non-negotiable rather than a convenience?
**A:** They make leakage impossible by construction (fit on train folds only, even inside CV), and they bundle preprocessing + model into one savable, deployable object.

**Q:** What does the double underscore in `model__n_estimators` mean?
**A:** GridSearch addressing for a parameter inside a pipeline step: `stepname__paramname` sets `n_estimators` on the step named `model`.

**Q:** "ValueError: expected 2D array" when passing one feature — fix?
**A:** sklearn wants X as 2D (samples × features). Use `df[['col']]` (stays a DataFrame) or `.values.reshape(-1, 1)`.

**Q:** Why save the whole pipeline with joblib instead of just the model?
**A:** The pipeline carries the fitted scalers/encoders — the loaded object accepts raw data directly. Saving only the model means manually recreating (and version-syncing) every preprocessing step at inference.
