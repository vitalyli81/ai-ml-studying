# Mixed Review Quiz — 25 Scenario Questions

> **Why this file exists:** each doc tests only itself, but interviews (and real projects) test whether you can *choose between* algorithms and *discriminate* between confusable concepts. Mixing topics in one session ("interleaving") is one of the strongest known boosters for long-term retention. Do 5 questions at a time, out loud, *before* opening the answer.
>
> Score yourself honestly: ≥20/25 → you're interview-ready on classical ML. Anything you miss → re-read that doc's "Build the Intuition From Zero" section.

---

## Round 1 — Picking the Right Algorithm

**1.** You have 200K rows of tabular customer data and need the best possible accuracy for churn prediction. A teammate suggests SVM with an RBF kernel. What do you recommend instead, and why is SVM the wrong call here?

<details><summary>Answer</summary>

Gradient boosting (XGBoost or LightGBM). SVM training is O(n²–n³) — 200K rows would take hours-to-forever, and tree ensembles dominate on tabular data anyway. SVM's sweet spot is *small* (<50K rows), high-dimensional data with a clear margin. ([gradient-boosting.md](gradient-boosting.md), [svm.md](svm.md))
</details>

**2.** A bank regulator requires you to explain *exactly* why each loan was denied, in plain rules. Which models are on the table, and which popular ones are off?

<details><summary>Answer</summary>

On the table: a single decision tree ("denied because income < $40K AND credit score < 700") or logistic regression (signed, interpretable coefficients). Off the table as natively interpretable: Random Forest and gradient boosting — you can't trace one decision through 100+ trees (you'd need post-hoc SHAP/LIME, which may not satisfy a regulator). ([decision-trees.md](decision-trees.md), [logistic-regression.md](logistic-regression.md))
</details>

**3.** 5,000 short text reviews, need a working sentiment classifier *today* with minimal compute. Pipeline?

<details><summary>Answer</summary>

TfidfVectorizer → MultinomialNB. Naive Bayes is the classic fast text baseline: trains in seconds, works with little data. (Next step up: TF-IDF → LogisticRegression; later: transformers.) Never GaussianNB for text, and never StandardScaler before MultinomialNB — it needs non-negative counts. ([naive-bayes.md](naive-bayes.md))
</details>

**4.** Genomics dataset: 800 samples, 20,000 features. Which classifier is famously strong in this exact "features ≫ samples" regime?

<details><summary>Answer</summary>

SVM with a *linear* kernel (high-dimensional data is often already linearly separable — RBF is unnecessary and slower). Regularized logistic regression is the other strong choice. This is the inverse of question 1: SVM loses on many *rows*, wins on many *features* with few rows. ([svm.md](svm.md))
</details>

**5.** "Find the 10 users most similar to this user" vs "discover what types of users we have" — which algorithm family answers each, and what does K mean in each?

<details><summary>Answer</summary>

First: KNN-style nearest-neighbor search (supervised/lookup; K = how many neighbors to retrieve). Second: K-Means clustering (unsupervised; K = how many groups to discover). Classic confusable pair — both use distances, completely different jobs. ([knn.md](knn.md), [k-means.md](k-means.md))
</details>

---

## Round 2 — Preprocessing & Leakage

**6.** Of these six — Random Forest, KNN, XGBoost, SVM, K-Means, decision tree — which need feature scaling, and what's the one-sentence rule?

<details><summary>Answer</summary>

Need scaling: KNN, SVM, K-Means (distance/margin-based). Don't: decision tree, Random Forest, XGBoost (threshold-based splits — scale-invariant). Rule: **if the algorithm measures distances or margins, scale; if it compares thresholds, don't.** ([feature-engineering.md](feature-engineering.md))
</details>

**7.** A teammate runs `scaler.fit_transform(X)` on the full dataset, *then* does train_test_split. The model scores 0.97. What's wrong, what's the symptom in production, and what's the fix?

<details><summary>Answer</summary>

Data leakage — the scaler learned mean/std from test rows, so the "unseen" test set isn't unseen. Symptom: great offline scores, mysteriously worse production performance. Fix: split first, then `fit_transform(X_train)` / `transform(X_test)` — or better, put the scaler in a Pipeline so CV does it correctly automatically. ([scikit-learn.md](scikit-learn.md), [feature-engineering.md](feature-engineering.md))
</details>

**8.** You target-encode a 500-value `city` column by replacing each city with its average house price, computed over the whole training table, then cross-validate. Scores look amazing. What happened?

<details><summary>Answer</summary>

The target leaked into its own feature: each row's encoding includes that row's own target value, and CV folds share the encoding. Fix: compute target encodings *inside* each CV fold (fit on the training fold only), and/or add smoothing. ([feature-engineering.md](feature-engineering.md))
</details>

**9.** `size` column: S, M, L, XL. `payment_method` column: card, cash, crypto. How do you encode each, and why differently?

<details><summary>Answer</summary>

`size` → ordinal encoding (1,2,3,4) because the order is real. `payment_method` → one-hot, because integers would invent a fake ranking ("crypto > cash") and fake arithmetic ("avg of card and crypto = cash"). The test: *does arithmetic on the numbers mean anything?* ([feature-engineering.md](feature-engineering.md))
</details>

**10.** Your fraud-detection dataset has extreme transaction amounts. A teammate winsorizes (caps) all outliers before training. Why might this be exactly wrong?

<details><summary>Answer</summary>

In fraud/anomaly/churn problems, the outliers often ARE the signal — capping them deletes the very pattern you're trying to detect. Only remove/cap outliers that are clearly errors (negative age) or irrelevant to the task. ([feature-engineering.md](feature-engineering.md))
</details>

---

## Round 3 — Trees & Ensembles

**11.** Bagging vs boosting in two lines — and which one parallelizes across all CPU cores trivially?

<details><summary>Answer</summary>

Bagging (Random Forest): trees trained *independently* on random resamples, predictions averaged — embarrassingly parallel (`n_jobs=-1`). Boosting (XGBoost): trees trained *sequentially*, each fitting the previous ensemble's residuals — tree-building can't parallelize across trees. ([random-forest.md](random-forest.md), [gradient-boosting.md](gradient-boosting.md))
</details>

**12.** Why does Random Forest deliberately grow *deep* trees while gradient boosting uses *shallow* ones (depth 3–6)?

<details><summary>Answer</summary>

RF wants each tree to be a low-bias, high-variance expert — averaging cancels the variance. Boosting wants each tree to be a *weak* learner making a small correction — a deep tree would overfit the residuals in a few aggressive steps; many shallow trees with a small learning rate generalize better.
</details>

**13.** Your colleague sets `n_estimators=100` for XGBoost "because it's the default." What's the professional pattern instead?

<details><summary>Answer</summary>

Large `n_estimators` (e.g., 5000) + `early_stopping_rounds=50` + a validation set in `eval_set`. The data decides the tree count; guessing it means underfitting or overfitting. RF is the opposite: more trees never hurt, they just plateau. ([gradient-boosting.md](gradient-boosting.md))
</details>

**14.** Two features are near-duplicates (height_cm, height_inches). Random Forest's `feature_importances_` shows each at ~15%. What's the trap, and the fix?

<details><summary>Answer</summary>

Correlated features split the credit — the true importance (~30%) is halved across them, so both look mediocre and might get dropped. Fix: permutation importance, or drop one of the duplicates first. ([random-forest.md](random-forest.md))
</details>

**15.** A single decision tree gets train=100%, test=72%. List three hyperparameters that fix this and what each does.

<details><summary>Answer</summary>

`max_depth` (cap the levels — stop memorizing), `min_samples_leaf` (each leaf needs N+ samples — no leaf-per-data-point), `min_samples_split` (nodes need N+ samples to split). All force the tree to learn patterns instead of noise. ([decision-trees.md](decision-trees.md))
</details>

---

## Round 4 — Evaluation & Metrics

**16.** Fraud is 0.3% of transactions. Your model: 99.7% accuracy. The fraud team is furious. Reconstruct what happened and name the metrics you should have reported.

<details><summary>Answer</summary>

The model likely predicts "not fraud" always — 99.7% accuracy, 0% recall, catches nothing. Report precision, recall, F1, and PR-AUC (not ROC-AUC — it stays rosy under heavy imbalance). Also train with `class_weight='balanced'` and use stratified splits. ([model-evaluation.md](model-evaluation.md))
</details>

**17.** Disease screening: the business demands "miss at most 5% of true cases." Translate that into a metric constraint and describe the sklearn workflow to satisfy it.

<details><summary>Answer</summary>

Recall ≥ 0.95. Get probabilities with `predict_proba`, run `precision_recall_curve(y_val, probs)`, find the threshold where recall ≥ 0.95, report the precision you must accept there, and apply `(probs >= threshold)` instead of the default 0.5. ([logistic-regression.md](logistic-regression.md), [model-evaluation.md](model-evaluation.md))
</details>

**18.** Model A: train 91%, test 89%. Model B: train 99%, test 84%. Which ships, and what is each model's diagnosis?

<details><summary>Answer</summary>

Model A ships — higher test score AND a small train/test gap (generalizes). Model B is overfitting (15-point gap = high variance); its higher train accuracy is memorization, not skill. The train-vs-test gap is the single most useful diagnostic in ML. ([model-evaluation.md](model-evaluation.md))
</details>

**19.** You tuned hyperparameters by repeatedly checking test-set scores until they looked great. Why is your final number untrustworthy, and what should the data splits have been?

<details><summary>Answer</summary>

You optimized *for the test set* — it's no longer unseen, so the score is optimistically biased (manual leakage). Correct: train (learn) / validation (tune against repeatedly, or use CV) / test (touched ONCE, at the very end). ([model-evaluation.md](model-evaluation.md))
</details>

**20.** Why is random K-fold CV invalid for forecasting next month's sales, and what replaces it?

<details><summary>Answer</summary>

Random folds put future rows in training and past rows in test — the model trains on the future to predict the past, which can't happen in production. Use `TimeSeriesSplit`: always train on the past, validate on the next window. ([model-evaluation.md](model-evaluation.md))
</details>

---

## Round 5 — Cross-Cutting Concepts

**21.** KNN works great with 10 features, falls apart at 500. Explain the mechanism (not just the name) and two fixes.

<details><summary>Answer</summary>

Curse of dimensionality: to be "close" in 500-D, points must be close in *every* dimension at once — vanishingly unlikely, so nearest ≈ farthest and the K "nearest" neighbors are effectively random voters. Fixes: PCA/UMAP down to 10–20 dimensions first, or switch to a non-distance model (trees). ([knn.md](knn.md), [pca.md](pca.md))
</details>

**22.** PCA vs Lasso feature selection: both "reduce features." A stakeholder asks "so which original features matter?" — which technique can answer, and why can't the other?

<details><summary>Answer</summary>

Lasso can: it zeroes out unhelpful original features, so survivors are nameable ("age and income matter"). PCA can't: components are blends of ALL original features (PC1 = 0.58·height + 0.57·weight + …) — you can only say "2 components capture 85% of variance." ([pca.md](pca.md), [linear-regression.md](linear-regression.md))
</details>

**23.** Ring-shaped data: class A forms a circle around class B. Logistic regression fails. Name two fundamentally different ways to make this separable, and the mechanism of each.

<details><summary>Answer</summary>

(1) Kernel trick (SVM with RBF): implicitly lift points into a higher dimension (e.g., distance-from-center as a new axis) where a flat plane separates them. (2) Feature engineering: explicitly add `r = x² + y²` as a feature — now even logistic regression separates on a threshold of r. Same idea — one implicit, one manual. (Trees also work — they approximate the ring with box splits.) ([svm.md](svm.md), [feature-engineering.md](feature-engineering.md))
</details>

**24.** Your K-Means on 40 features gives silhouette 0.12 at every K from 2–10. List three distinct hypotheses and the action for each.

<details><summary>Answer</summary>

(1) Curse of dimensionality — distances are meaningless at 40-D → PCA to ~10 dims first. (2) Wrong cluster shape assumption — structure is non-spherical → try DBSCAN/GMM. (3) No real cluster structure in these features — profile the data, engineer features that capture what you actually care about (or accept there are no natural segments). ([k-means.md](k-means.md))
</details>

**25.** Write (from memory) the leakage-proof skeleton for: mixed numeric/categorical data → impute → scale/encode → Random Forest → 5-fold CV F1. Which objects guarantee the "no leakage" property?

<details><summary>Answer</summary>

```python
preprocessor = ColumnTransformer([
    ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                      ('scale', StandardScaler())]), numeric_cols),
    ('cat', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='missing')),
                      ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols),
])
pipe = Pipeline([('pre', preprocessor),
                 ('model', RandomForestClassifier(n_estimators=200, n_jobs=-1))])
scores = cross_val_score(pipe, X, y, cv=5, scoring='f1')
```

The `Pipeline` (+ `ColumnTransformer`) guarantees it: `cross_val_score` re-fits all transforms on each training fold only, transforming the validation fold with frozen parameters. This skeleton is **the** tabular-ML pattern — memorize it. ([scikit-learn.md](scikit-learn.md))
</details>

---

## Scoring yourself

| Score | Verdict |
|---|---|
| 23–25 | Interview-ready. Move on to deep learning / LLM material — review this folder via [flashcards.md](flashcards.md) weekly. |
| 18–22 | Solid. Re-read the "Build the Intuition From Zero" sections of the docs you missed; retake in 3 days. |
| < 18 | Foundations still forming — normal! Redo the per-doc self-checks first, then return here. |
