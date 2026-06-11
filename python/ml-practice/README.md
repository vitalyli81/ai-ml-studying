# ml-practice

Hands-on Python + ML exercises. A [uv](https://docs.astral.sh/uv/) project — Python 3.14, with numpy, pandas, matplotlib, and scikit-learn preinstalled.

```bash
uv run <path/to/script.py>   # run anything in this project
```

## Practice Projects

| Project | What you learn | Theory companion |
|---------|----------------|------------------|
| [linear-regression/](linear-regression/) | Python through ML: lists → functions → comprehensions → NumPy vectorization → classes → scikit-learn, building the same regression model three ways | [ml/linear-regression.md](../../ml/linear-regression.md) |
| [linear-regression/from-scratch.md](linear-regression/from-scratch.md) | The no-sklearn sequel: implement the closed-form solution, train/test split, R², and self-stopping gradient descent by hand (NumPy + pandas only) | [ml/linear-regression.md](../../ml/linear-regression.md) |
| [logistic-regression/](logistic-regression/) | Part 2: the same training loop becomes a classifier — boolean masks, np.clip, dicts as counters, the accuracy trap, and threshold tuning | [ml/logistic-regression.md](../../ml/logistic-regression.md) |
| [logistic-regression/from-scratch.md](logistic-regression/from-scratch.md) | The no-sklearn sequel: the `@` operator, axis= reductions, stratified split, per-class report — and L2 regularization built by hand | [ml/logistic-regression.md](../../ml/logistic-regression.md) |
| [decision-trees/](decision-trees/) | Part 3: no gradients — recursion, Counter, self-referencing dataclasses; grow a tree from scratch, then measure overfitting with sklearn's depth sweep | [ml/decision-trees.md](../../ml/decision-trees.md) |
| [decision-trees/from-scratch.md](decision-trees/from-scratch.md) | The no-sklearn sequel: your own module imports, defaultdict, feature importance via recursion accumulators, overfitting measured in nodes | [ml/decision-trees.md](../../ml/decision-trees.md) |
| [random-forest/](random-forest/) | Part 4: the overfitting fix — sets, pathlib + sys.path, bootstrap sampling, OOB scoring, and the jury vote that beats the lone expert | [ml/random-forest.md](../../ml/random-forest.md) |
| [gradient-boosting/](gradient-boosting/) | Part 5: sequential trees fitting residuals — regression trees via a two-line swap, early stopping with rollback-by-slicing, the learning-rate tradeoff measured | [ml/gradient-boosting.md](../../ml/gradient-boosting.md) |
| [svm/](svm/) | Part 6: the widest lane — hinge loss, subgradient masks, kernels as dict values, assert-as-test, the kernel trick performed by hand, and meshgrid boundary plots | [ml/svm.md](../../ml/svm.md) |
| [knn/](knn/) | Part 7: the model that doesn't train — argsort, broadcasting with None for pairwise distances, lazy-learning economics timed, the curse of dimensionality measured | [ml/knn.md](../../ml/knn.md) |
| [naive-bayes/](naive-bayes/) | Part 8: the first text model — string methods, dict comprehensions, Counter.update, star-unpacking, log-space underflow fixes, and a spam filter that matches sklearn exactly | [ml/naive-bayes.md](../../ml/naive-bayes.md) |
| [k-means/](k-means/) | Part 9: the first unsupervised model — generators with yield, the walrus operator, while-True convergence, n_init demystified as min(), and pandas groupby for naming clusters | [ml/k-means.md](../../ml/k-means.md) |
| [pca/](pca/) | Part 10: eigendecomposition with np.linalg.eigh, cumsum/searchsorted, sign conventions, real datasets (iris, digits), and image compression as a JPEG slider | [ml/pca.md](../../ml/pca.md) |
| [model-evaluation/](model-evaluation/) | Part 11 (capstone): build the judges — k-fold CV, ROC/AUC via trapezoid, PR-AUC, threshold-as-business-contract, and GridSearchCV demystified as itertools.product × CV | [ml/model-evaluation.md](../../ml/model-evaluation.md) |
| [feature-engineering/](feature-engineering/) | Part 12 (the pandas part): a genuinely messy CSV — inspection ritual, fillna + indicators, the encoding trap measured in R², one outlier worth −6.5 R², and the leak-proof ColumnTransformer pipeline | [ml/feature-engineering.md](../../ml/feature-engineering.md) |
