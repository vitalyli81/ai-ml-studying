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
