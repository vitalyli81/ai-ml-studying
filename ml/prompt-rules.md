# Classical Machine Learning — Doc Generation

> This folder uses the canonical doc prompt. **See [../prompt-rules.md](../prompt-rules.md)** for the full template and rules.

## How to use it

When generating a new doc in this folder, copy the prompt from `../prompt-rules.md` and set:

```
TOPIC: <one of the topics below, or any classical ML subtopic>
```

Save the result as `<topic-slug>.md` in this folder, then link it from [README.md](README.md).

## Topics covered in this folder

- **Supervised Learning** — linear/logistic regression, decision trees, SVMs, random forests, gradient boosting, KNN, Naive Bayes
- **Unsupervised Learning** — K-means clustering, PCA dimensionality reduction
- **Model Evaluation** — cross-validation, bias-variance tradeoff, precision/recall/F1, ROC/AUC
- **Scikit-learn** — unified `fit`/`predict`/`transform` API, Pipelines
- **Feature Engineering** — encoding, scaling, selection techniques

See [README.md](README.md) for the recommended reading order.
