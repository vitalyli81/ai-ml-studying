# Scikit-learn

## TL;DR

Scikit-learn is Python's **universal ML toolkit** — a single, consistent API that covers preprocessing, training, evaluation, and tuning for almost every classical ML algorithm. Every estimator speaks the same three verbs: `fit()`, `predict()`, `transform()`. Once you learn those, switching algorithms is literally a one-line change. Pipelines let you chain steps together so the whole workflow — from raw data to predictions — is a single object you can train, save, and deploy.

> 💡 **Key Insight:** The biggest win of sklearn is not any single algorithm — it's the **uniform API**. Learn the pattern once; it works for 150+ models.

---

## The Mental Model

Think of sklearn like a **universal remote control**.

Every TV, soundbar, and streaming box used to have its own remote with different buttons. A universal remote has the same three buttons — power, volume, input — and they do the right thing no matter which device. Sklearn is that universal remote for ML: 150+ algorithms, all speaking the same three methods.

Mapping:
- Universal remote → sklearn API
- `power` button → `.fit(X, y)` (turn it on with data)
- `volume` → `.predict(X)` (run it)
- `input` → `.transform(X)` (preprocess)
- Different devices → different algorithms
- Remote codes → hyperparameters

You swap `LinearRegression()` for `RandomForestClassifier()` and nothing else in your code changes. That's the magic.

---

## Why It Exists

### The Problem Before

Each ML algorithm used to ship as its own library with its own API. Want to try 5 different classifiers? Learn 5 different interfaces, 5 different conventions for handling missing values, 5 different ways to save models. Pain.

### The Solution

Scikit-learn (started 2007, Google Summer of Code project) standardized everything around the **Estimator protocol**: every model implements `fit`, `predict`, and/or `transform`. Built on NumPy + SciPy, open source, ridiculously well-documented.

### What Changed

Sklearn became the default ML library for non-deep-learning tasks. Every introductory course teaches it. Every kaggle competition on tabular data uses it. And its API influenced everyone after: XGBoost, LightGBM, CatBoost, imbalanced-learn, and even parts of PyTorch's torchvision adopted the same `fit/transform` pattern so they'd slot into sklearn pipelines.

---

## Core Concepts

### 1. The Estimator Protocol (The One Pattern That Matters)

**One-line definition:** Every sklearn object implements a tiny, predictable interface — learn it once, apply it everywhere.

**The three core methods:**
```python
# fit: learn from data
model.fit(X_train, y_train)      # supervised (takes labels)
scaler.fit(X_train)              # unsupervised / preprocessor

# predict: make predictions (models only)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)   # classifiers — probabilities

# transform: reshape data (preprocessors, decomposers)
X_scaled = scaler.transform(X_test)

# fit_transform: shortcut — fit THEN transform on the same data
X_train_scaled = scaler.fit_transform(X_train)
```

**The golden rule:**
```
fit   → on training data ONLY
transform → on training AND test data
predict   → on test / new data
```

**Common misconception:** People call `fit_transform` on both train and test. That's leakage — test data influenced the scaler. Use `fit_transform` on train, `transform` on test.

---

### 2. The Dataset Shape

**One-line definition:** sklearn expects inputs as a 2D array `X` (samples × features) and labels as 1D `y`.

```
X (features):               y (labels):
[[5.1, 3.5, 1.4, 0.2],        [0,
 [4.9, 3.0, 1.4, 0.2],         0,
 [6.7, 3.1, 4.7, 1.5],  ...    1,
 ...]                          ...]
shape: (n_samples, n_features) shape: (n_samples,)
```

Accepts NumPy arrays, pandas DataFrames, sparse matrices. **Not** lists of dicts, not pandas Series as X.

```python
# ✅ Correct
X = df[['age', 'salary', 'tenure']]  # DataFrame with feature columns
y = df['churned']                     # Series with target

model.fit(X, y)

# ❌ Common mistake — single feature as 1D
X = df['age']              # pandas Series, 1D
model.fit(X, y)            # ValueError: expected 2D

# ✅ Fix — reshape to 2D
X = df[['age']]            # DataFrame (still 2D)
# or
X = df['age'].values.reshape(-1, 1)
```

---

### 3. Models: Classifiers, Regressors, Clusterers

**One-line definition:** Three families of estimators, picked by task type.

| Family | Task | Examples | Key methods |
|---|---|---|---|
| Classifier | Predict a category | `LogisticRegression`, `RandomForestClassifier`, `SVC` | `fit`, `predict`, `predict_proba` |
| Regressor | Predict a number | `LinearRegression`, `RandomForestRegressor`, `SVR` | `fit`, `predict` |
| Clusterer | Find groups (no labels) | `KMeans`, `DBSCAN` | `fit`, `predict` or `fit_predict` |
| Transformer | Reshape data | `StandardScaler`, `PCA`, `OneHotEncoder` | `fit`, `transform` |

**Swapping models is one line:**
```python
# Try logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Don't like it? Swap for random forest — everything else unchanged
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)

# Or gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()

# Same three lines work for all of them:
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(model.score(X_test, y_test))
```

---

### 4. Pipelines: Chaining Steps Together

**One-line definition:** A `Pipeline` glues preprocessing + model into a single object that acts like a normal estimator.

**Why it matters:** Without pipelines, you write `fit`/`transform` for every step manually on both train and test — error-prone and a common source of data leakage. With pipelines, sklearn handles it automatically.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)       # scales AND trains
preds = pipe.predict(X_test)     # scales AND predicts
```

Under the hood:
```
Training:
  X_train → scaler.fit_transform → X_train_scaled → model.fit(X_train_scaled, y_train)

Inference:
  X_test  → scaler.transform     → X_test_scaled  → model.predict(X_test_scaled)
```

No leakage. No bugs. One object to save and deploy.

---

### 5. ColumnTransformer: Different Preprocessing per Column

**One-line definition:** Apply different transforms to different columns in the same pipeline.

Real datasets have mixed types — numerics need scaling, categoricals need encoding, some columns should be dropped. `ColumnTransformer` handles this.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numeric_cols = ['age', 'salary']
categorical_cols = ['city', 'department']

numeric_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

categorical_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipe, numeric_cols),
    ('cat', categorical_pipe, categorical_cols)
])

full_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier())
])

full_pipeline.fit(X_train, y_train)
```

This is **the** pattern for tabular ML. Memorize it.

---

### 6. Hyperparameter Tuning: GridSearchCV & RandomizedSearchCV

**One-line definition:** Try many hyperparameter combinations with cross-validation; pick the best.

**GridSearchCV** — exhaustive search over every combination
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [5, 10, None]
}

grid = GridSearchCV(
    full_pipeline, param_grid,
    cv=5, scoring='f1', n_jobs=-1
)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
best_model = grid.best_estimator_
```

**Notice `model__n_estimators`** — the double underscore lets you tune params inside a pipeline step. `stepname__paramname`.

**RandomizedSearchCV** — sample N random combinations (faster, often as good)
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'model__n_estimators': randint(50, 500),
    'model__max_depth': randint(3, 20)
}

rand = RandomizedSearchCV(
    full_pipeline, param_dist, n_iter=50,
    cv=5, scoring='f1', n_jobs=-1, random_state=42
)
rand.fit(X_train, y_train)
```

**Rule of thumb:** Start with `RandomizedSearchCV` with 20–50 iterations. Upgrade to `GridSearchCV` only for final tuning around the best region.

---

### 7. Cross-Validation Utilities

**One-line definition:** Split data in smart ways so your scores are stable, not lucky.

```python
from sklearn.model_selection import (
    cross_val_score, cross_validate,
    KFold, StratifiedKFold, TimeSeriesSplit
)

# Quick single score
scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"{scores.mean():.3f} ± {scores.std():.3f}")

# Multiple metrics at once
results = cross_validate(model, X, y, cv=5,
                         scoring=['accuracy', 'f1', 'roc_auc'])

# For imbalanced classification — preserves class ratios
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# For time series — never trains on the future
tscv = TimeSeriesSplit(n_splits=5)
```

---

### 8. Persisting Models (Save / Load)

**One-line definition:** Use `joblib` to serialize a fitted model — the whole pipeline, preprocessing included.

```python
import joblib

# Save
joblib.dump(full_pipeline, 'model.joblib')

# Load later — in an API, a batch job, whatever
loaded = joblib.load('model.joblib')
preds = loaded.predict(new_data)
```

Because the pipeline includes preprocessing, the loaded model takes **raw data** and produces predictions — no need to recreate scalers/encoders at inference time. This is why pipelines are non-negotiable for production.

**Common misconception:** "I'll just save the model weights." Then you have to remember every preprocessing step manually at inference time, and version them, and keep them in sync. Pipelines solve this.

---

## How It Actually Works: Full Workflow

```
┌─────────────────────────────────────────────────┐
│ 1. LOAD DATA                                    │
│    df = pd.read_csv(...)                        │
│    X, y = df.drop('target', 1), df['target']    │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 2. SPLIT                                        │
│    X_train, X_test, y_train, y_test =           │
│        train_test_split(X, y, test_size=0.2,    │
│                         stratify=y)             │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 3. BUILD PIPELINE                               │
│    preprocessor = ColumnTransformer([...])      │
│    pipe = Pipeline([('pre', preprocessor),      │
│                     ('model', RandomForest())]) │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 4. TUNE                                         │
│    grid = GridSearchCV(pipe, params, cv=5)      │
│    grid.fit(X_train, y_train)                   │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 5. EVALUATE ON TEST (touched once)              │
│    best = grid.best_estimator_                  │
│    print(classification_report(y_test,          │
│                                 best.predict(X_test)))│
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│ 6. SAVE                                         │
│    joblib.dump(best, 'model.joblib')            │
└─────────────────────────────────────────────────┘
```

---

## Code in Practice

### Example 1: Minimal — Hello-World Classifier

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
```

### Example 2: Realistic — Full Pipeline on Mixed Data

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('customers.csv')
X, y = df.drop('churned', axis=1), df['churned']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

numeric_cols = X.select_dtypes(include='number').columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ]), numeric_cols),
    ('cat', Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_cols)
])

pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])

pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
print(classification_report(y_test, preds))
```

### Example 3: Production Pattern — Tune, Evaluate, Save, Serve

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import joblib

param_dist = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(3, 20),
    'model__min_samples_split': randint(2, 20)
}

search = RandomizedSearchCV(
    pipe, param_dist, n_iter=30, cv=5,
    scoring='f1', n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)

print(f"Best CV F1: {search.best_score_:.3f}")
print(f"Best params: {search.best_params_}")

# Final evaluation on held-out test set
best_model = search.best_estimator_
print(classification_report(y_test, best_model.predict(X_test)))

# Save — the full pipeline goes in one file
joblib.dump(best_model, 'churn_model.joblib')

# Later, in an API endpoint:
# model = joblib.load('churn_model.joblib')
# prediction = model.predict(raw_customer_dataframe)
```

---

## Gotchas & Pitfalls

- ❌ **"I'll fit the scaler on all data first"** → ✅ Fit on train only. Use `Pipeline`.
- ❌ **"Single feature as `df['col']`"** → ✅ sklearn needs 2D. Use `df[['col']]` or `.reshape(-1, 1)`.
- ❌ **"`GridSearchCV` with 6 params × 5 values each"** → ✅ That's 15,625 combinations × 5 folds = 78,125 fits. Use `RandomizedSearchCV`.
- ❌ **"Save model weights only"** → ✅ Save the full pipeline with `joblib` — preprocessing included.
- ❌ **"`predict` when I need probabilities"** → ✅ Use `predict_proba` for probabilities; `predict` gives hard labels.
- ❌ **"One-hot encode everything"** → ✅ Tree-based models often work fine with ordinal encoding and run faster.
- ❌ **"Forgot `handle_unknown='ignore'` on OneHotEncoder"** → ✅ Without it, a new category in production crashes the pipeline.
- ❌ **"Random K-fold on time-series"** → ✅ Use `TimeSeriesSplit` — random folds train on the future.

---

## When to Use / When NOT to Use

### Use sklearn when:
- Working with **tabular data** (CSVs, SQL, pandas DataFrames)
- Classical ML algorithms suffice (regression, trees, SVM, clustering)
- You need a quick baseline before considering deep learning
- Building end-to-end pipelines that include preprocessing
- Prototyping — speed of iteration is unmatched

### Don't reach for sklearn when:
- Working with **images, audio, or raw text at scale** — use PyTorch / HuggingFace
- Training on **huge datasets** (100M+ rows) — use Spark ML, Dask-ML, or cloud ML services
- You need **GPU acceleration** — sklearn is CPU-only (use RAPIDS cuML for GPU)
- Serving **LLMs or transformers** — sklearn doesn't cover these
- You need **distributed training** — sklearn is single-machine

---

## Related Concepts (The Map)

- **If you know jQuery, sklearn is like jQuery for ML** — a consistent API over many implementations, glued into workflows.
- **Pipelines** are like function composition — chain transforms so output of one is input to the next.
- **XGBoost / LightGBM / CatBoost** implement the sklearn API, so they drop into pipelines unchanged.
- **HuggingFace Transformers** have a `Pipeline` concept too, deliberately named for familiarity.
- **PyTorch / TensorFlow** are for deep learning — different API, different problems, complementary to sklearn.
- **MLflow / W&B** plug into sklearn for experiment tracking — log pipeline + params + metrics per run.

---

## Cheat Sheet

### Key imports (memorize these)
```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
```

### The core API
```python
# Every estimator
estimator.fit(X, y)              # learn
estimator.predict(X)             # predict labels
estimator.predict_proba(X)       # classifier probabilities
estimator.score(X, y)            # default scoring
estimator.get_params()           # see hyperparameters
estimator.set_params(**kwargs)   # override them

# Transformers add
transformer.transform(X)
transformer.fit_transform(X)
```

### The 3 things that matter most
1. **Uniform API** — `fit` / `predict` / `transform` work for everything. Learn them once.
2. **Pipelines are non-negotiable** — they prevent leakage and make models deployable.
3. **`fit` on train, `transform` on test** — this one rule saves you from most bugs.

---

## Self-Check Questions

1. What's the difference between `fit_transform` and `transform`, and when do you use each?
2. Why wrap preprocessing in a `Pipeline` instead of doing it manually?
3. Your model expects a 2D input but you're passing a single feature — how do you fix it?
4. What's the purpose of the double underscore in `'model__n_estimators'`?
5. Why save the whole pipeline instead of just the trained model?

<details>
<summary>Answers</summary>

1. `fit_transform` learns parameters (like mean/std) AND applies them — use on training data. `transform` only applies the already-learned parameters — use on test and new data. Using `fit_transform` on test data leaks test statistics into your model.
2. Pipelines apply preprocessing only to training folds during CV (preventing leakage), make the whole workflow a single serializable object, and save you from repeating transforms manually on train and test.
3. Reshape to 2D: use `X[['col']]` (DataFrame syntax keeps it 2D) or `X['col'].values.reshape(-1, 1)`.
4. It's sklearn's syntax for addressing parameters inside pipeline steps. `'model__n_estimators'` means "set `n_estimators` on the step named `model`." Works for nested steps and ColumnTransformers too.
5. The pipeline includes all preprocessing (scalers, encoders, imputers). Saving just the model means you'd have to recreate and re-fit every preprocessing step at inference time — error-prone and a common source of production bugs.

</details>

---

## Go Deeper

1. **scikit-learn User Guide** (scikit-learn.org/stable/user_guide.html) — the gold-standard ML reference. Read it cover-to-cover over a few weeks; you'll be better than 90% of practitioners.
2. **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron** (O'Reilly, 3rd ed.) — the single best book for learning sklearn in real projects. Chapters 1-8 alone are worth the price.
3. **sklearn API cheat sheet** (scikit-learn.org/stable/tutorial/machine_learning_map/) — the official "which algorithm should I use" flowchart. Print and pin it.
4. **Sebastian Raschka's "Python Machine Learning"** — clear, opinionated, with deep explanations of *why* each sklearn API is shaped the way it is.
5. **Kaggle's "Intro to Machine Learning" and "Intermediate ML"** micro-courses — free, hands-on, and use sklearn throughout. Finish both in a weekend.
