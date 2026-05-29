# Feature Engineering

## TL;DR

Feature engineering is the craft of transforming **raw data into inputs a model can learn from**. Models don't understand "Tuesday" or "New York" or "$1,200" — they understand numbers on a reasonable scale. Good features often matter more than fancy algorithms: a linear model with great features beats a neural network with garbage inputs. The three pillars are: **encoding** (turn categories into numbers), **scaling** (put numbers on comparable ranges), and **selection** (keep the features that carry signal, drop the rest).

> 💡 **Key Insight:** Data scientists say "garbage in, garbage out" for a reason. 80% of a real ML project is cleaning and transforming data. Model.fit() is the easy part.

---

## The Mental Model

Think of **prepping ingredients before cooking**.

You don't throw a whole onion, unpeeled garlic, and a frozen chicken into a pan. You chop, peel, thaw, measure, and season — *then* you cook. The recipe (model) assumes ingredients arrive in a specific form. Feature engineering is the chopping board of ML.

Mapping:
- Raw ingredients → raw data (strings, dates, categories, missing values)
- Chopping → encoding (categories → numbers)
- Measuring consistently → scaling (everything on the same range)
- Removing rotten parts → handling missing values / outliers
- Seasoning → creating new features (interactions, ratios, aggregates)
- Cooking → `model.fit()`

A Michelin chef with bad ingredients still makes a bad meal. Same with ML.

---

## Build the Intuition From Zero

Feature engineering is a bag of techniques, but two ideas cause more silent bugs than anything else: **why you can't just number your categories, and why "fit on train, transform on test" is sacred.** Get these two and you'll avoid the mistakes that quietly wreck real projects.

### Idea 1: Numbering categories lies to the model

You have a `color` column: red, blue, green. The obvious move — `red=1, blue=2, green=3` — is a trap. Watch what the model now believes:

```
red=1, blue=2, green=3
              ↑
   the model does MATH on these numbers:
   "green (3) > blue (2) > red (1)"        ← you invented a ranking!
   "the average of red and green is blue"  ← (1+3)/2 = 2 = blue. Nonsense.
   "green is 3× red"                        ← also nonsense
```

Colors have no order, no midpoint, no multiples — but the integers you assigned scream all three. A linear model or KNN will faithfully act on this fiction and learn garbage.

The fix is **one-hot encoding**: give each category its own yes/no column, so no fake ordering can sneak in:

```
            is_red  is_blue  is_green
  red    →    1       0        0
  blue   →    0       1        0
  green  →    0       0        1
   → every category is equidistant from every other. No invented ranking. ✓
```

> 💡 **The rule:** if the categories have a *real* order — `small < medium < large` — then numbering them `1,2,3` (ordinal encoding) is correct, because the order is true. If they don't — colors, cities, payment methods — use one-hot. The question is always: *does arithmetic on these numbers mean anything?*

### Idea 2: Why "fit on train, transform on test" is sacred (data leakage)

Every transformation that *learns something from the data* — the mean for filling missing values, the min/max for scaling, the categories for encoding — must learn it from the **training set only**, then apply that frozen knowledge to the test set. Mixing them is **data leakage**, and it silently inflates your score.

Here's the leak in action with scaling:

```
WRONG — scale using all the data at once:
  the scaler peeks at the test set's values to compute the mean/range
  → test information has bled into training
  → your evaluation looks great, then production is worse. You won't know why.

RIGHT — fit on train, apply to test:
  scaler.fit(X_train)        ← learn mean & range from TRAINING data only
  scaler.transform(X_train)  ← apply it
  scaler.transform(X_test)   ← apply the SAME frozen numbers to test
  → test set stays a true "unseen" exam
```

The mental model: **the test set is a final exam.** Any statistic you compute from it is like peeking at the exam while studying — your practice scores soar and mean nothing. This is exactly why scikit-learn separates `.fit()` from `.transform()`, and why wrapping steps in a `Pipeline` (shown later) is the safe default — it makes leakage almost impossible by construction.

The encoding, scaling, and imputation sections below are the specific tools; these two principles govern all of them.

---

## Why It Exists

### The Problem Before

Early ML assumed "clean numeric data." Reality: data is messy — strings, missing values, inconsistent units, dates, unstructured text, categorical fields with 10,000 unique values. Feed any of that raw into a model and you get either an error or nonsense predictions.

### The Solution

A set of transformations that turn any real-world dataset into a numeric matrix the algorithm can optimize over. And once you go beyond survival — creating *new* features (ratios, interactions, aggregates) — you often encode domain knowledge that lifts model performance more than any hyperparameter tuning ever will.

### What Changed

For tabular data, feature engineering is still the #1 predictor of project success. For deep learning (images, text, audio), models learn features themselves — but for LLMs you're back to engineering prompts, retrieval chunks, and embeddings. The skill transfers.

---

## Core Concepts

### 1. Handling Missing Values

**One-line definition:** Decide what to do when a cell is empty — drop, fill, or flag.

**Analogy:** A survey with some questions left blank. You can throw out the whole survey (drop row), guess the answer (impute), or add a "refused to answer" column (indicator).

**Strategies:**
| Strategy | When to use | Tradeoff |
|---|---|---|
| Drop rows | Missing < 5% and random | Simple but loses data |
| Drop column | Column is mostly empty (>50%) | Gone forever |
| Fill with mean/median | Numeric, roughly normal | Shrinks variance |
| Fill with mode | Categorical | Over-represents majority |
| Forward/back-fill | Time series | Only valid if ordered |
| Model-based (KNN, iterative) | Important feature, non-random missing | Expensive but often best |
| Add "is_missing" indicator | Missingness itself is informative | Doubles column count |

**Code:**
```python
from sklearn.impute import SimpleImputer
import pandas as pd

df = pd.DataFrame({'age': [25, None, 30, None, 45]})

# Mean imputation
imputer = SimpleImputer(strategy='mean')
df['age'] = imputer.fit_transform(df[['age']])

# Add missing indicator (often helps!)
df['age_was_missing'] = df['age'].isna().astype(int)
```

**Common misconception:** "Just fill with 0." That's lying to your model — 0 has meaning (zero salary ≠ unknown salary). Use `NaN` or a dedicated sentinel + indicator.

---

### 2. Encoding Categorical Variables

**One-line definition:** Turn text labels into numbers without inventing false relationships.

**The four main techniques:**

**a) One-Hot Encoding** — one column per category, 1 if present
```
color       →   color_red  color_blue  color_green
"red"       →      1          0           0
"blue"      →      0          1           0
"green"     →      0          0           1
```
✅ Use when: low cardinality (< ~15 categories), no ordinal relationship
❌ Avoid when: 1000s of categories (explosion of columns)

**b) Ordinal Encoding** — integers preserving order
```
size   →   size_encoded
"S"    →       1
"M"    →       2
"L"    →       3
"XL"   →       4
```
✅ Use when: there's a natural order (small → large, low → high)
❌ Avoid when: categories have no order (colors, cities — gives model false "bigger-than" signal)

**c) Target Encoding** — replace category with the mean target for that category
```
city       →   city_encoded (mean price)
"NYC"      →      1.2M
"SF"       →      1.5M
"Austin"   →      0.7M
```
✅ Use when: high cardinality categorical with predictive signal (zip codes, product IDs)
⚠️ **Watch for leakage:** compute means on training fold only, then apply to validation.

**d) Embeddings** — learned dense vectors (deep learning territory)
```
"NYC"    →   [0.2, -0.5, 0.8, ...]  (32-dim vector)
"SF"     →   [0.3, -0.4, 0.7, ...]
```
✅ Use when: very high cardinality (user IDs, products), neural nets
❌ Overkill for small tabular data — stick with one-hot or target encoding

**Code:**
```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd

df = pd.DataFrame({'color': ['red', 'blue', 'green'], 'size': ['S', 'M', 'L']})

# One-hot for nominal
ohe = OneHotEncoder(sparse_output=False, drop='first')
color_encoded = ohe.fit_transform(df[['color']])

# Ordinal for ordered
oe = OrdinalEncoder(categories=[['S', 'M', 'L', 'XL']])
df['size_encoded'] = oe.fit_transform(df[['size']])
```

**Common misconception:** People label-encode colors as red=1, blue=2, green=3 and feed that to a linear model. Now the model thinks "green > red" — a relationship you invented. Use one-hot instead.

---

### 3. Feature Scaling

**One-line definition:** Rescale numeric features so no single feature dominates by magnitude.

**Why it matters:** Imagine salary ([30k, 200k]) and age ([20, 80]) in the same model. For distance-based algorithms (KNN, K-Means, SVM), salary's numbers dwarf age's — the model effectively ignores age. Scaling fixes this.

**Three common scalers:**

**a) StandardScaler (z-score)** — mean 0, std 1
```
z = (x - mean) / std
```
✅ Default choice. Works for most algorithms. Assumes roughly normal distribution.

**b) MinMaxScaler** — rescale to [0, 1]
```
x' = (x - min) / (max - min)
```
✅ Use when: bounded output needed (neural nets with sigmoid), no extreme outliers.
❌ Sensitive to outliers — one extreme value shrinks everything else to near 0.

**c) RobustScaler** — uses median and IQR, not mean/std
```
x' = (x - median) / IQR
```
✅ Use when: data has outliers. Outliers don't distort the scale.

**Which algorithms need scaling?**
| Needs scaling | Doesn't care |
|---|---|
| KNN, SVM (distance-based) | Decision Trees |
| Linear/Logistic Regression (gradient descent converges faster) | Random Forest |
| Neural Networks | Gradient Boosting (XGBoost, LightGBM) |
| K-Means, PCA | Gaussian Naive Bayes |

> ⚠️ **Multinomial / Bernoulli Naive Bayes** expect **non-negative counts** (word frequencies). Never apply `StandardScaler` before them — it produces negative values and breaks the model. If you must rescale, use `MinMaxScaler` or leave raw counts / TF-IDF alone.

**Code:**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit AND transform on train
X_test_scaled = scaler.transform(X_test)         # only transform on test
```

**Common misconception:** People fit the scaler on the full dataset before splitting. That's **data leakage** — test statistics have bled into training. Always `fit` on train only.

---

### 4. Handling Outliers

**One-line definition:** Decide what to do about extreme values that distort your model.

**Detection:**
```python
import numpy as np

# IQR method (robust)
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers = data[(data < lower) | (data > upper)]

# Z-score method (assumes normal)
from scipy import stats
z_scores = np.abs(stats.zscore(data))
outliers = data[z_scores > 3]
```

**Strategies:**
- **Remove** — only if clearly an error (negative age, 500-year-old person)
- **Cap/Winsorize** — clip to the 1st/99th percentile
- **Transform** — log/sqrt transform compresses the tail
- **Keep** — if outliers are real signal (fraud IS the outlier!)

**Common misconception:** People reflexively remove outliers. In fraud, churn, failure prediction — **the outliers ARE your target**. Removing them destroys the signal.

---

### 5. Feature Creation (The Creative Part)

**One-line definition:** Build new features from existing ones using domain knowledge.

This is where you beat the competition. The model can only learn from features you give it — if `price_per_sqft` is what actually matters, giving it `price` and `sqft` separately might not be enough (especially for linear models).

**Common patterns:**

**a) Ratios & differences**
```python
df['price_per_sqft'] = df['price'] / df['sqft']
df['days_since_signup'] = (today - df['signup_date']).dt.days
df['profit_margin'] = (df['revenue'] - df['cost']) / df['revenue']
```

**b) Interactions**
```python
df['location_x_size'] = df['is_downtown'] * df['sqft']
# Polynomial features via sklearn
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
```

**c) Datetime decomposition**
```python
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['timestamp'].dt.month
```

**d) Aggregations (group-level features)**
```python
# Avg price per category — captures market context
df['avg_price_by_category'] = df.groupby('category')['price'].transform('mean')
df['price_vs_category_avg'] = df['price'] / df['avg_price_by_category']
```

**e) Binning (numeric → categorical)**
```python
df['age_bucket'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=['kid', 'young', 'mid', 'senior'])
```

**f) Text features**
```python
df['title_word_count'] = df['title'].str.split().str.len()
df['has_exclamation'] = df['text'].str.contains('!').astype(int)
```

**The rule:** if you can think of a question like *"Is this above or below average?"*, *"How old is this?"*, *"How does X compare to Y?"* — encode the answer as a feature.

---

### 6. Feature Selection

**One-line definition:** Keep features that carry signal, drop ones that don't — or worse, add noise.

**Why it matters:** More features ≠ better. Irrelevant features slow training, cause overfitting, and make models harder to interpret. The "curse of dimensionality" is real.

**Three main approaches:**

**a) Filter methods** — score each feature independently of the model
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)  # keep top 10 by ANOVA F-score
X_selected = selector.fit_transform(X, y)
```
✅ Fast. ❌ Ignores feature interactions.

**b) Wrapper methods** — train models on subsets, pick best (expensive)
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe = RFE(LogisticRegression(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
```
✅ Considers interactions. ❌ Slow with many features.

**c) Embedded methods** — selection happens inside model training
```python
# L1 regularization (Lasso) drives unhelpful coefficients to exactly 0
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1).fit(X, y)
important = X.columns[model.coef_ != 0]

# Tree-based importance
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier().fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
```
✅ Free with the model. ✅ Accounts for interactions. Usually the best default.

**Common misconception:** "More features = more information." Nope. Noise features add variance, slow training, and make models overfit. Less is often more.

---

### 7. Preventing Data Leakage

**One-line definition:** Leakage happens when information from the future (or the test set) sneaks into training features.

**Classic leaks:**
- **Fitting scalers/encoders on the full dataset before splitting** → test stats leak into training
- **Target encoding without fold-safe CV** → the target leaks into its own feature
- **Features that implicitly contain the target** (e.g., `total_spent_after_signup` when predicting churn)
- **Future information in time-series features** — rolling averages that peek forward

**The safe pattern:** do ALL transformations inside a `Pipeline`, fit only on training folds.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', LogisticRegression())
])

# Now cross_val_score fits the scaler separately on each fold — no leak
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5)
```

**If your model scores 0.99 and everyone else scores 0.70 — check for leakage first.**

---

## How It Actually Works: End-to-End Example

Predicting housing prices from raw CSV:

```
Step 1: LOAD raw data
  price, sqft, bedrooms, city, built_year, last_sold, condition, ...

Step 2: INSPECT
  - 12% missing in `last_sold`
  - `city` has 400 unique values
  - `built_year` ranges 1880–2024
  - `price` has 3 extreme outliers ($100M+)

Step 3: HANDLE MISSING
  - Fill `last_sold` with median + add `last_sold_was_missing` flag

Step 4: ENCODE CATEGORICAL
  - `condition` ordinal: poor=1, fair=2, good=3, excellent=4
  - `city` target-encoded (with fold-safe CV)

Step 5: CREATE FEATURES
  - `age = current_year - built_year`
  - `price_per_sqft` (as an engineered input for related predictions)
  - `years_since_last_sold`

Step 6: HANDLE OUTLIERS
  - Cap `price` at 99th percentile (business decision: not modeling mega-mansions)

Step 7: SCALE NUMERIC
  - StandardScaler on numeric columns

Step 8: SPLIT (stratified if classification, random for regression)
  - 80/20 train/test

Step 9: WRAP in Pipeline to prevent leakage, feed to model
```

Every step above is part of feature engineering. The `model.fit()` call is the *last* 5% of the work.

---

## Code in Practice

### Example 1: Minimal — Clean & Encode a Small Dataset

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    'age': [25, None, 30, 45, 22],
    'city': ['NYC', 'SF', 'NYC', None, 'Austin'],
    'salary': [50000, 80000, None, 95000, 45000]
})

# Impute
df['age'] = SimpleImputer(strategy='median').fit_transform(df[['age']])
df['salary'] = SimpleImputer(strategy='median').fit_transform(df[['salary']])
df['city'] = df['city'].fillna('unknown')

# Encode
city_encoded = pd.get_dummies(df['city'], prefix='city')
df = pd.concat([df.drop('city', axis=1), city_encoded], axis=1)

# Scale numeric
scaler = StandardScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])

print(df)
```

### Example 2: Realistic — ColumnTransformer + Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

numeric_cols = ['age', 'salary', 'tenure']
categorical_cols = ['city', 'department', 'role']

numeric_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

full_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classify', RandomForestClassifier())
])

# One call — handles all feature engineering AND training, leak-free
full_pipeline.fit(X_train, y_train)
preds = full_pipeline.predict(X_test)
```

### Example 3: Creating Domain Features for a Sales Dataset

```python
import pandas as pd

df = pd.read_csv('sales.csv', parse_dates=['timestamp', 'signup_date'])

# Temporal features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['timestamp'].dt.month

# Customer age (days since signup)
df['customer_age_days'] = (df['timestamp'] - df['signup_date']).dt.days

# Ratios
df['discount_pct'] = df['discount'] / df['list_price']
df['unit_price'] = df['total'] / df['quantity']

# Group aggregations
df['avg_spend_by_customer'] = df.groupby('customer_id')['total'].transform('mean')
df['spend_vs_avg'] = df['total'] / df['avg_spend_by_customer']

# Binning
df['spend_tier'] = pd.qcut(df['total'], q=4, labels=['low', 'mid', 'high', 'vip'])
```

---

## Gotchas & Pitfalls

- ❌ **"Fit scaler on the whole dataset"** → ✅ Fit on train only. Use `Pipeline` to enforce this automatically.
- ❌ **"Label-encode unordered categories"** → ✅ Use one-hot for nominal (colors, cities), ordinal only when order exists.
- ❌ **"Fill missing numerics with 0"** → ✅ Use median/mean + an `is_missing` indicator. 0 is a real value with meaning.
- ❌ **"More features = better"** → ✅ Noise features cause overfitting. Drop features with zero importance.
- ❌ **"Remove all outliers"** → ✅ For fraud/churn/anomaly, outliers ARE the signal. Investigate first.
- ❌ **"Target-encode naively"** → ✅ Do it inside CV folds or use smoothing; otherwise target leaks.
- ❌ **"Skip feature engineering, use deep learning"** → ✅ For tabular data, XGBoost + good features still beats neural nets.

---

## When to Use / When NOT to Use

### Always do feature engineering when:
- Working with tabular data (CSVs, databases, spreadsheets)
- Dataset has mixed types (text, dates, categories, numerics)
- Model is linear or tree-based
- Business domain knowledge can be encoded (ratios, buckets, flags)

### Less critical when:
- Using deep learning on raw modalities (images, audio, raw text for transformers)
- Data is already clean, scaled, numeric (rare in practice)
- Using models that are scale/encoding-invariant (tree-based)
- Running a quick prototype (but clean for the real run)

---

## Related Concepts (The Map)

- **If you know data validation (TypeScript types, JSON schemas), encoding/imputation is similar — normalize inputs to a clean shape before passing downstream.**
- **Pipelines** are the enforcement mechanism — they encode your feature steps so they replay identically at inference.
- **Model evaluation** depends on feature engineering — leak once, and your test scores lie.
- **Embeddings** (in deep learning / NLP) are learned feature engineering — the model creates features you used to hand-craft.
- **Prompt engineering** for LLMs is a kind of feature engineering — you're shaping inputs so the model performs better.

---

## Cheat Sheet

### Key terms
- **Feature** — a single column / input variable
- **Encoding** — categories → numbers
- **Scaling** — numbers → comparable ranges
- **Imputation** — filling missing values
- **Leakage** — test info accidentally in training
- **Cardinality** — # of unique values in a categorical column

### The decision tree
```
Is column numeric?
├── Yes
│   ├── Has missing? → impute (median usually)
│   ├── Has outliers? → RobustScaler or cap
│   └── Otherwise → StandardScaler
└── No (categorical)
    ├── Low cardinality (<15)? → OneHotEncoder
    ├── Ordinal (size, rank)? → OrdinalEncoder
    ├── High cardinality + predictive? → Target encoding (fold-safe)
    └── Huge cardinality (user IDs)? → Embeddings (deep learning)
```

### The 3 things that matter most
1. **Always fit transforms on train only.** Use `Pipeline` + `ColumnTransformer`.
2. **Encoding choice depends on whether order exists.** One-hot for nominal, ordinal for ordered.
3. **Great features > fancy models.** Spend time here first.

---

## Self-Check Questions

1. You have a `city` column with 500 unique values and strong predictive signal. How do you encode it?
2. Why is fitting `StandardScaler` on the full dataset before splitting a bug?
3. Your tree-based model ignores feature scaling — why?
4. When should you keep outliers instead of removing them?
5. Your model scores 0.99 F1 and the next team's best is 0.72. What's your first suspicion?

<details>
<summary>Answers</summary>

1. Target encoding (with fold-safe cross-validation to avoid leakage) or embeddings if using deep learning. One-hot would create 500 columns — usually too many.
2. Data leakage — the scaler learned the mean/std using test data. In production, your test set's stats aren't available, so you'd be scaling with different numbers.
3. Decision trees split on thresholds (`x > 0.5`). Rescaling doesn't change the *ordering* of values, only their magnitudes, so splits are identical.
4. When outliers ARE the target: fraud detection, defect detection, churn, anomaly detection. Removing them destroys the signal.
5. Data leakage. Investigate target-dependent features, fit-on-full-data transforms, and features that couldn't plausibly exist at prediction time.

</details>

---

## Go Deeper

1. **"Feature Engineering for Machine Learning" by Alice Zheng & Amanda Casari** (O'Reilly, 2018) — the canonical book. Short, practical, covers every transformation you'll use. Worth reading cover-to-cover.
2. **Kaggle Learn — Feature Engineering micro-course** (kaggle.com/learn/feature-engineering) — free, hands-on notebooks. Best practical intro if you want to learn by doing.
3. **scikit-learn preprocessing docs** (scikit-learn.org/stable/modules/preprocessing.html) — definitive reference for every scaler, encoder, imputer. Bookmark it.
4. **"Feature Engineering and Selection" by Kuhn & Johnson** (free online at feat.engineering) — deep, statistical, rigorous. Read when you want the *why* behind each technique.
5. **Abhishek Thakur's Kaggle walkthroughs on YouTube** — watch how a 4x Grandmaster does feature engineering in real competitions. You'll absorb patterns you can't learn from books.
