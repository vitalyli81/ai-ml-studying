# Machine Learning

## What Is Machine Learning?

Machine learning is when a computer **learns patterns from data** instead of being explicitly programmed. You don't write rules — you show examples, and the algorithm figures out the rules on its own.

### Traditional Programming vs ML

```
Traditional Programming:
  Input: DATA + RULES     →  Output: ANSWERS
  "If temperature > 30 and humidity > 80, then rain"

Machine Learning:
  Input: DATA + ANSWERS   →  Output: RULES
  Give thousands of weather records with outcomes
  → ML figures out: "temperature > 28 and humidity > 75 and wind < 10 → rain"
```

### Frontend Analogy

Think of it like this:

```javascript
// Traditional: you write all the rules manually
function isSpam(email) {
  if (email.includes('free money')) return true;
  if (email.includes('click here')) return true;
  if (email.includes('winner')) return true;
  // ... 1000 more rules you have to think of
  return false;
}

// Machine Learning: the model learns rules from examples
const model = trainModel(thousandsOfLabeledEmails);
model.predict(newEmail); // → spam or not spam
// It found rules you'd never think of writing manually
```

## The Three Types of Machine Learning

### 1. Supervised Learning (Most Common)

You give the model **labeled data** — inputs AND correct answers. It learns the mapping.

```
Training data:
  [photo of cat] → "cat"
  [photo of dog] → "dog"
  [photo of cat] → "cat"
  ...10,000 more

Model learns the pattern. Then:
  [new photo] → "cat" (87% confident)
```

**Two subtypes:**
- **Classification** → predict a category (spam/not spam, cat/dog/bird)
- **Regression** → predict a number (house price, temperature, salary)

**Algorithms in this folder:** Linear Regression, Logistic Regression, Decision Trees, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting

### 2. Unsupervised Learning

You give the model **data without labels**. It finds hidden structure on its own.

```
Training data:
  [customer data] → ???
  [customer data] → ???
  ...

Model discovers:
  "I found 3 natural groups!"
  Group A: Young, high-spending, tech products
  Group B: Older, budget-conscious, essentials
  Group C: Mid-age, seasonal shoppers
```

**Algorithms in this folder:** K-Means (clustering), PCA (dimensionality reduction)

### 3. Reinforcement Learning (Not Covered Here)

An agent learns by **trial and error** in an environment, getting rewards or penalties. Used for games (AlphaGo), robotics, and self-driving cars. Important but less common in day-to-day AI engineering.

## The ML Workflow

Every ML project follows these steps:

```
1. COLLECT DATA        → gather relevant data
2. EXPLORE & CLEAN     → understand it, handle missing values, outliers
3. FEATURE ENGINEERING → select/create the right input features
4. SPLIT DATA          → training set (80%) + test set (20%)
5. TRAIN MODEL         → fit the algorithm on training data
6. EVALUATE            → test on unseen data (accuracy, F1, etc.)
7. TUNE                → adjust hyperparameters, try other models
8. DEPLOY              → put the model in production
```

## Key Concepts You Need to Know

### Train / Test Split

Never evaluate a model on the same data you trained it on — that's like grading a student on the exact questions they practiced.

```
All Data (100%)
├── Training Set (80%) → model learns from this
└── Test Set (20%)     → evaluate on this (model never sees it during training)
```

### Overfitting vs Underfitting

```
Underfitting:          Just Right:           Overfitting:
Model too simple       Good balance          Model too complex
Misses the pattern     Captures the pattern  Memorizes the noise
   ____                  ___                   ∿∿∿∿∿∿
       ____             /   \                 /\  /\  /\
           \_          /     \___            /  \/  \/  \

Train: 60%            Train: 88%            Train: 99%
Test:  58%            Test:  85%            Test:  65%
```

### Bias-Variance Tradeoff

- **High bias** = model is too simple, underfits (e.g., linear model for curved data)
- **High variance** = model is too complex, overfits (e.g., deep tree that memorizes data)
- **Goal** = find the sweet spot in between

### Feature Scaling

Many algorithms need features on the same scale:

```
Before scaling:                After scaling (StandardScaler):
  Salary: [30000, 120000]       Salary: [-1.2, 1.8]
  Age:    [20, 65]              Age:    [-1.5, 1.5]

Algorithms that NEED scaling: KNN, SVM, Linear/Logistic Regression, Neural Networks
Algorithms that DON'T:        Decision Trees, Random Forest, Gradient Boosting
```

## How to Pick an Algorithm

```
Is your target a NUMBER or CATEGORY?
│
├── NUMBER (regression):
│   ├── Start with → Linear Regression
│   ├── Better accuracy → Random Forest / Gradient Boosting
│   └── Need interpretability → Decision Tree
│
└── CATEGORY (classification):
    ├── Text data → Naive Bayes (baseline) → then Transformers
    ├── Tabular data:
    │   ├── Start with → Logistic Regression
    │   ├── Better accuracy → Random Forest / XGBoost
    │   └── Small data, many features → SVM
    ├── Need to find groups (no labels) → K-Means
    └── Too many features → PCA first, then classify
```

## Docs in This Folder

Read in this order for the best learning path:

### Algorithms

| # | File | Type | One-Line Summary |
|---|------|------|-----------------|
| 1 | [linear-regression.md](linear-regression.md) | Regression | Fit a straight line to predict numbers |
| 2 | [logistic-regression.md](logistic-regression.md) | Classification | Predict probabilities for yes/no decisions |
| 3 | [decision-trees.md](decision-trees.md) | Both | Flowchart of yes/no questions |
| 4 | [random-forest.md](random-forest.md) | Ensemble | Many trees voting together |
| 5 | [gradient-boosting.md](gradient-boosting.md) | Ensemble | Trees fixing each other's mistakes (XGBoost) |
| 6 | [svm.md](svm.md) | Classification | Find the widest boundary between classes |
| 7 | [knn.md](knn.md) | Both | Predict based on closest neighbors |
| 8 | [naive-bayes.md](naive-bayes.md) | Classification | Probability-based, great for text |
| 9 | [k-means.md](k-means.md) | Unsupervised | Discover groups in data without labels |
| 10 | [pca.md](pca.md) | Reduction | Reduce features while keeping information |

### Cross-Cutting Topics (read alongside the algorithms)

| # | File | Topic | One-Line Summary |
|---|------|-------|-----------------|
| 11 | [feature-engineering.md](feature-engineering.md) | Data prep | Turn raw data into inputs models can learn from |
| 12 | [model-evaluation.md](model-evaluation.md) | Measuring quality | CV, precision/recall/F1, ROC/AUC, bias-variance |
| 13 | [scikit-learn.md](scikit-learn.md) | The toolkit | Unified `fit`/`predict`/`transform` API + Pipelines |

## ML vs Deep Learning

| Machine Learning | Deep Learning |
|-----------------|---------------|
| You **design features** manually | Model **learns features** automatically |
| Works on **tabular/structured** data | Works on **raw data** (images, text, audio) |
| Needs less data (100s-1000s) | Needs lots of data (10000s+) |
| Trains on CPU (seconds-minutes) | Needs GPU (minutes-hours) |
| Explainable (you can see the rules) | Black box (hard to explain why) |
| scikit-learn | PyTorch / TensorFlow |
| **Still the best for tabular data** | **Best for images, text, audio** |

Don't skip ML for deep learning. **XGBoost still beats neural networks on most tabular datasets.** ML is simpler, faster, and more interpretable. Use deep learning when you need it (images, text, audio), not because it sounds fancier.
