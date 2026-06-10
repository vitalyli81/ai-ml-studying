# Linear Regression

## TL;DR

Linear regression draws the best-fit straight line through your data to predict a **number**. You give it inputs (square footage), it predicts an output (house price). It learns by adjusting the line's slope until its predictions are as close as possible to the real values. It's the simplest and most interpretable ML model — always try it first.

> 💡 **Key Insight:** If your output is a number and your data roughly follows a line, linear regression is often all you need. Fancy models don't always beat a simple, well-fitted line.

---

## The Mental Model

Think of a **salary negotiation based on experience**.

Every year of experience, your expected salary goes up by roughly the same amount. Plot salary vs years of experience on a graph, draw the best line through the dots — that's linear regression.

Mapping:
- Years of experience → input feature (x)
- Expected salary → prediction (y)
- Average salary raise per year → the slope (m)
- Entry-level salary → the intercept (b)
- "Best line" → the learned model

The algorithm's only job: find the slope and intercept that minimize prediction errors across all data points.

---

## Build the Intuition From Zero

The one thing that feels like magic here is **"the model learns the line by itself."** How? Let's watch a line physically tighten onto two data points, by hand, so "learning" stops being a black box.

Say we have just two houses and we're learning `price = w × sqft` (ignore the intercept for a moment). Truth: a 1000 sqft house costs $200k, so the perfect `w` is 200 (price in $k). We don't know that yet — we **start with a random guess** and let the errors push us toward it:

```
Start:  w = 100   ("I think it's $100/sqft")
        predict 1000 sqft → 100×1000... in $k that's a guess of $100k
        actual is $200k → we're $100k too LOW
        → the error says "w is too small, push it UP"

Step 1: nudge w up a little        → w = 140   (prediction now $140k, error $60k, smaller)
Step 2: still too low, nudge up     → w = 175   (error $25k)
Step 3: nudge up                    → w = 192   (error $8k)
Step 4: nudge up                    → w = 198   (error $2k)
...the nudges shrink as the error shrinks, until...
        w ≈ 200   (error ≈ 0)  ← the line has "learned" the data
```

That's the entire learning process. Three things to notice — they're the whole rest of the doc:

- **The sign of the error tells you which way to move `w`.** Too low → increase; too high → decrease. (That's the *gradient*.)
- **The size of each nudge** is the **learning rate** — too big and you overshoot past 200 and bounce; too small and it takes forever.
- **What counts as "the error"** — we square it so big misses scream louder than small ones. (That's **MSE**, the loss.)

> 💡 **"Training a model" = this loop, repeated.** Start with a random line, measure how wrong it is, nudge the weights in the direction that reduces the error, repeat until it stops improving. Every model in this folder — and every neural network — is a fancier version of this exact loop.

The concepts below (MSE, gradient descent, learning rate) just name the three pieces you just watched.

---

## Why It Exists

### The Problem Before

Humans guessed relationships manually: "house prices go up about $150/sqft." But guesses are biased, inconsistent, and can't handle multiple inputs at once. And when you have 20 features (bedrooms, location, age, floor, ...) — manual rules fall apart.

### The Solution

Let the algorithm find the weights automatically by minimizing prediction error over all training examples. No guessing. No bias. Generalizes to any number of features.

### What Changed

Linear regression (formalized in the 19th century) became the foundation for all of supervised learning. Every neural network is essentially stacked, non-linear versions of linear regression.

---

## Core Concepts

### 1. The Line Equation

**One-line definition:** The formula that maps inputs to a predicted number.

**Analogy:** `y = mx + b` from 9th grade math — that's literally it for one feature.

**Technical explanation:** For one feature: `y = w×x + b`. For multiple features: `y = w₁x₁ + w₂x₂ + ... + b`. Each `w` (weight) says "how much does this feature affect the output?" `b` (bias) is the baseline prediction when all inputs are zero.

```python
# Manually: 1000 sqft house, learned weights w=150, b=50000
price = 150 * 1000 + 50000  # = $200,000

# With multiple features:
# price = w_sqft×sqft + w_rooms×rooms + w_age×age + b
```

**Common misconception:** People think "linear" means the data must look like a line. It means the *equation* is linear in the weights — you can still use `x²` as a feature and it's still "linear regression."

---

### 2. Mean Squared Error (MSE) — The Loss Function

**One-line definition:** The score that tells the model how wrong it is — the average of squared differences between predicted and actual values.

**Analogy:** Like a golf handicap — it measures how far off your shots are on average. Square the distances so big misses hurt much more than small ones.

```
Actual prices:     [200K, 300K, 250K]
Predicted prices:  [190K, 320K, 240K]
Errors:            [-10K,  20K, -10K]
Squared errors:    [100M, 400M, 100M]
MSE:               (100M + 400M + 100M) / 3 = 200M

The model will try to minimize this number.
```

**Common misconception:** Why square the errors instead of taking the absolute value? Squaring penalizes large errors disproportionately (a miss of 20K hurts 4× more than a miss of 10K), which pushes the model toward avoiding big mistakes.

---

### 3. Gradient Descent — How the Model Learns

**One-line definition:** An iterative algorithm that adjusts weights by nudging them in the direction that reduces the error.

**Analogy:** You're blindfolded on a hilly landscape and want to reach the lowest valley. You feel which direction goes downhill and take a small step that way. Repeat until you stop moving. That's gradient descent.

```
Error landscape (U-shaped / convex):

Error ▲
      |*       *
      | *     *
      |  *   *
      |    *        ← minimum error
      └─────────── weights

The algorithm rolls the ball down to the bottom of the U by adjusting weights.
```

**Technical explanation:** The gradient (derivative) of the loss tells us which direction error increases. We move the weights in the *opposite* direction (downhill) by a small amount called the **learning rate**.

```python
# One gradient descent step:
weight = weight - learning_rate * gradient
# If gradient is positive (error increases with weight) → decrease weight
# If gradient is negative (error decreases as weight increases) → increase weight
```

**Common misconception:** Gradient descent always finds the global minimum. For linear regression it does (the loss is convex — a single U-shaped valley with one bottom). For neural networks, it only finds a local minimum, which is often good enough.

---

### 4. Learning Rate

**One-line definition:** The step size for each gradient descent update — how boldly you move downhill.

**Analogy:** Finding your way in the dark with a flashlight. Too big a step = you overshoot the path. Too small = you take forever and might get stuck. Just right = steady progress.

```
Too high (0.1+): weight bounces, never converges
         *       *
           *   *
             * ← bouncing around the minimum

Too low (0.00001): converges but takes 10,000 steps
──────────────────────────────────> (very slow descent)

Just right (0.001-0.01): smooth convergence
       *
         *
           *
             * ← reaches minimum cleanly
```

**Common misconception:** Bigger learning rate = faster training = better. A large learning rate often causes the model to diverge (get worse over time, not better). Start with 0.01 and tune from there.

---

### 5. Regularization — Preventing Overfitting

**One-line definition:** Adding a penalty to the loss function that discourages the model from learning weights that are too large.

**Analogy:** It's like adding a rule to your essay grading: "lose points for using unnecessarily complex words." The model is forced to be simpler and more general.

```
Ridge (L2): penalty = λ × sum(weights²)    → shrinks all weights
Lasso (L1): penalty = λ × sum(|weights|)   → pushes some weights to exactly 0

Without regularization: w = [0.8, 150.3, -200.1, 0.0001, ...]  (some huge)
With Ridge:             w = [0.6, 100.2, -130.8, 0.0001, ...]  (all smaller)
With Lasso:             w = [0.4, 120.1, 0, 0, ...]             (some zeroed out)

Lasso effectively does feature selection — it eliminates unimportant features.
```

**Common misconception:** Regularization always hurts accuracy. It hurts training accuracy slightly but improves test/real-world accuracy by preventing overfitting.

---

## How It Actually Works (Step-by-Step)

Let's learn `price = w × sqft + b` from scratch, doing one full gradient
descent step by hand with real numbers, then jumping to the converged model.

```
Dataset (n = 5):
  Sqft (x) | Price (y)
  ─────────────────────
   600     | 150,000
   800     | 200,000
  1000     | 250,000
  1200     | 280,000
  1500     | 350,000

Step 1 — Initialize the weights
  w = 0,  b = 0          ← a flat line at zero; deliberately terrible

Step 2 — Predict with the current line
  ŷ = w·x + b = [0, 0, 0, 0, 0]

Step 3 — Score it with MSE (how wrong are we?)
  error = y − ŷ = [150K, 200K, 250K, 280K, 350K]
  MSE = mean(error²)
      = (150K² + 200K² + 250K² + 280K² + 350K²) / 5
      ≈ 65.2 billion        ← huge, as expected for a zero line

Step 4 — Compute the gradient (which way is downhill?)
  ∂MSE/∂w = −2 · mean(error · x) = −2 · 272,200,000 ≈ −544,400,000
  ∂MSE/∂b = −2 · mean(error)     = −2 · 246,000     ≈   −492,000
  Both gradients are negative → MSE drops if we INCREASE w and b.

Step 5 — Take one step downhill (subtract lr · gradient)
  The gradients are enormous, so the learning rate must be tiny (1e-8):
  w = 0 − 1e-8 · (−544,400,000) ≈ 5.44
  b = 0 − 1e-8 · (−492,000)     ≈ 0.0049

Step 6 — Did the loss actually go down? (re-score with the new line)
  ŷ = 5.44·x + 0.0049
  MSE ≈ 62.3 billion   ← down from 65.2B after a single step. It works.

Step 7 — Repeat steps 2–6 thousands of times
  Each pass nudges w and b further downhill until the MSE stops shrinking.
  (Raw-feature gradient descent like this is slow because sqft is in the
  thousands while the intercept isn't — their gradients live on wildly
  different scales. Standardizing the feature first makes it converge in
  far fewer steps.)

Step 8 — The converged model
  Solving exactly (the closed form) gives:
    price ≈ 218.0 × sqft + 23,607      (R² ≈ 0.995 — a near-perfect fit)
  Predict a 1100 sqft house:
    price ≈ 218.0 × 1100 + 23,607 ≈ $263,400

  (scikit-learn's LinearRegression jumps straight to this closed-form
  solution — no iterative gradient descent needed for plain MSE. Gradient
  descent earns its keep when the dataset is too big to solve directly, or
  when there's no closed form, like in neural networks.)
```

---

## Code in Practice

### 1. Hello World — Single Feature

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Sqft → price data
X = np.array([[600], [800], [1000], [1200], [1500]])
y = np.array([150000, 200000, 250000, 280000, 350000])

model = LinearRegression()
model.fit(X, y)

# Predict a new house
print(f"Predicted: ${model.predict([[1100]])[0]:,.0f}")
print(f"Slope ($/sqft): {model.coef_[0]:.1f}")
print(f"Intercept: ${model.intercept_:,.0f}")
```

### 2. Practical — Multiple Features + Train/Test Split

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Multiple features: [sqft, bedrooms, age]
X = np.array([
    [600, 1, 20], [800, 2, 15], [1000, 2, 10],
    [1200, 3, 5],  [1500, 4, 2], [900, 2, 12],
])
y = np.array([150000, 200000, 250000, 285000, 360000, 220000])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R² score: {r2_score(y_test, y_pred):.3f}")    # 1.0 = perfect, 0 = useless
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")

for name, weight in zip(["sqft", "bedrooms", "age"], model.coef_):
    print(f"  {name}: {weight:+.1f}")
```

### 3. Real-World Pattern — With Regularization

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Always scale features before regularization — weights need to be comparable
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))   # alpha = regularization strength
])

pipeline.fit(X_train, y_train)
print(f"Ridge R²: {pipeline.score(X_test, y_test):.3f}")

# Lasso for feature selection
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso(alpha=0.1))
])
lasso_pipeline.fit(X_train, y_train)
# Some weights will be exactly 0 — those features were dropped
print(f"Lasso weights: {lasso_pipeline['model'].coef_}")
```

---

## Gotchas & Pitfalls

```
❌ Not splitting into train/test sets
   You measure accuracy on the same data the model learned from — inflated scores
✅ Always: train_test_split() before fitting, evaluate only on test set

❌ Forgetting to scale features for regularized regression
   Ridge/Lasso penalize large weights, but a weight's size depends on feature scale
✅ Use StandardScaler() in a Pipeline before Ridge/Lasso

❌ Using linear regression for categorical outputs (yes/no, cat/dog)
   It can predict values like 1.7 or -0.3, which aren't valid categories
✅ Use logistic regression for classification tasks

❌ Ignoring outliers
   One house that sold for $5M pulls the whole regression line toward it
✅ Check for outliers with box plots; consider robust regression (HuberRegressor)

❌ Assuming the relationship is linear when it isn't
   If data is curved, a straight line will always underfit
✅ Plot your data first; consider polynomial features or a different algorithm

❌ Using R² as the only metric
   R² = 0.95 sounds great but could hide large individual prediction errors
✅ Also report RMSE — it's in the same units as your target variable
```

---

## When to Use / When NOT to Use

### Use Linear Regression When:
- Your target is a continuous number (price, temperature, score)
- You need an interpretable model — you can read off exactly which features matter
- You want a fast baseline before trying complex models
- The relationship between features and target is approximately linear

### Don't Use Linear Regression When:
- Your output is a category (use logistic regression or tree-based models)
- The data has strong non-linear patterns (use polynomial features or decision trees)
- You need to handle complex feature interactions automatically (use tree-based models)
- You have text or image data (use neural networks)

---

## Related Concepts

| Concept | Connection |
|---------|------------|
| Logistic Regression | Same idea, but with a sigmoid on top to predict probabilities instead of numbers |
| Gradient Descent | The learning algorithm that trains linear regression — also used in neural networks |
| Regularization (Ridge/Lasso) | Built-in overfitting prevention — critical for high-dimensional data |
| Feature Engineering | Adding `sqft²` as a new feature lets linear regression fit curves |
| Neural Networks | Stack many linear regression layers with non-linear activations between them |

---

## Cheat Sheet

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Basic
model = LinearRegression()
model.fit(X_train, y_train)          # learn weights
model.predict(X_new)                  # make predictions
model.coef_                           # learned weights (feature importance)
model.intercept_                      # bias term

# Regularization (scale features first!)
Ridge(alpha=1.0)  # L2 — shrinks all weights, no features eliminated
Lasso(alpha=0.1)  # L1 — pushes unimportant weights to exactly 0

# Metrics
from sklearn.metrics import r2_score, mean_squared_error
r2_score(y_test, y_pred)              # 1.0 = perfect, 0 = baseline
np.sqrt(mean_squared_error(y, pred))  # RMSE — in same units as target

Key formula:  y = w₁x₁ + w₂x₂ + ... + b
Loss:         MSE = mean((actual - predicted)²)
Learning:     Gradient descent minimizes MSE

Remember:
  1. Scale features when using Ridge/Lasso
  2. Always split train/test before fitting
  3. R² measures explained variance; RMSE measures average error
```

---

## Self-Check Questions

<details>
<summary>Click to reveal answers</summary>

**Q1: What does the slope (weight) in linear regression actually represent?**
It represents how much the prediction changes when that feature increases by 1 unit. A weight of 150 for square footage means: "every extra square foot adds $150 to the predicted price."

**Q2: Why do we square the errors in MSE instead of just taking their absolute value?**
Squaring makes larger errors much more costly (20K error = 4× the penalty of 10K error), pushing the model to avoid big mistakes. It's also mathematically easier to differentiate — gradient descent works smoothly with squared errors.

**Q3: What's the difference between Ridge and Lasso regularization?**
Ridge (L2) shrinks all weights toward zero but never eliminates them. Lasso (L1) pushes some weights to exactly zero, effectively removing those features. Use Lasso when you suspect only a few features truly matter.

**Q4: Can linear regression handle non-linear data?**
Yes, indirectly. You can add polynomial features (like x²) to capture curves — this is called polynomial regression. The model is still "linear regression" because it's linear in the weights, even if the feature is non-linear.

**Q5: Why do you evaluate on the test set instead of the training set?**
The training set was used to learn the weights — the model has "seen" it and can memorize it. The test set is data the model has never seen, so it measures how well the model generalizes to new data. Training accuracy is always optimistic; test accuracy is realistic.

</details>

---

## Go Deeper

| Resource | Why It's Worth Your Time |
|----------|--------------------------|
| [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo) | Josh Starmer explains every concept visually. The single best 20-minute intro to linear regression. |
| [Scikit-learn Linear Models docs](https://scikit-learn.org/stable/modules/linear_model.html) | Official docs with all variants (Ridge, Lasso, ElasticNet). Essential reference. |
| [3Blue1Brown: Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) | The most beautiful visual explanation of how gradient descent works. |
| *Hands-On Machine Learning* Ch. 4 — Aurélien Géron | The best book chapter on linear models. Goes from basics to regularization with clear examples. |
| [Kaggle: House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) | Practice applying linear regression on real data. Great for building intuition. |
