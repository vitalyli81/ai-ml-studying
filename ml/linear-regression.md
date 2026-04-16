# Linear Regression

## What Is It?

Linear regression finds a **straight line** (or flat plane) that best fits your data. It predicts a **number** (not a category).

Think of it like drawing the best-fit line through a scatter plot.

## Real-World Examples

- Predict **house price** based on square footage
- Predict **salary** based on years of experience
- Predict **temperature** based on time of year

## How It Works (Step by Step)

### 1. The Equation

```
y = mx + b
```

- **y** — the thing you're predicting (house price)
- **x** — the input feature (square footage)
- **m** — slope (how much y changes when x changes by 1)
- **b** — intercept (the starting value when x = 0)

With multiple features it becomes:

```
y = w1*x1 + w2*x2 + ... + b
```

Each `w` is a **weight** — how important that feature is.

### 2. Finding the Best Line

The algorithm tries many lines and picks the one with the **smallest total error**.

Error is measured by **Mean Squared Error (MSE)**:

```
MSE = average of (actual - predicted)²
```

Why squared? So negative and positive errors don't cancel out, and big errors get penalized more.

### 3. Gradient Descent (How It Learns)

Imagine you're blindfolded on a hill and need to find the lowest point:

1. Feel which direction goes downhill (compute the **gradient**)
2. Take a small step in that direction
3. Repeat until you reach the bottom

The "step size" is called the **learning rate**:
- Too big → you overshoot and bounce around
- Too small → takes forever to converge
- Just right → smooth convergence to the minimum

## When to Use It

| Good For | Bad For |
|----------|---------|
| Continuous predictions (prices, scores) | Yes/No decisions (use logistic regression) |
| Linear relationships between variables | Complex, curvy patterns |
| Understanding which features matter most | When features interact in complex ways |

## Key Assumptions

1. **Linear relationship** — the relationship between x and y is roughly a straight line
2. **Independence** — data points don't influence each other
3. **No multicollinearity** — input features aren't highly correlated with each other

## Common Pitfalls

- **Outliers** destroy the line — one extreme point pulls the whole fit
- **Overfitting with too many features** — the model memorizes noise instead of patterns
- **Ignoring non-linear patterns** — if the data curves, a straight line won't capture it

## Python Example

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Data: square footage → price
X = np.array([[600], [800], [1000], [1200], [1500]])
y = np.array([150000, 200000, 250000, 280000, 350000])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict price for a 1100 sq ft house
prediction = model.predict([[1100]])
print(f"Predicted price: ${prediction[0]:,.0f}")

# See what the model learned
print(f"Slope (per sq ft): ${model.coef_[0]:,.0f}")
print(f"Intercept: ${model.intercept_:,.0f}")
```

## Regularization (Preventing Overfitting)

When you have many features, add a penalty to keep weights small:

- **Ridge (L2)** — shrinks all weights evenly. Use when all features might matter.
- **Lasso (L1)** — pushes some weights to exactly 0. Use when you suspect only a few features matter (acts as feature selection).

## Key Takeaway

Linear regression is the **simplest predictive model**. Start here, understand the results, then move to complex models only if needed. If linear regression works well enough, there's no reason to use something fancier.
