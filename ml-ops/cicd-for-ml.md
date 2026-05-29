# CI/CD for ML (Experiment Tracking & Versioning)

## TL;DR

CI/CD for ML automates the pipeline from "I changed the model" to "it's safely running in production." Experiment tracking (MLflow, W&B) lets you record every training run so you can reproduce any result from 3 months ago. Model versioning ensures you always know exactly which model is running where, and can roll back in seconds. Without these, ML teams end up asking "which version was working last week?" — and nobody knows.

> 💡 **Key Insight:** In regular software, `git diff` tells you exactly what changed. In ML, the "code" is also the model weights, the training data, the hyperparameters, and the random seed. CI/CD for ML is about versioning ALL of these — not just the Python files.

---

## The Mental Model

Think of it like **version control for science experiments**.

In school chemistry class:
- You write down every experiment: materials used, procedure, temperature, results
- So you can reproduce the experiment next week, or next year
- And compare two experiments that changed just one variable

Mapping:
- Lab notebook entry → Experiment run (logged in MLflow/W&B)
- Materials list → Dataset version + model weights + hyperparameters
- Results → Metrics (accuracy, loss, latency, cost)
- Reproducible experiment → A run you can re-execute and get the same result
- Controlled variable change → A/B test between two model versions

The difference: your lab notebook is searchable, comparable, and triggers automatic quality checks before "publishing" (deploying).

---

## Why It Exists

### The Problem

```
Without experiment tracking:

  Monday:    train model_v1 → 85% accuracy → deploy
  Tuesday:   tweak hyperparams → "hmm, this seems worse"
  Wednesday: try different data → 88% accuracy!
  Thursday:  "wait what did we change?"
  Friday:    "can we reproduce Wednesday's result?"
             → nobody knows, notebook was overwritten
  
  3 months later:
  "Production model is degrading. What were the settings when it was working?"
  → ¯\_(ツ)_/¯
```

### The Solution

Log everything. Automatically. Every run.

```
With experiment tracking:

  Monday:    train model_v1 → logged: dataset=v1.2, lr=0.001, accuracy=0.85
  Tuesday:   tweak hyperparams → logged: dataset=v1.2, lr=0.0001, accuracy=0.82
  Wednesday: try different data → logged: dataset=v2.0, lr=0.001, accuracy=0.88
  Thursday:  compare runs in dashboard → "Wednesday's run is the winner"
  Friday:    reproduce Wednesday's run exactly → one click
  
  3 months later:
  "Production model is degrading."
  → check which run was deployed → reproduce it → compare to current → fix
```

### What Changed

Tools like MLflow, Weights & Biases, and DVC turned ML from an ad-hoc exploration into a reproducible engineering discipline.

---

## Core Concepts

### 1. Experiment Tracking

**One-line definition:** Automatically logging the parameters, metrics, code, and artifacts from every training run so you can compare, reproduce, and understand your experiments.

**Analogy:** It's like git log but for model training. Just as `git log` shows every commit with what changed and when, experiment tracking shows every training run with what hyperparameters were used and what metrics resulted.

```
A single experiment "run" logs:

Parameters (input):
  learning_rate: 0.001
  batch_size: 32
  epochs: 10
  model_architecture: "bert-base"
  dataset: "train_v2.csv"

Metrics (output, per epoch):
  epoch 1: train_loss=0.42, val_accuracy=0.78
  epoch 2: train_loss=0.31, val_accuracy=0.83
  epoch 10: train_loss=0.12, val_accuracy=0.91

Artifacts:
  model weights: model_epoch10.pkl
  training curve plot: loss_curve.png
  confusion matrix: confusion_matrix.png

Environment:
  python: 3.11.2
  torch: 2.1.0
  commit: abc1234 (exact git state)
```

**Common misconception:** You have to log manually for each thing you care about. Modern tools like W&B autolog most PyTorch/TensorFlow metrics automatically — you often just add 3 lines of code.

---

### 2. MLflow — Open-Source Experiment Tracking

**One-line definition:** MLflow is an open-source platform for managing the ML lifecycle: tracking experiments, packaging models, and deploying them.

**Analogy:** MLflow is like GitHub for ML experiments — it's self-hosted, free, and stores everything centrally so the whole team can see it.

```python
import mlflow
import mlflow.pytorch

# Start an experiment run
with mlflow.start_run(run_name="bert-lr-0001"):
    
    # Log parameters (the "input" to this experiment)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("model", "bert-base-uncased")
    
    # Train your model
    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader)
        val_acc = evaluate(model, val_loader)
        
        # Log metrics (the "output" of each epoch)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
    
    # Log the final model as an artifact
    mlflow.pytorch.log_model(model, "model")
    
    # Log any other files you want to save
    mlflow.log_artifact("confusion_matrix.png")

# Now in the MLflow UI: compare this run to all others
# See which hyperparameters led to the best val_accuracy
```

**Common misconception:** MLflow is only for training. It also handles model serving (`mlflow models serve`) and has a model registry for versioning deployed models.

---

### 3. Weights & Biases (W&B) — The Industry Standard

**One-line definition:** W&B is a managed experiment tracking platform with richer visualizations, collaboration features, and automatic hyperparameter sweeps.

**Analogy:** If MLflow is a self-hosted spreadsheet for experiments, W&B is Notion — more polished, collaborative, and feature-rich, but hosted in the cloud.

```python
import wandb

# Initialize a run
wandb.init(
    project="sentiment-classifier",
    name="bert-lr-0001",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "architecture": "bert-base-uncased",
    }
)

# Training loop
for epoch in range(config.epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_acc = evaluate(model, val_loader)
    
    # Log metrics — W&B automatically builds live charts
    wandb.log({
        "train/loss": train_loss,
        "val/accuracy": val_acc,
        "epoch": epoch
    })

# Save model as artifact with version
artifact = wandb.Artifact("sentiment-model", type="model")
artifact.add_file("model.pkl")
wandb.log_artifact(artifact)

wandb.finish()
```

**W&B's killer feature — Sweeps (hyperparameter search):**

```python
# Define search space
sweep_config = {
    "method": "bayes",  # Bayesian optimization (smarter than grid search)
    "metric": {"name": "val/accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.01},
        "batch_size": {"values": [16, 32, 64]},
        "dropout": {"min": 0.1, "max": 0.5},
    }
}

# W&B automatically runs 50 trials, picking the best hyperparams
sweep_id = wandb.sweep(sweep_config, project="sentiment-classifier")
wandb.agent(sweep_id, function=train, count=50)
# After 50 runs: "best config: lr=0.003, batch=32, dropout=0.2 → 93.2% accuracy"
```

**Common misconception:** W&B is only for deep learning. It works for any ML framework and even for LLM evaluation — you can log LLM outputs, user ratings, and retrieval quality.

---

### 4. Model Registry & Versioning

**One-line definition:** A model registry is a catalog that tracks every trained model version, its metadata, and its deployment status (staging, production, archived).

**Analogy:** It's like npm but for ML models. `npm install react@18.2.0` pins an exact version. The model registry pins an exact model checkpoint, making deployment reproducible and rollback instant.

```
Model Registry state:

Model: "sentiment-classifier"
──────────────────────────────────────────────────────────────
Version │ Accuracy │ Registered │ Aliases              │ Run ID
──────────────────────────────────────────────────────────────
v1      │ 85.2%    │ 2024-01-15 │ (none — archived)    │ run_abc
v2      │ 87.8%    │ 2024-02-01 │ @production          │ run_def
v3      │ 89.1%    │ —          │ @staging, @challenger│ run_ghi

Current action: Move @production from v2 → v3
Rollback plan:  Move @production back to v2 (one API call, ~instant)
```

```python
# MLflow model registry (MLflow 2.9+ uses aliases, not stages)
import mlflow

# Register a model (after training)
model_uri = f"runs:/{run_id}/model"
model_version = mlflow.register_model(model_uri, "sentiment-classifier")

client = mlflow.MlflowClient()

# Tag as "staging" for testing — aliases are free-form labels
client.set_registered_model_alias(
    name="sentiment-classifier",
    alias="staging",
    version=model_version.version,
)

# After passing evaluation, promote: move the "production" alias to this version
client.set_registered_model_alias(
    name="sentiment-classifier",
    alias="production",
    version=model_version.version,
)

# Load the production model by alias (not by hardcoded version or path)
model = mlflow.pytorch.load_model("models:/sentiment-classifier@production")
```

> ⚠️ **Note:** Older tutorials use `transition_model_version_stage("Staging"|"Production")`. That stages API is deprecated as of MLflow 2.9. Use aliases (`@production`, `@staging`, `@champion`, `@challenger`) — they're more flexible and reflect current best practice.

---

### 5. The ML CI/CD Pipeline

**One-line definition:** An automated pipeline that runs every time you push code, testing data quality, retraining models, evaluating them, and deploying only if they pass all checks.

**Analogy:** It's a GitHub Actions / Jenkins pipeline — but the "build step" is model training and the "tests" are evaluation metrics.

```
Traditional CI/CD:          ML CI/CD:
──────────────────────────────────────────────────────────
Code push                   Code push OR new data
  ↓                           ↓
Run unit tests              Data validation (schema, distribution)
  ↓                           ↓
Build artifact              Train model (or fine-tune)
  ↓                           ↓
Integration tests           Evaluate model (accuracy, safety, bias)
  ↓                           ↓
Deploy if all pass          Compare to current production model
                              ↓
                            Deploy if better AND passes thresholds
                              ↓
                            Monitor for regression, auto-rollback
```

---

## How It Actually Works (Step-by-Step)

### A Complete ML CI/CD Pipeline

```
Step 1: Developer pushes code change (or new training data arrives)
        GitHub Actions / GitLab CI triggers pipeline

Step 2: Data validation
        - Check schema: all expected columns present?
        - Check distribution: data drift from previous version?
        - Check volume: enough samples to train?
        - FAIL if any check fails → block pipeline

Step 3: Model training
        - Launch training job (local, cloud, or GPU cluster)
        - Log ALL parameters and metrics to W&B / MLflow
        - Save model artifact to model registry as "Candidate"

Step 4: Automated evaluation
        - Run evaluation suite on held-out test set
        - Compare candidate vs current production model:
            candidate accuracy: 89.1%
            production accuracy: 87.8%
            delta: +1.3% ← improvement ✅
        - Check safety/bias thresholds: all pass ✅
        - FAIL pipeline if candidate is worse than production

Step 5: Deploy to staging
        - Update staging environment to use new model
        - Run integration tests (API endpoints, response format)
        - Run smoke tests (a few real user queries)

Step 6: Human approval (optional for high-stakes)
        - Slack notification: "New model ready for review"
        - Reviewer checks W&B dashboard, approves

Step 7: Production deployment
        - Blue/green deployment: spin up new version alongside old
        - Gradually shift traffic: 5% → 25% → 50% → 100%
        - Monitor quality metrics during rollout
        - Auto-rollback if quality drops below threshold

Step 8: Register as production version
        - Model registry updated: v3 → Production
        - Old production (v2) → Archived (not deleted, for rollback)
```

---

## Code in Practice

### 1. GitHub Actions ML Pipeline

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Training & Deployment Pipeline

on:
  push:
    branches: [main]
    paths: ['src/model/**', 'data/**', 'configs/**']

jobs:
  validate-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate training data
        run: python scripts/validate_data.py --data data/train.csv

  train-and-evaluate:
    needs: validate-data
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Train model
        id: train
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: python scripts/train.py --config configs/training.yaml
      
      - name: Evaluate against production
        run: |
          python scripts/evaluate.py \
            --candidate ${{ steps.train.outputs.model_path }} \
            --baseline sentiment \
            --min-improvement 0.005  # Must be at least 0.5% better
            # --baseline is a bare model name; evaluate.py builds the
            # full "models:/<name>@production" URI itself.
      
      - name: Register model
        if: success()
        run: python scripts/register_model.py --alias staging

  deploy-to-staging:
    needs: train-and-evaluate
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: ./deploy.sh staging

      - name: Run smoke tests
        run: python scripts/smoke_tests.py --env staging

  deploy-to-production:
    needs: deploy-to-staging
    runs-on: ubuntu-latest
    environment: production  # Requires manual approval in GitHub
    steps:
      - name: Deploy to production (blue/green)
        run: ./deploy.sh production --strategy blue-green
```

### 2. Full W&B Training Script

```python
# train.py — production-ready training with full logging
import wandb
import argparse
from sklearn.metrics import accuracy_score, classification_report

def train(config=None):
    with wandb.init(config=config) as run:
        cfg = wandb.config
        
        # Log the dataset as a versioned artifact
        dataset_artifact = wandb.Artifact("training-data", type="dataset")
        dataset_artifact.add_file("data/train.csv")
        run.log_artifact(dataset_artifact)
        
        # Load data and model
        X_train, y_train, X_val, y_val = load_data("data/train.csv")
        model = build_model(
            learning_rate=cfg.learning_rate,
            dropout=cfg.dropout,
            architecture=cfg.architecture,
        )
        
        # Training loop with metric logging
        for epoch in range(cfg.epochs):
            train_loss = train_epoch(model, X_train, y_train, cfg.batch_size)
            val_preds = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_preds)
            
            # Log metrics — W&B builds charts automatically
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/accuracy": val_accuracy,
            })
            
            # Save checkpoint every 5 epochs as a versioned artifact
            if epoch % 5 == 0:
                ckpt = wandb.Artifact(
                    f"sentiment-checkpoint",
                    type="model",
                    metadata={"epoch": epoch, "val_accuracy": val_accuracy},
                )
                # Persist the model file on disk first, then attach to the artifact
                save_model(model, f"checkpoint-epoch-{epoch}.pkl")
                ckpt.add_file(f"checkpoint-epoch-{epoch}.pkl")
                run.log_artifact(ckpt)
        
        # Final evaluation
        final_report = classification_report(y_val, val_preds, output_dict=True)
        wandb.log({
            "final/accuracy": final_report["accuracy"],
            "final/f1_macro": final_report["macro avg"]["f1-score"],
        })
        
        # Log final model as artifact
        model_artifact = wandb.Artifact(
            "sentiment-model",
            type="model",
            metadata={
                "accuracy": final_report["accuracy"],
                "framework": "sklearn",
                "training_data": "train_v2.csv",
            }
        )
        model_artifact.add_file("model.pkl")
        run.log_artifact(model_artifact)
        
        print(f"Final accuracy: {final_report['accuracy']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    train(config)
```

### 3. Model Comparison Before Deploy

```python
# evaluate.py — compare candidate to production before deploying
import mlflow
import argparse
from evaluation import run_eval_suite

def compare_models(candidate_path: str, production_model_name: str, min_improvement: float):
    """Returns True if candidate is good enough to deploy."""
    
    # Load candidate (just trained)
    candidate = mlflow.pyfunc.load_model(candidate_path)
    
    # Load current production model from registry (alias-based, MLflow 2.9+)
    production = mlflow.pyfunc.load_model(f"models:/{production_model_name}@production")
    
    # Run both on the same test set
    candidate_metrics = run_eval_suite(candidate, TEST_DATA)
    production_metrics = run_eval_suite(production, TEST_DATA)
    
    print(f"Candidate accuracy:   {candidate_metrics['accuracy']:.4f}")
    print(f"Production accuracy:  {production_metrics['accuracy']:.4f}")
    
    # Check improvement
    delta = candidate_metrics['accuracy'] - production_metrics['accuracy']
    print(f"Delta: {delta:+.4f}")
    
    if delta < min_improvement:
        print(f"❌ Not enough improvement (need {min_improvement:+.4f})")
        return False
    
    # Check safety thresholds
    if candidate_metrics['hallucination_rate'] > 0.05:
        print(f"❌ Hallucination rate too high: {candidate_metrics['hallucination_rate']:.2%}")
        return False
    
    print("✅ Candidate passes all checks — ready to deploy")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate")
    parser.add_argument("--baseline")
    parser.add_argument("--min-improvement", type=float, default=0.005)
    args = parser.parse_args()
    
    success = compare_models(args.candidate, args.baseline, args.min_improvement)
    exit(0 if success else 1)  # Exit code 1 fails the CI/CD pipeline
```

---

## Gotchas & Pitfalls

```
❌ Not tracking the dataset version alongside the model
   "We got 91% accuracy!" — "Which data? The one before or after the cleaning?"
✅ Log dataset hash/version in every experiment run

❌ Overwriting the same model file on every training run
   You can never go back to a previous good model
✅ Save each run as a new versioned artifact, never overwrite

❌ Only logging final metrics, not per-epoch curves
   You can't see if the model was overfitting
✅ Log metrics at every epoch / every N steps

❌ Deploying "the latest model" without comparison to production
   New model might be worse than the one you're replacing
✅ Always compare candidate vs current production — block deploy if worse

❌ No rollback plan
   Deploy fails → hours of downtime figuring out what to revert to
✅ Keep last N model versions in registry, test rollback procedure monthly

❌ Using random seeds inconsistently
   "I can't reproduce run #47 from last month"
✅ Set and log the random seed: random.seed(42), torch.manual_seed(42)

❌ Training on the full dataset before splitting
   Data leaks from test into training → inflated accuracy
✅ Always split BEFORE any preprocessing; log train/val/test sizes
```

---

## When to Use / When NOT to Use

### Definitely Set Up Experiment Tracking When:
- More than one person is working on the same model
- You're running more than a few training experiments
- You need to reproduce results or explain why a model was chosen
- Production models need to be updated or retrained regularly

### Definitely Set Up CI/CD for ML When:
- New model versions are deployed more than once a month
- Model quality regressions have real user impact
- You need audit trails for compliance (healthcare, finance, legal)
- The team is larger than 2 people working on the same system

### Can Start Simple When:
- Solo project or learning exercise → just log to a CSV or W&B free tier
- One-time model training → just save the model file and note the params
- Research prototype → notebooks are fine, just don't ship them directly

---

## Related Concepts

| Concept | Connection |
|---------|------------|
| Evaluation & Monitoring | Evaluation is the "test" step in ML CI/CD; monitoring catches regressions post-deploy |
| Model Serving | CI/CD deploys new model versions to the serving infrastructure |
| Fine-tuning | Fine-tuning runs need experiment tracking to compare against the base model |
| RAG | When you update the knowledge base (vector DB), that also triggers evaluation |

---

## Cheat Sheet

```
Experiment tracking (log these every run):
  Parameters:  all hyperparameters, model architecture, dataset version
  Metrics:     loss, accuracy, etc. per step/epoch
  Artifacts:   model weights, plots, config files
  Environment: python/library versions, git commit hash, random seed

Tools:
  MLflow     → open source, self-hosted, great model registry
  W&B        → managed, best visualizations, team collaboration, sweeps
  DVC        → git for data and models (works with any experiment tracker)

ML CI/CD steps:
  1. Validate data        (schema, distribution, volume)
  2. Train model          (log everything to experiment tracker)
  3. Evaluate candidate   (compare to production — block if worse)
  4. Deploy to staging    (test with smoke tests)
  5. Deploy to prod       (blue/green, gradual rollout)
  6. Monitor post-deploy  (auto-rollback if quality drops)

Remember:
  1. Log EVERYTHING: params, metrics, artifacts, dataset version, git commit
  2. Never deploy without comparing to the current production model first
  3. Always have a rollback: keep N previous model versions in the registry
```

---

## Self-Check Questions

<details>
<summary>Click to reveal answers</summary>

**Q1: Why do you need to log the dataset version alongside model metrics?**
The same model code with different training data can produce completely different models. If you only log accuracy, you can't tell if a 5% improvement came from better hyperparameters or better data. Logging both lets you isolate variables.

**Q2: What's the difference between MLflow and Weights & Biases?**
MLflow is open-source and self-hosted — you control the data and there's no subscription cost. W&B is a managed cloud service with richer visualizations, better team collaboration, and automatic hyperparameter sweeps. MLflow is great for solo/enterprise; W&B is great for team/startup.

**Q3: Why should a CI/CD pipeline fail if the new model isn't better than production?**
Because "new" doesn't mean "better." A new model might train on slightly different data, use different hyperparameters, or have a bug — and be worse than the current production model. Blocking deployment protects users from regressions.

**Q4: What's a model registry and why is it better than just saving model files?**
A model registry tracks versions, metadata (accuracy, training date, dataset), and deployment status (staging/production/archived) for all your models. It lets you promote, demote, and roll back models by name+version instead of managing file paths. It's a single source of truth for "what's running in production right now."

**Q5: What's the point of blue/green deployment for ML models?**
Blue/green runs two versions simultaneously and gradually shifts traffic from old to new. If the new model degrades quality in production, you can instantly shift 100% of traffic back to the old version with zero downtime. It's much safer than replacing the old model in-place.

</details>

---

## Go Deeper

| Resource | Why It's Worth Your Time |
|----------|--------------------------|
| [W&B Quickstart](https://docs.wandb.ai/quickstart) | Get experiment tracking running in 5 minutes. The visualizations will immediately show you why this is worth doing. |
| [MLflow Getting Started](https://mlflow.org/docs/latest/getting-started/index.html) | Best if you want self-hosted. The tracking tutorial is excellent and takes ~45 minutes. |
| [Made With ML — MLOps Course](https://madewithml.com) | Goku Mohandas's free course is the single best end-to-end MLOps resource. Highly practical, no fluff. |
| *Designing Machine Learning Systems* by Chip Huyen | Chapters on deployment pipelines and monitoring are the industry standard reference. |
| [DVC Documentation](https://dvc.org/doc) | If you care about versioning training data (you should), DVC integrates with git to version datasets like code. |
