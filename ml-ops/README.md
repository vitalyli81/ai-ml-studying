# Phase 6: MLOps & Production

> Taking AI models from your laptop to the real world — reliably, repeatedly, at scale.

MLOps (Machine Learning Operations) is the discipline of **shipping and maintaining AI systems in production**. You already know how to build models — MLOps is everything that makes them actually work for real users over time.

## Topics

| # | Topic | File |
|---|-------|------|
| 1 | Model Serving | [model-serving.md](model-serving.md) |
| 2 | Vector Databases | [vector-databases.md](vector-databases.md) |
| 3 | Evaluation & Monitoring | [evaluation-monitoring.md](evaluation-monitoring.md) |
| 4 | CI/CD for ML | [cicd-for-ml.md](cicd-for-ml.md) |

## Learning Path

```
Model Serving → Vector Databases → Evaluation & Monitoring → CI/CD for ML
  (get model      (give it           (know if it's            (automate
   online)         knowledge)         working)                 everything)
```

## The MLOps Mindset

```
Research ML:                Production ML:
────────────────────────────────────────────────────────
"Does the model work?"       "Does it work for users?"
Run once on a notebook       Runs 24/7 reliably
1 dataset, 1 result          Continuous data, drift
You validate manually        Automated evaluation pipelines
No one else uses it          10,000 requests per day
```

## What You'll Be Able to Do After This Phase

- Deploy any ML model as a REST API with Docker
- Build a production RAG system backed by a real vector DB
- Monitor LLM outputs for quality, cost, and safety regressions
- Set up automated pipelines that retrain, evaluate, and redeploy models
- Track experiments so you can reproduce any result from 3 months ago
