# 🎓 AI Engineer Mentorship — Session 9
## MLOps: Experiment Tracking, Model Versioning, Reproducible Training & Model Registry

**Level:** Intermediate → Advanced AI Engineering  
**Duration:** 60 minutes  
**Today's project:** Build a reproducible ML training pipeline using Python, Scikit-learn, and MLflow.

In Session 8, we learned how to fine-tune LLMs using LoRA and QLoRA. We also built a small backend error-classification training example.

Today, we'll focus on an equally important question:

**How do you manage, reproduce, compare, and deploy machine learning models reliably in production?**

Imagine you fine-tune an LLM and achieve 92% classification accuracy.

Two weeks later, you train another version and achieve 95%.

Your manager asks:

> What exactly changed between these two models? Which dataset did you use? Can we reproduce the old model? And can we safely roll back if the new model performs poorly?

Answering these questions is part of **Machine Learning Operations (MLOps)**.

---

# 1. Learning Objectives

By the end of today's session, you should understand:

1. What MLOps is and how it differs from DevOps.
2. Why ML experiments must be reproducible.
3. How to track parameters, metrics, datasets, and artifacts.
4. How to use MLflow for experiment tracking.
5. How to compare multiple model versions.
6. What a Model Registry is and how model promotion works.
7. How to design automated training and deployment pipelines.
8. How to apply these principles to LLM fine-tuning and production AI systems.

## ⏱️ Today's 60-Minute Agenda

| Time | Topic |
|---|---|
| 0–10 min | MLOps fundamentals and reproducibility |
| 10–20 min | Experiments, metrics, artifacts, and dataset versioning |
| 20–40 min | Build a Python MLflow training pipeline |
| 40–47 min | Model Registry, versioning, and rollback |
| 47–53 min | Production MLOps architecture and CI/CD |
| 53–57 min | Practical assignment |
| 57–60 min | Interview questions and knowledge quiz |

---

# Part 1: What Is MLOps?

**Time: 0–10 minutes**

You already understand backend deployment.

Consider a normal FastAPI application.

```text
Developer
    |
    v
Write Python Code
    |
    v
Git Commit
    |
    v
Unit Tests
    |
    v
Docker Build
    |
    v
CI/CD Pipeline
    |
    v
Production Server
```

This is a traditional software delivery workflow.

Now consider an AI application.

```text
Training Data
     |
     v
Data Validation
     |
     v
Feature Preparation
     |
     v
Model Training
     |
     v
Model Evaluation
     |
     v
Model Registry
     |
     v
Deployment
     |
     v
Monitoring
     |
     v
Retraining
```

The difference is that AI systems depend on more than application code.

Their behavior also depends on training data, model weights, hyperparameters, preprocessing logic, and training environments.

## DevOps vs MLOps

| Aspect | DevOps | MLOps |
|---|---|---|
| Primary artifact | Application code and binaries | Model and preprocessing artifacts |
| Versioning | Source code | Code, data, models, configurations |
| Testing | Unit, integration, performance | Plus data quality and model evaluation |
| Deployment | Application release | Application and model release |
| Monitoring | Errors, CPU, latency | Plus drift, prediction quality, model behavior |
| Rollback | Previous application version | Compatible model and application versions |
| Automation | Build and deployment | Data preparation, training, evaluation, deployment |

MLOps extends established software engineering practices to machine learning.

It does not replace DevOps.

### Why does this matter?

Suppose you trained a model with:

```python
learning_rate = 0.0002
batch_size = 4
epochs = 3
```

Later, you change:

```python
learning_rate = 0.00005
batch_size = 8
epochs = 5
```

Your model's performance changes.

But you didn't record which configuration produced which result.

Now you cannot reliably explain the difference.

This is one of the problems experiment tracking solves.

---

# Part 2: Understanding ML Experiments

**Time: 10–20 minutes**

Let's learn five essential MLOps concepts.

## 2.1 Experiment

An experiment is a collection of related training runs.

For example:

```text
Experiment:
Backend Error Classification

    |
    ├── Run 1: Baseline classifier
    |
    ├── Run 2: TF-IDF + Logistic Regression
    |
    ├── Run 3: Fine-tuned small LLM
    |
    └── Run 4: LLM with LoRA rank 16
```

All these runs attempt to solve the same general problem.

But each uses a different model or configuration.

## 2.2 Run

A run represents an individual execution of a training or evaluation process.

It records information such as:

```json
{
  "run_id": "experiment-run-001",
  "model": "logistic_regression",
  "learning_configuration": {
    "max_iter": 1000
  },
  "validation_macro_f1": 0.87,
  "dataset_version": "v1"
}
```

The values are illustrative.

A run should allow you to answer:

**What was executed, with which configuration, on which data, and what happened?**

## 2.3 Parameters vs metrics

This distinction is important.

**Parameters** are configuration values you choose.

Examples:

```python
learning_rate = 0.0002
epochs = 3
batch_size = 4
lora_rank = 8
```

**Metrics** are measurements produced by training or evaluation.

Examples:

```python
training_loss = 0.35
validation_accuracy = 0.92
macro_f1 = 0.89
inference_latency_ms = 125
```

The basic relationship is:

\[
\text{Model Quality}
=
f(\text{Data},\text{Model},\text{Parameters},\text{Training Process})
\]

Model quality is not determined by hyperparameters alone.

The dataset and training procedure are equally important.

## 2.4 Artifacts

An artifact is a file or collection of files generated during an experiment.

Examples include model weights, tokenizer configurations, evaluation reports, confusion matrices, and training checkpoints.

For our previous LoRA example, artifacts might include:

```text
backend-error-model/
    adapter_config.json
    adapter_model.safetensors
    tokenizer_config.json
    training_config.json
    evaluation_report.json
```

A LoRA adapter normally depends on a compatible base model.

Therefore, the base model identifier and revision should also be recorded.

## 2.5 Dataset versioning

Imagine your original dataset contains 10,000 examples.

Later, you add another 5,000.

You retrain your model and see different results.

Was the improvement caused by the new data, a changed training configuration, or both?

Without dataset versioning, answering this becomes difficult.

A reproducible training run should identify the exact data snapshot used.

One approach is to calculate a dataset hash:

```python
import hashlib
from pathlib import Path

dataset_bytes = Path(
    "training_data.jsonl"
).read_bytes()

dataset_hash = hashlib.sha256(
    dataset_bytes
).hexdigest()

print(dataset_hash)
```

If the file contents change, the hash will almost certainly change.

However, a hash is only an identifier. You must also retain an immutable copy of the dataset to reproduce the training run.

### Senior Engineer insight

Reproducibility requires more than setting a random seed.

It involves recording code revisions, dataset snapshots, dependencies, model revisions, hardware assumptions, and training configurations.

Even with all these recorded, exact bit-for-bit reproducibility may be difficult across different GPU hardware or nondeterministic operations.

---

# Part 3: Introduction to MLflow

**Time: 20–25 minutes**

Today we'll use MLflow.

MLflow provides capabilities for experiment tracking, model management, evaluation, and deployment workflows.

Its tracking system supports recording parameters, metrics, datasets, and artifacts for individual runs. [MLflow AI Platform](https://mlflow.org/docs/latest/tracking/?utm_source=chatgpt.com)

Official reference: [MLflow Tracking Documentation](https://mlflow.org/docs/latest/ml/tracking/?utm_source=chatgpt.com)

## Install dependencies

Create a new project.

```bash
mkdir ai-mlops
cd ai-mlops

python -m venv .venv
```

Activate the virtual environment.

On Linux or WSL:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install:

```bash
pip install -U mlflow scikit-learn pandas
```

Our project will contain:

```text
ai-mlops/
    train.py
    mlflow.db
    mlartifacts/
```

The database and artifact directories will be created as needed.

For this lesson, we'll use a small conventional ML model rather than training an LLM.

**Why?**

Because the experiment-tracking principles are the same, but a small classifier lets you understand the complete lifecycle without waiting for GPU training.

Once you understand the pipeline, we'll apply it to LoRA fine-tuning.

---

# Part 4: Build a Reproducible Training Pipeline

**Time: 25–40 minutes**

## Project: Backend Error Classifier

Our model will classify backend error messages into four categories:

- Database
- Authentication
- Cache
- Background worker

We'll train two models, compare their validation results, evaluate the selected model on a held-out test set, and register the selected artifact.

## Step 1: Prepare the dataset

Create `train.py`.

```python
import hashlib
import json

import pandas as pd

from sklearn.model_selection import train_test_split


examples = {
    "database": [
        "PostgreSQL too many connections",
        "Database connection refused",
        "SQL query execution timeout",
        "Database deadlock detected",
        "Postgres connection pool exhausted",
        "Database authentication connection failed",
        "SQL transaction rollback error",
        "Database server unavailable",
    ],
    "authentication": [
        "JWT token expired",
        "Invalid access token",
        "User login authentication failed",
        "Incorrect password supplied",
        "Authentication credentials missing",
        "Session token invalid",
        "Unable to verify user identity",
        "JWT signature verification failed",
    ],
    "cache": [
        "Redis MISCONF snapshot failure",
        "Redis memory limit exceeded",
        "Cache server connection refused",
        "Redis key eviction warning",
        "Cache persistence failed",
        "Redis timeout during GET operation",
        "Cache backend unavailable",
        "Redis RDB save operation failed",
    ],
    "worker": [
        "Celery worker lost",
        "Background task execution failed",
        "Celery task exceeded time limit",
        "Worker process terminated unexpectedly",
        "Task queue consumer disconnected",
        "Background worker unavailable",
        "Celery worker heartbeat missed",
        "Task execution retry limit exceeded",
    ],
}

rows = [
    {"text": text, "label": label}
    for label, messages in examples.items()
    for text in messages
]

df = pd.DataFrame(rows)

dataset_hash = hashlib.sha256(
    json.dumps(
        rows,
        sort_keys=True
    ).encode("utf-8")
).hexdigest()

train_val_df, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=1 / 3,
    random_state=42,
    stratify=train_val_df["label"]
)

print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))
```

Expected:

```text
Training: 16
Validation: 8
Testing: 8
```

### Why three datasets?

**Training set:** Used to fit model parameters.

**Validation set:** Used to compare model configurations and select a candidate.

**Test set:** Used for final evaluation after model selection.

The test set should not repeatedly influence model development.

Otherwise, you risk indirectly overfitting to the test data.

Our dataset is deliberately tiny and synthetic. It demonstrates the pipeline but cannot establish production-grade classification quality.

---

## Step 2: Define candidate models

Add:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


models = {
    "baseline": Pipeline([
        (
            "tfidf",
            TfidfVectorizer()
        ),
        (
            "classifier",
            DummyClassifier(
                strategy="most_frequent"
            )
        )
    ]),

    "logistic_regression": Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])
}
```

Why are we using a baseline?

Because we need to know whether the more sophisticated model actually improves performance.

If your LLM fine-tuning experiment cannot outperform a simple baseline, you should investigate whether the added complexity is justified.

For this dataset, a majority-class baseline is intentionally weak.

In production, compare against stronger alternatives as well.

---

## Step 3: Configure MLflow tracking

Add:

```python
import mlflow
import mlflow.sklearn
import mlflow.data

from sklearn.metrics import (
    accuracy_score,
    f1_score
)


mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "backend-error-classification"
)
```

This tells MLflow to store experiment metadata in a local SQLite database.

MLflow also supports database-backed tracking with PostgreSQL and other databases. [MLflow AI Platform](https://www.mlflow.org/docs/latest/ml/tracking/tutorials/local-database/?utm_source=chatgpt.com)

For our local experiment, SQLite is sufficient.

## Step 4: Log training runs

Add:

```python
results = []

for model_name, model in models.items():

    with mlflow.start_run(
        run_name=model_name
    ) as run:

        # Train
        model.fit(
            train_df["text"],
            train_df["label"]
        )

        # Validate
        predictions = model.predict(
            val_df["text"]
        )

        accuracy = accuracy_score(
            val_df["label"],
            predictions
        )

        macro_f1 = f1_score(
            val_df["label"],
            predictions,
            average="macro",
            zero_division=0
        )

        # Log configuration
        mlflow.log_param(
            "model_type",
            model_name
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "dataset_sha256",
            dataset_hash
        )

        # Log metrics
        mlflow.log_metric(
            "validation_accuracy",
            accuracy
        )

        mlflow.log_metric(
            "validation_macro_f1",
            macro_f1
        )

        # Log dataset lineage
        for name, data, context in [
            ("train", train_df, "training"),
            ("validation", val_df, "validation"),
        ]:
            dataset = mlflow.data.from_pandas(
                data,
                name=name,
                source="synthetic-backend-errors-v1",
                targets="label"
            )

            mlflow.log_input(
                dataset,
                context=context
            )

        # Save model artifact
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="classifier"
        )

        results.append({
            "name": model_name,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "validation_macro_f1": macro_f1,
            "model": model
        })

        print(
            model_name,
            "Validation F1:",
            round(macro_f1, 3)
        )
```

Let's understand the important operations.

### `mlflow.start_run()`

Creates a new tracked execution.

### `mlflow.log_param()`

Records a configuration parameter.

### `mlflow.log_metric()`

Records a numerical measurement.

### `mlflow.log_input()`

Records dataset metadata and lineage.

### `mlflow.sklearn.log_model()`

Stores the trained model as an MLflow model artifact.

In MLflow 3, logged models receive model identifiers and can be referenced through their returned model URIs. [MLflow AI Platform](https://mlflow.org/docs/latest/tracking/?utm_source=chatgpt.com)

---

## Step 5: Select the best candidate

Add:

```python
winner = max(
    results,
    key=lambda result: (
        result["validation_macro_f1"],
        result["name"] == "baseline"
    )
)

print(
    "Selected model:",
    winner["name"]
)
```

We select the model with the highest validation macro F1 score.

In the event of a tie, this example prefers the simpler baseline.

This is a basic model-selection strategy.

For production, you may also consider latency, inference cost, model size, and safety requirements.

## Step 6: Evaluate on held-out test data

Add:

```python
best_model = winner["model"]

test_predictions = best_model.predict(
    test_df["text"]
)

test_accuracy = accuracy_score(
    test_df["label"],
    test_predictions
)

test_macro_f1 = f1_score(
    test_df["label"],
    test_predictions,
    average="macro",
    zero_division=0
)

with mlflow.start_run(
    run_id=winner["run_id"]
):

    mlflow.log_metric(
        "test_accuracy",
        test_accuracy
    )

    mlflow.log_metric(
        "test_macro_f1",
        test_macro_f1
    )

    test_dataset = mlflow.data.from_pandas(
        test_df,
        name="held-out-test",
        source="synthetic-backend-errors-v1",
        targets="label"
    )

    mlflow.log_input(
        test_dataset,
        context="testing"
    )

print(
    "Test accuracy:",
    round(test_accuracy, 3)
)

print(
    "Test macro F1:",
    round(test_macro_f1, 3)
)
```

Notice the sequence:

```text
Train Models
     |
     v
Evaluate on Validation Set
     |
     v
Select Candidate
     |
     v
Evaluate on Held-Out Test Set
```

We are not using the test results to repeatedly choose new hyperparameters.

That distinction matters.

---

# Part 5: View Your Experiments

Run:

```bash
python train.py
```

Now start the MLflow tracking interface:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Open:

[MLflow Local Dashboard](http://127.0.0.1:5000?utm_source=chatgpt.com)

The local dashboard should display your experiment and its runs.

You can compare:

| Run | Validation accuracy | Validation macro F1 |
|---|---|---|
| Baseline | Measured value | Measured value |
| Logistic regression | Measured value | Measured value |

Your actual results will depend on the dataset and model behavior.

In MLflow, you can inspect parameters, metrics, dataset information, and logged model artifacts.

### Why is this useful?

Imagine running 100 experiments.

Without tracking, you might have:

```text
model_final.pkl
model_final2.pkl
model_latest.pkl
model_best.pkl
model_really_best.pkl
```

You don't know which dataset or configuration produced each one.

With experiment tracking, every model can be associated with its originating run and configuration.

---

# Part 6: Model Registry & Versioning

**Time: 40–47 minutes**

Now let's address another important problem.

You have trained several models.

**How do you decide which model should be used in production?**

This is where a Model Registry becomes useful.

## What is a Model Registry?

A Model Registry is a centralized system for managing model versions and lifecycle metadata.

MLflow's registry supports model versions, aliases, tags, and lineage information. [MLflow AI Platform](https://www.mlflow.org/docs/latest/model-registry/?utm_source=chatgpt.com)

Think of it as a managed catalog for machine learning models.

For example:

```text
Registered Model:
backend-error-classifier

        |
        ├── Version 1
        |
        ├── Version 2
        |
        └── Version 3
```

You can associate aliases with particular versions.

```text
candidate → Version 3

champion  → Version 2
```

Here:

**Candidate** represents a model being evaluated for deployment.

**Champion** represents the currently approved model.

These are conventions established by your team, not automatic quality guarantees.

## Register the selected model

Add this to the end of `train.py`:

```python
from mlflow import MlflowClient

registered = mlflow.register_model(
    model_uri=winner["model_uri"],
    name="backend-error-classifier"
)

client = MlflowClient()

client.set_registered_model_alias(
    name="backend-error-classifier",
    alias="candidate",
    version=registered.version
)

print(
    "Registered version:",
    registered.version
)
```

This registers the selected model and assigns it the `candidate` alias.

It does **not** automatically promote the model to production.

For a real system, promotion should depend on evaluation gates and deployment approval.

## Load the registered model

```python
loaded_model = mlflow.sklearn.load_model(
    "models:/backend-error-classifier@candidate"
)

prediction = loaded_model.predict([
    "PostgreSQL connection pool exhausted"
])

print(prediction)
```

The application loads whichever registered version currently has the `candidate` alias.

MLflow supports this alias-based loading pattern. [MLflow AI Platform](https://mlflow.org/docs/latest/ml/model-registry/tutorial?utm_source=chatgpt.com)

## How does rollback work?

Imagine:

```text
Version 1 → Approved
Version 2 → Approved
Version 3 → New candidate
```

You deploy Version 3 after validation.

Later, monitoring reveals a regression.

You can restore the previous approved version.

Conceptually:

```python
client.set_registered_model_alias(
    name="backend-error-classifier",
    alias="champion",
    version="2"
)
```

However, changing an alias alone does not necessarily replace a model already loaded into a running application.

Your serving infrastructure must explicitly reload the selected version or perform a controlled deployment.

A production rollback should also account for tokenizer, preprocessing, API contract, and dependency compatibility.

---

# Part 7: Production MLOps Architecture

**Time: 47–53 minutes**

Let's design how these components would work in a production AI platform.

We'll use technologies familiar to a Python backend developer.

```text
                 DATA SOURCES
                      |
          ┌───────────┴───────────┐
          |                       |
          v                       v
      PostgreSQL              Object Storage
          |                       |
          └───────────┬───────────┘
                      |
                      v
              Data Validation
                      |
                      v
             Dataset Versioning
                      |
                      v
               Training Job
                      |
                      v
              MLflow Tracking
                      |
              ┌───────┴───────┐
              |               |
              v               v
          PostgreSQL      Artifact Storage
           Metadata       Model Weights
              |               |
              └───────┬───────┘
                      |
                      v
                Model Registry
                      |
                      v
                Evaluation Gate
                      |
                      v
               Deployment Approval
                      |
                      v
                Model Serving
                      |
                      v
                    FastAPI
                      |
                      v
                  Monitoring
```

## Why separate metadata and artifacts?

MLflow distinguishes between its backend store and artifact store.

The backend store records information such as run IDs, parameters, metrics, and tags.

The artifact store holds files such as model weights and generated reports. [MLflow AI Platform](https://mlflow.org/docs/latest/tracking/backend-stores/?utm_source=chatgpt.com)

A production configuration might use:

| Component | Technology |
|---|---|
| Training API | FastAPI |
| Training orchestration | Celery or workflow orchestrator |
| Experiment tracking | MLflow |
| Metadata database | PostgreSQL |
| Model artifacts | S3-compatible object storage |
| Model serving | Dedicated inference service |
| Application deployment | Docker and Kubernetes |
| Monitoring | Prometheus, Grafana, OpenTelemetry |
| CI/CD | GitLab CI/CD |

These are architectural options, not requirements to use every technology.

For larger training workloads, a dedicated training orchestrator or managed ML platform may be preferable to running GPU training directly inside general-purpose Celery workers.

### Important distinction: MLflow is not a training engine

MLflow tracks and manages experiments.

It does not replace PyTorch, Transformers, or your training framework.

For example:

```text
PyTorch
   |
   v
Computes Gradients
   |
   v
Updates Model Weights
   |
   v
MLflow Records Results
```

MLflow provides the tracking and lifecycle infrastructure around training.

---

## Applying MLOps to our LoRA project

In Session 8, we trained LoRA adapters.

Now imagine running three experiments:

| Run | LoRA rank | Learning rate | Epochs |
|---|---:|---:|---:|
| A | 8 | 2e-4 | 2 |
| B | 16 | 2e-4 | 2 |
| C | 16 | 1e-4 | 3 |

We can record these values:

```python
with mlflow.start_run(
    run_name="lora-rank-16"
):

    mlflow.log_params({
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "epochs": 2
    })

    # Execute the fine-tuning process here.

    # After evaluation:
    mlflow.log_metric(
        "validation_macro_f1",
        0.87
    )
```

The metric above is an illustrative placeholder, not an actual training result.

You should log metrics calculated from your evaluation dataset.

For reproducibility, also record the exact base-model revision, adapter configuration, dataset version, random seed, and relevant dependency versions.

### Model version vs application version

These are different.

Suppose your FastAPI application is Version 2.3.

It might use Model Version 7.

```text
FastAPI v2.3
     |
     v
Model v7
```

You could deploy Model Version 8 without changing the application's source code, provided the interface remains compatible.

But this separation introduces another responsibility.

**You must track which application deployment used which model version.**

Otherwise, debugging production incidents becomes difficult.

---

# Part 8: CI/CD for Machine Learning

A conventional backend CI pipeline might execute:

```text
Lint
  ↓
Unit Tests
  ↓
Build Docker Image
  ↓
Deploy
```

An ML delivery pipeline introduces additional checks.

```text
Code Commit
     |
     v
Code Quality Tests
     |
     v
Data Schema Validation
     |
     v
Training Pipeline
     |
     v
Model Evaluation
     |
     v
Quality Threshold Check
     |
     v
Register Candidate
     |
     v
Security & Integration Tests
     |
     v
Deployment Approval
     |
     v
Canary Deployment
     |
     v
Production Monitoring
```

For example, a deployment gate could require:

```python
def validate_model(
    macro_f1: float,
    latency_ms: float
) -> bool:

    return (
        macro_f1 >= 0.90
        and latency_ms <= 200
    )
```

These thresholds are illustrative.

A production system should derive them from business requirements and an appropriate evaluation dataset.

Also, model quality is not the only consideration.

For an LLM-based system, you may need to evaluate hallucinations, structured-output validity, privacy risks, prompt-injection resistance, and tool-execution safety.

### What about continuous retraining?

You should not retrain a model automatically just because new data exists.

A safer approach is:

```text
New Data Available
        |
        v
Data Quality Checks
        |
        v
Retraining Trigger
        |
        v
Train Candidate
        |
        v
Compare Against Champion
        |
        v
Evaluate Regression Risk
        |
        v
Approve or Reject
```

Retraining is only one part of the lifecycle.

Evaluation determines whether the new model should be deployed.

---

# Part 9: Today's Practical Assignment

**Time: 53–57 minutes**

## Mini-Project: Build an MLOps Pipeline

Your assignment is to extend the backend error-classification project.

### Task 1 — Run three experiments

Train and track:

1. Majority-class baseline.
2. TF-IDF with Logistic Regression.
3. TF-IDF with a Random Forest classifier.

Use the same training and validation splits.

Record the parameters and metrics for every run.

### Task 2 — Compare results

Create a comparison table:

| Model | Validation macro F1 | Test macro F1 | Training time |
|---|---|---|---|
| Baseline | Measure | Final evaluation only | Measure |
| Logistic regression | Measure | Final evaluation only | Measure |
| Random Forest | Measure | Final evaluation only | Measure |

Select the best model using validation results.

Then evaluate that selected model on the held-out test set.

### Task 3 — Register the selected model

Register it in MLflow and assign the `candidate` alias.

### Task 4 — Add dataset tracking

Log the dataset hash and the dataset split information.

### Task 5 — Write a model-loading function

Create:

```python
def load_current_model():
    ...
```

It should load the registered model using an alias rather than a hardcoded local file path.

### Bonus: Integrate with FastAPI

Create a prediction endpoint:

```http
POST /api/v1/errors/classify
```

Request:

```json
{
  "message": "Redis MISCONF snapshot failure"
}
```

Response:

```json
{
  "category": "cache",
  "model_version": "1"
}
```

The model version should come from the actual deployed model metadata, not from a hardcoded value.

---

# Part 10: Senior AI Engineer Interview Preparation

**Time: 57–60 minutes**

### Q1. What is the difference between MLOps and DevOps?

DevOps focuses on reliable software delivery and operations.

MLOps extends these practices to include data pipelines, model training, evaluation, versioning, deployment, and monitoring.

### Q2. What is the difference between a parameter, metric, and artifact?

A parameter is a configuration value.

A metric is a measured result.

An artifact is a file or model output generated during an experiment.

### Q3. Why is dataset versioning important?

Because model behavior depends on training data.

Without identifying the exact dataset, reproducing or explaining a model's results becomes difficult.

### Q4. What is the purpose of a Model Registry?

To manage model versions, lineage, metadata, and references used in deployment workflows.

### Q5. What is the difference between model drift and data drift?

**Data drift** occurs when the statistical distribution of input data changes.

**Concept drift** occurs when the relationship between inputs and target outcomes changes.

A model's quality can deteriorate because of either, although drift does not automatically imply worse performance.

### Q6. How would you roll back a deployed model?

Identify the previously approved model version, verify compatibility, update the deployment configuration or registry alias, and ensure the serving infrastructure loads the intended version.

Then validate the rollback using health checks and representative predictions.

---

# 🧠 Today's Knowledge Quiz

Try answering these questions before our next session.

**Question 1**

You trained two models.

Model A:

```text
Learning rate: 0.0002
Epochs: 3
Validation accuracy: 91%
```

Model B:

```text
Learning rate: 0.0001
Epochs: 5
Validation accuracy: 94%
```

Which values are parameters, and which are metrics?

**Question 2**

Why is setting:

```python
random_state = 42
```

not sufficient to guarantee complete reproducibility?

**Question 3**

What is the difference between an MLflow experiment and an MLflow run?

**Question 4**

Your model achieves 99% training accuracy and 72% test accuracy.

What possible problem does this indicate?

How would you investigate it?

**Question 5**

Why should you avoid selecting hyperparameters repeatedly using your final test dataset?

**Question 6 — Advanced System Design**

Your company fine-tunes a 7-billion-parameter LLM every week.

Each training run costs significant GPU time.

You need to maintain dataset versions, track experiments, compare model quality, store LoRA adapters, deploy approved versions, and support rollback.

Design an architecture using Python, PyTorch, MLflow, PostgreSQL, object storage, and Docker.

Explain how you would handle failed training jobs, model evaluation, and production monitoring.

---

# Session 9 Recap

Today, you learned how to move from experimental model training toward reliable machine learning operations.

We covered experiment tracking, dataset lineage, reproducibility, model artifacts, model registries, deployment approvals, and production architecture.

The central MLOps workflow is:

\[
\boxed{
\text{Data}
\rightarrow
\text{Training}
\rightarrow
\text{Evaluation}
\rightarrow
\text{Registry}
\rightarrow
\text{Deployment}
\rightarrow
\text{Monitoring}
}
\]

A strong AI engineer should be able to explain not only how a model was trained, but also which data and configuration produced it, why it was selected, how it was deployed, and how to recover if it fails.

**Key takeaway:** Training a good model is only one part of AI engineering. Building a reproducible, observable, and reliable system around that model is what makes it production-ready.

---

## 🔜 Session 10: Production Model Serving with FastAPI, Docker & Inference Optimization

Next, we'll take the registered model from today's project and build a production-oriented inference service.

We'll cover synchronous and asynchronous inference, model loading, request batching, concurrency, worker memory, CPU versus GPU serving, and latency optimization.

We'll also examine the differences between serving conventional ML models and serving large language models, including continuous batching, KV caching, quantization, and inference engines such as vLLM.

**Your preparation:** Complete today's MLflow project and make sure you understand the difference between a training run, a logged model, and a registered model version.