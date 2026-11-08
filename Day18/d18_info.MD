# 🎓 AI Engineer Mentorship — Session 18
## Production MLOps: MLflow, Experiment Tracking, Dataset Versioning, Model Registry & CI/CD

**Level:** Intermediate → Advanced  
**Duration:** 60 minutes  
**Career path:** Python Backend Developer → Senior AI Engineer  
**Today's project:** Build an experiment-tracking and model-registration pipeline using Python and MLflow.

---

# 1. Continuing Your AI Engineering Journey

In Session 17, we learned how to fine-tune language models using LoRA, QLoRA, and PEFT.

We built an incident-classification example that could produce:

```json
{
  "priority": "P1",
  "team": "backend"
}
```

We also discussed training loss, validation loss, model evaluation, and saving LoRA adapters.

Today, we'll solve the next production problem.

Imagine you've trained five different versions of your incident-classification model.

Each version uses different training parameters.

```text
Model A → LoRA rank 8, learning rate 0.0002
Model B → LoRA rank 16, learning rate 0.0002
Model C → LoRA rank 8, learning rate 0.0001
Model D → LoRA rank 16, learning rate 0.0001
Model E → LoRA rank 32, learning rate 0.0001
```

After training, Model C appears to perform best.

Three weeks later, your manager asks:

> Which dataset was used to train Model C? What were its hyperparameters? Can we reproduce it? And can we safely deploy it to production?

If you cannot answer these questions, you have a serious production engineering problem.

This is where **MLOps — Machine Learning Operations** becomes important.

MLOps applies software engineering and operational practices to the machine learning lifecycle.

It helps teams build, test, version, deploy, monitor, and maintain models reliably.

## Today's learning objectives

By the end of this session, you should understand:

- MLOps versus traditional DevOps.
- Experiments, runs, metrics, parameters, and artifacts.
- Experiment tracking with MLflow.
- Dataset versioning and reproducibility.
- Model registries and model aliases.
- Automated model evaluation gates.
- CI/CD versus continuous training.
- Model deployment, rollback, and monitoring.
- How to apply MLOps to your LoRA fine-tuning project.

---

# ⏱️ 2. Your 60-Minute Agenda

| Time | Topic | Learning outcome |
|---|---|---|
| 0–10 min | MLOps fundamentals | Understand the ML lifecycle |
| 10–20 min | MLflow experiment tracking | Learn experiments, runs, and artifacts |
| 20–35 min | Practical Python project | Track and compare model training runs |
| 35–43 min | Model registry and versioning | Register and retrieve models |
| 43–50 min | Dataset versioning and reproducibility | Understand data lineage |
| 50–56 min | CI/CD and production architecture | Design automated deployment |
| 56–60 min | Exercise, interview questions, and quiz | Consolidate your knowledge |

---

# Part 1: What Is MLOps?
**Time: 0–10 minutes**

You already understand traditional backend development.

Suppose you're developing a FastAPI application.

Your workflow might be:

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
GitLab CI/CD
    |
    v
Automated Tests
    |
    v
Docker Build
    |
    v
Production Deployment
```

This is a conventional DevOps workflow.

Now consider a machine learning application.

Its behavior depends on more than source code.

It also depends on:

1. Training data.
2. Data preprocessing.
3. Model architecture.
4. Hyperparameters.
5. Training environment.
6. Random initialization and seeds.
7. Model weights.
8. Evaluation results.

Suppose your Python code doesn't change, but your training dataset changes.

You train the model again.

Will you necessarily get the same model?

**No.**

The new dataset may produce different model parameters and different predictions.

This is a fundamental difference between traditional software development and machine learning engineering.

## 1.1 The MLOps lifecycle

A typical production machine learning lifecycle looks like:

```text
                  Raw Data
                     |
                     v
              Data Validation
                     |
                     v
              Data Versioning
                     |
                     v
               Preprocessing
                     |
                     v
                Training
                     |
                     v
             Experiment Tracking
                     |
                     v
                Evaluation
                     |
                     v
               Model Registry
                     |
                     v
             Deployment Approval
                     |
                     v
             Production Serving
                     |
                     v
                Monitoring
                     |
                     v
             Feedback / Retraining
```

Notice that this is a continuous process.

The model doesn't become permanently correct after deployment.

Data distributions may change.

New categories may appear.

Model performance may degrade.

Therefore, MLOps covers the complete lifecycle rather than only training.

### DevOps vs MLOps

| Aspect | DevOps | MLOps |
|---|---|---|
| Primary artifact | Application code/build | Model and serving artifacts |
| Version control | Source code | Code, data, model, configuration |
| Testing | Unit and integration tests | Software tests and model evaluation |
| Deployment | Application release | Model release |
| Monitoring | Latency, errors, infrastructure | Infrastructure and model quality |
| Updates | Code changes | Code, data, or model changes |
| Rollback | Previous application version | Previous model and configuration |

### Interview insight

**Question: Why is MLOps more complex than traditional DevOps?**

A strong answer:

> Machine learning systems depend on code, data, model parameters, and training configurations. Even without changing the application code, changes in training data can produce different model behavior. MLOps manages these additional dependencies through experiment tracking, dataset versioning, model evaluation, registries, and production monitoring.

---

# Part 2: Understanding MLflow
**Time: 10–20 minutes**

Let's introduce [MLflow](https://mlflow.org/?utm_source=chatgpt.com).

MLflow is an open-source platform for managing machine learning and AI workflows.

Its capabilities include experiment tracking, model packaging, model registration, and lifecycle management.

For today's lesson, we'll focus on MLflow Tracking and the Model Registry.

Official documentation: [MLflow Tracking Guide](https://mlflow.org/docs/latest/ml/tracking/?utm_source=chatgpt.com). [MLflow AI Platform](https://www.mlflow.org/docs/latest/ml/tracking?utm_source=chatgpt.com)

## 2.1 What is an experiment?

An experiment groups related training runs.

Suppose your project is:

```text
Incident Classification
```

You want to compare different training configurations.

Create an experiment named:

```text
incident-classification
```

Inside it, you may have:

```text
Experiment: incident-classification
    |
    ├── Run 1: Logistic Regression C=0.1
    |
    ├── Run 2: Logistic Regression C=1.0
    |
    ├── Run 3: Logistic Regression C=10.0
    |
    ├── Run 4: LoRA rank=8
    |
    └── Run 5: LoRA rank=16
```

Each run represents an individual execution of a training workflow.

## 2.2 What information should a run contain?

A well-tracked run records several categories of information.

| Category | Example |
|---|---|
| Parameters | Learning rate, batch size, LoRA rank |
| Metrics | Accuracy, F1, validation loss |
| Artifacts | Model weights, reports, plots |
| Dataset metadata | Dataset version and digest |
| Source code | Git commit |
| Environment | Python and dependency versions |
| Model identity | Base model and adapter version |

### Parameters vs metrics

This distinction is important.

**Parameters** are values used to configure the experiment.

```python
learning_rate = 0.0002
batch_size = 4
lora_rank = 8
```

**Metrics** are measurements obtained during or after the experiment.

```python
validation_loss = 0.45
accuracy = 0.91
f1_score = 0.89
```

Parameters describe how you configured training.

Metrics describe how the model performed.

## 2.3 What is an artifact?

An artifact is an output file produced by an experiment.

Examples include:

```text
model.safetensors
adapter_config.json
evaluation_report.json
confusion_matrix.png
training_config.json
```

MLflow can associate these files with the training run.

That means you can later inspect which files were produced by a particular experiment.

---

# Part 3: Practical Python Project — Experiment Tracking
**Time: 20–35 minutes**

We're going to build a small incident-classification experiment.

Why not immediately fine-tune a 7B model?

Because today's objective is learning MLOps infrastructure, not spending time on GPU training.

We'll use a lightweight scikit-learn classifier first.

Afterward, we'll connect the same concepts to your LoRA project.

## Step 1: Create the project

```text
mlops-incident-classifier/
    |
    ├── train.py
    ├── mlflow.db
    └── requirements.txt
```

The database file will be created automatically.

Install dependencies:

```bash
pip install mlflow scikit-learn pandas
```

For a reproducible project, record the exact installed versions in a lockfile.

## Step 2: Configure MLflow

Create `train.py`.

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "incident-classification"
)
```

### Explanation

```python
mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)
```

Configures a local SQLite database for experiment metadata.

This is convenient for learning because you don't need to deploy a separate database server.

MLflow documents SQLite-backed local experiment tracking. [MLflow AI Platform](https://www.mlflow.org/docs/latest/ml/tracking/tutorials/local-database/?utm_source=chatgpt.com)

```python
mlflow.set_experiment(
    "incident-classification"
)
```

Creates or selects the experiment.

All our training runs will be grouped under this experiment.

---

## Step 3: Prepare a small dataset

We'll classify incidents into three teams:

```text
backend
infrastructure
security
```

Add:

```python
examples = {
    "backend": [
        "Payment API returns HTTP 500",
        "Login endpoint is timing out",
        "Orders API returns incorrect JSON",
        "Serializer fails for null values",
        "Webhook endpoint returns 502",
        "Request validation is broken",
        "Pagination endpoint crashes",
        "API response is missing fields",
        "Background task retries indefinitely",
    ],

    "infrastructure": [
        "PostgreSQL database is unavailable",
        "Redis cluster stopped responding",
        "Kubernetes node is offline",
        "Docker container keeps restarting",
        "Server disk storage is full",
        "Load balancer health checks fail",
        "Database connection pool is exhausted",
        "GPU worker ran out of memory",
        "Production server CPU is overloaded",
    ],

    "security": [
        "Unauthorized access was detected",
        "Production API credentials were leaked",
        "Suspicious login attempts increased",
        "Customer data was exposed",
        "Authentication tokens were stolen",
        "A secret key appeared in logs",
        "SQL injection attempt was detected",
        "An attacker bypassed access control",
        "Sensitive documents were accessed illegally",
    ],
}
```

Convert the examples into training inputs and labels:

```python
texts = []
labels = []

for team, incidents in examples.items():

    for incident in incidents:

        texts.append(incident)
        labels.append(team)
```

We now have 27 labeled examples.

This is only a demonstration dataset.

It is far too small to establish production-quality performance.

---

## Step 4: Split the dataset

Add:

```python
from sklearn.model_selection import train_test_split


X_dev, X_test, y_dev, y_test = train_test_split(
    texts,
    labels,
    test_size=0.22,
    random_state=42,
    stratify=labels
)


X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.30,
    random_state=42,
    stratify=y_dev
)
```

We now have three groups.

**Training set:** Used to fit the model.

**Validation set:** Used to compare hyperparameters and choose a candidate.

**Test set:** Used for a final evaluation after model selection.

### Important concept: Data leakage

Suppose you repeatedly evaluate different models against the test set and select whichever achieves the highest test accuracy.

That makes the test set part of your model-selection process.

Your final test performance may then be overly optimistic.

A better workflow is:

```text
Training Data → Train Models

Validation Data → Select Model

Test Data → Final Evaluation
```

For production, also check for near-duplicate incidents, shared customers, and temporal leakage across splits.

---

## Step 5: Train multiple models

We'll use TF-IDF and Logistic Regression.

Add:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


best_score = -1
best_model = None
best_model_uri = None
best_run_id = None


for c_value in [0.1, 1.0, 10.0]:

    with mlflow.start_run(
        run_name=f"logistic-regression-C-{c_value}"
    ) as run:

        model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2)
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    max_iter=1000,
                    random_state=42
                )
            )
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        accuracy = accuracy_score(
            y_val,
            predictions
        )

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )

        mlflow.log_params({
            "algorithm": "logistic_regression",
            "C": c_value,
            "max_iter": 1000,
            "ngram_range": "1,2"
        })

        mlflow.log_metrics({
            "validation_accuracy": accuracy,
            "validation_macro_f1": macro_f1
        })

        report = classification_report(
            y_val,
            predictions,
            output_dict=True,
            zero_division=0
        )

        mlflow.log_dict(
            report,
            "reports/validation_report.json"
        )

        model_info = mlflow.sklearn.log_model(
            model,
            name="incident_classifier"
        )

        if macro_f1 > best_score:

            best_score = macro_f1
            best_model = model
            best_model_uri = model_info.model_uri
            best_run_id = run.info.run_id

        print(
            f"C={c_value} | "
            f"Accuracy={accuracy:.3f} | "
            f"Macro F1={macro_f1:.3f}"
        )
```

### What happens?

The script trains three models with different regularization strengths.

For each model, MLflow records:

```text
Run ID
    |
    ├── Parameters
    |      └── C value
    |
    ├── Metrics
    |      ├── Accuracy
    |      └── Macro F1
    |
    ├── Artifacts
    |      └── Evaluation report
    |
    └── Model
           └── Trained classifier
```

The script also identifies the best model using validation macro F1.

### Why macro F1?

Macro F1 calculates F1 separately for each class and averages the results.

This is useful when you want to avoid overlooking poor performance on less common categories.

For example, a classifier that performs well on backend incidents but poorly on security incidents may have an acceptable overall accuracy while still being unsuitable for production.

---

## Step 6: Evaluate the selected model

After the training loop, add:

```python
test_predictions = best_model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

with mlflow.start_run(
    run_id=best_run_id
):

    mlflow.log_metrics({
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1
    })


print("Best validation F1:", best_score)
print("Final test F1:", test_macro_f1)
```

Now the selected model has both validation and test metrics recorded.

Remember that the test set contains very few examples, so the result has high uncertainty.

The important learning outcome is the experiment-management workflow, not the absolute score.

---

## Step 7: Run the training pipeline

Execute:

```bash
python train.py
```

You should see three training runs complete.

Next, start the MLflow UI:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --port 5000
```

Open the local MLflow UI at:

[MLflow local dashboard](http://127.0.0.1:5000?utm_source=chatgpt.com)

Select the `incident-classification` experiment.

You should be able to inspect the runs, compare metrics, and examine model artifacts.

### What should you look for?

Compare the three runs.

Identify which `C` value achieved the highest validation macro F1.

Inspect its evaluation report.

Then compare its validation and test performance.

This is your first practical MLOps experiment.

---

# Part 4: Model Registry and Model Versioning
**Time: 35–43 minutes**

Now imagine you've trained 100 different models.

Some are experimental.

Some are being evaluated.

One is currently serving production traffic.

How do you manage them?

This is the purpose of a **model registry**.

MLflow's Model Registry supports model versions, aliases, tags, and lineage between registered models and their originating runs. [MLflow AI Platform](https://mlflow.org/docs/latest/ml/model-registry/?utm_source=chatgpt.com)

Official guide: [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/?utm_source=chatgpt.com).

## 4.1 Understanding model versions

Suppose your registry contains:

```text
Registered Model:
incident-team-classifier
    |
    ├── Version 1
    |
    ├── Version 2
    |
    ├── Version 3
    |
    └── Version 4
```

Each version represents a registered model artifact.

You can associate metadata with each version.

For example:

```text
Version 3
    |
    ├── Source Run: abc123
    ├── Dataset Version: incidents-v3
    ├── Validation F1: 0.91
    ├── Test F1: 0.89
    └── Approval Status: passed
```

These metrics are illustrative.

## 4.2 Register your trained model

Add this after the final evaluation in `train.py`:

```python
from mlflow import MlflowClient


MODEL_NAME = "incident-team-classifier"

client = MlflowClient()


registered_version = mlflow.register_model(
    model_uri=best_model_uri,
    name=MODEL_NAME
)


quality_passed = test_macro_f1 >= 0.80

client.set_model_version_tag(
    name=MODEL_NAME,
    version=registered_version.version,
    key="quality_gate",
    value=(
        "passed"
        if quality_passed
        else "failed"
    )
)


print(
    "Registered candidate version:",
    registered_version.version
)

print(
    "Quality gate:",
    "PASSED" if quality_passed else "FAILED"
)
```

This registers the selected model as a candidate.

The `0.80` threshold is an illustrative training exercise, not a justified production requirement.

In a real system, the threshold should come from business requirements, baseline performance, and the cost of different errors.

### Important distinction

**Registering a model is not the same as deploying it.**

You can register a model even if it fails evaluation.

The registry can retain failed candidates for auditing and comparison.

Only approved versions should become eligible for production deployment.

---

## 4.3 What are model aliases?

Suppose your model registry contains:

```text
Version 1
Version 2
Version 3
Version 4
```

Instead of configuring production to use a hardcoded version, you can use an alias.

For example:

```text
champion → Version 3
```

Your inference service can load:

```python
model = mlflow.sklearn.load_model(
    "models:/incident-team-classifier@champion"
)
```

The alias identifies the approved model version.

After a new version passes evaluation and deployment approval, the alias can be reassigned.

For example:

```text
Before:

champion → Version 3


After approval:

champion → Version 4
```

### Python example

After an explicit approval decision:

```python
client.set_registered_model_alias(
    name="incident-team-classifier",
    alias="champion",
    version="4"
)
```

Replace `"4"` with the actual approved version.

MLflow supports loading models using version aliases. [MLflow AI Platform](https://mlflow.org/docs/latest/ml/model-registry/tutorial?utm_source=chatgpt.com)

### Production consideration

Updating an alias does not automatically replace a model already loaded into memory by a long-running inference process.

Your deployment system must deliberately load the approved version, validate readiness, and route traffic to it.

For controlled rollouts, pin the exact model version in each deployment revision rather than allowing an alias to change unexpectedly during runtime.

---

# Part 5: Dataset Versioning and Reproducibility
**Time: 43–50 minutes**

Now let's examine an important problem.

Suppose you train a model using:

```text
incidents.csv
```

The dataset contains 100,000 rows.

A month later, another developer modifies the file.

It now contains 120,000 rows.

You retrain the model.

But the file still has the same name:

```text
incidents.csv
```

How can you determine which data produced which model?

A filename is not a reliable dataset version.

## 5.1 What should be versioned?

For a reproducible training run, you should preserve:

| Component | Example |
|---|---|
| Source code | Git commit SHA |
| Dataset | Immutable dataset version |
| Preprocessing | Transformation code version |
| Training configuration | Learning rate, rank, epochs |
| Base model | Exact model revision |
| Dependencies | Locked package versions |
| Random seed | 42 |
| Output artifact | Model or adapter version |
| Evaluation | Metrics and test dataset version |

### Reproducibility vs determinism

Reproducibility means you have sufficient information and artifacts to repeat or inspect the experiment.

It does not necessarily mean every rerun will produce bit-for-bit identical weights.

GPU kernels, distributed training, library versions, and nondeterministic operations can affect results.

---

## 5.2 Dataset fingerprints

A simple approach is to calculate a cryptographic hash.

```python
import hashlib
from pathlib import Path


def calculate_sha256(file_path):

    digest = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


dataset_hash = calculate_sha256(
    "datasets/train.jsonl"
)

print(dataset_hash)
```

You can record the result in MLflow:

```python
mlflow.set_tag(
    "training_dataset_sha256",
    dataset_hash
)
```

The hash identifies the exact file contents.

If the file changes, its hash will almost certainly change.

However, a hash alone doesn't preserve the dataset.

You still need to store the corresponding data in a versioned location.

---

## 5.3 What is DVC?

[DVC — Data Version Control](https://dvc.org/?utm_source=chatgpt.com) is a tool for managing datasets, models, and data pipelines alongside Git.

DVC stores metadata about data versions in Git while managing the underlying data through its cache and configured storage. [Data Version Control · DVC](https://doc.dvc.org/start?utm_source=chatgpt.com)

Imagine this project:

```text
incident-ai/
    |
    ├── train.py
    ├── evaluate.py
    ├── datasets/
    │     └── incidents.csv
    |
    └── .git/
```

You don't necessarily want to commit a 20 GB dataset directly into Git.

Instead, you can track it using DVC.

Install:

```bash
pip install dvc
```

Initialize:

```bash
git init

dvc init
```

Track the dataset:

```bash
dvc add datasets/incidents.csv
```

DVC creates a metadata file such as:

```text
datasets/incidents.csv.dvc
```

Commit the metadata:

```bash
git add datasets/incidents.csv.dvc .gitignore

git commit -m "Track incident dataset"
```

For team collaboration, configure a DVC remote storage location and push the dataset:

```bash
dvc remote add -d storage <configured-storage-location>

dvc push
```

The remote location must be configured and accessible.

Later, another developer can retrieve the tracked version using:

```bash
dvc pull
```

### Why combine Git and DVC?

Git tracks the code and DVC metadata.

DVC manages the actual dataset versions.

Together, they allow you to associate a code revision with a particular dataset state.

---

# Part 6: MLOps CI/CD and Continuous Training
**Time: 50–56 minutes**

You already have experience with GitLab CI/CD.

Let's connect that knowledge to machine learning systems.

## 6.1 Traditional CI/CD

In backend development:

```text
Code Commit
    |
    v
Run Tests
    |
    v
Build Docker Image
    |
    v
Deploy Application
```

## 6.2 Machine learning CI/CD

For an ML application:

```text
Code or Data Change
        |
        v
   Data Validation
        |
        v
   Training Pipeline
        |
        v
  Experiment Tracking
        |
        v
    Model Evaluation
        |
        v
    Quality Gate
        |
   ┌────┴────┐
   |         |
   v         v
 Passed    Failed
   |         |
   v         v
 Register   Reject
   |
   v
 Staging Deployment
   |
   v
 Integration Tests
   |
   v
 Canary Deployment
   |
   v
 Production Monitoring
```

The important difference is that deployment should depend on model quality, not merely successful execution of the training script.

### Example quality gate

Suppose the current production model has:

```text
Macro F1: 0.88
```

A newly trained model achieves:

```text
Macro F1: 0.82
```

Should we deploy the new model just because training completed successfully?

No.

A deployment pipeline should compare the candidate against predefined acceptance criteria.

For example:

```python
def evaluate_release(
    candidate_f1,
    champion_f1,
    minimum_f1=0.85,
    allowed_regression=0.01
):

    return (
        candidate_f1 >= minimum_f1
        and
        candidate_f1 >= (
            champion_f1 - allowed_regression
        )
    )


approved = evaluate_release(
    candidate_f1=0.82,
    champion_f1=0.88
)

print(approved)
```

Output:

```text
False
```

The candidate fails the gate.

In production, the comparison must use equivalent evaluation datasets and metrics.

You may also require latency, cost, safety, fairness, and reliability checks.

## 6.3 CI, CD, and CT

There are three concepts worth distinguishing.

**Continuous Integration (CI):** Automatically validates changes to code and related configurations.

**Continuous Delivery/Deployment (CD):** Automates preparation and release of approved software or models.

**Continuous Training (CT):** Automates retraining when suitable triggers occur, such as new validated data, a scheduled retraining cycle, or detected performance degradation.

Continuous training should not automatically imply continuous deployment.

A newly trained model still needs evaluation and release controls.

---

# Part 7: Production MLOps Architecture

Now let's design an architecture suitable for a Senior AI Engineer.

Imagine you're building an AI incident-analysis platform.

It uses historical incidents to train a classifier and an LLM-based assistant to explain incidents.

Your architecture could look like:

```text
                   DATA SOURCES
                        |
            ┌───────────┴───────────┐
            |                       |
            v                       v
       PostgreSQL              Support Tickets
            |                       |
            └───────────┬───────────┘
                        |
                        v
                 Data Pipeline
                        |
                        v
                Data Validation
                        |
                        v
                Versioned Dataset
                        |
                        v
                 Training Pipeline
                        |
              ┌─────────┴─────────┐
              |                   |
              v                   v
         ML Classifier       LoRA Fine-Tuning
              |                   |
              └─────────┬─────────┘
                        |
                        v
                 MLflow Tracking
                        |
                        v
                 Evaluation Gates
                        |
                        v
                  Model Registry
                        |
                        v
                  Staging Tests
                        |
                        v
                Production Approval
                        |
                        v
                 Model Deployment
                        |
                        v
                   FastAPI
                        |
                        v
                 User Requests
                        |
                        v
                 Observability
                        |
                        v
              Quality Monitoring
                        |
                        v
               Retraining Decision
```

### Production responsibilities

A Senior AI Engineer should be able to answer these questions.

**Data lineage:** Which dataset and preprocessing version produced the model?

**Reproducibility:** Can we reconstruct the training environment?

**Model quality:** Does the candidate outperform the baseline on representative evaluation data?

**Deployment safety:** Can we test the model before exposing it to all users?

**Rollback:** Can we quickly restore the previous model version?

**Monitoring:** Can we detect changes in input distributions, model quality, latency, and costs?

**Security:** Who can access training datasets, model artifacts, and the registry?

---

# Part 8: Connecting MLOps to Your LoRA Project

Let's return to Session 17.

Previously, we trained a LoRA adapter using `SFTTrainer`.

Now we can track that training process using MLflow.

The following example assumes you've already created `trainer`, `training_args`, and your evaluation functions.

```python
import mlflow


mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "incident-lora-finetuning"
)


with mlflow.start_run(
    run_name="qwen-lora-rank-8"
):

    mlflow.log_params({
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "lora_rank": 8,
        "lora_alpha": 16,
        "learning_rate": 2e-4,
        "epochs": 3
    })

    train_result = trainer.train()

    mlflow.log_metric(
        "train_loss",
        train_result.training_loss
    )

    trainer.save_model(
        "./incident-lora-final"
    )

    mlflow.log_artifacts(
        "./incident-lora-final",
        artifact_path="lora_adapter"
    )
```

This connects your training configuration and adapter files to an experiment run.

You should also log validation metrics, dataset identifiers, the exact base-model revision, and final held-out evaluation results.

For larger models, storing adapters in a dedicated artifact store is preferable to relying on local disk.

### Why is this useful?

Imagine you train three LoRA adapters.

```text
Run A → rank 8  → Validation F1 0.84
Run B → rank 16 → Validation F1 0.89
Run C → rank 32 → Validation F1 0.87
```

These values are hypothetical.

You can compare them in MLflow, inspect their configurations, and identify the best candidate.

You can then register the approved adapter as a versioned artifact and deploy it with the correct base model.

**Important:** A LoRA adapter is not a standalone LLM. The serving system must load it with a compatible base model.

---

# 🛠️ Part 9: Today's Practical Assignment
**Time: 56–60 minutes**

## Mini-Project: Build an MLOps Pipeline for Incident Classification

Your task is to extend today's experiment-tracking project.

### Task 1 — Experiment tracking

Train at least three classifier configurations.

Record their parameters, validation metrics, and artifacts in MLflow.

### Task 2 — Dataset versioning

Save the incident dataset as a CSV or JSONL file.

Calculate its SHA-256 hash and record the hash in each experiment.

### Task 3 — Model registry

Register the best validation candidate.

Add a model-version tag indicating whether it passed the final evaluation gate.

### Task 4 — Model loading

Write a Python script that loads an approved registered model version and predicts the team responsible for a new incident.

### Task 5 — CI/CD design

Design a GitLab CI/CD pipeline with stages for data validation, training, evaluation, registration, staging deployment, and approval.

### Bonus challenge

Connect MLflow to your LoRA fine-tuning project.

Record the LoRA rank, alpha, learning rate, training loss, validation loss, adapter artifacts, and dataset version.

### Expected project structure

```text
incident-mlops/
    |
    ├── datasets/
    │   ├── train.jsonl
    │   ├── validation.jsonl
    │   └── test.jsonl
    |
    ├── train.py
    ├── evaluate.py
    ├── register_model.py
    ├── predict.py
    |
    ├── requirements.txt
    ├── .gitlab-ci.yml
    └── README.md
```

---

# 🎯 Senior AI Engineer Interview Preparation

### Q1. What is MLOps?

MLOps is a set of practices for managing the machine learning lifecycle, including data preparation, training, experiment tracking, evaluation, deployment, monitoring, and retraining.

### Q2. What is the difference between an experiment and a run in MLflow?

An experiment groups related training activities.

A run represents one execution and records its parameters, metrics, and artifacts.

### Q3. Why is dataset versioning important?

Because changes in training data can change model behavior even when the training code remains unchanged.

Dataset versioning helps establish lineage and reproducibility.

### Q4. What is a model registry?

A model registry is a centralized system for managing model versions, metadata, lineage, and lifecycle information.

It helps teams identify and deploy approved model artifacts.

### Q5. What is the difference between model registration and deployment?

Registration stores and organizes a model version.

Deployment makes a particular model version available to serve predictions.

### Q6. What is continuous training?

Continuous training is the automated execution of a training pipeline based on suitable triggers.

The resulting model must still pass evaluation and deployment requirements.

### Q7. How would you roll back a poorly performing AI model?

I would identify the previously approved model version, restore its deployment configuration, and redirect traffic after verifying readiness.

I would also investigate the failed release using experiment records, evaluation results, and production telemetry.

### Q8. How would you make an LLM fine-tuning pipeline reproducible?

I would preserve the base-model revision, dataset version, preprocessing code, training configuration, random seeds, dependency versions, adapter artifacts, and evaluation results.

I would also document any remaining sources of nondeterminism.

---

# 🧠 Session 18 Quiz

Try answering these questions before checking the answers.

**Question 1:** What is the difference between MLOps and traditional DevOps?

**Question 2:** In MLflow, what is the difference between a parameter, metric, and artifact?

**Question 3:** Why should we not repeatedly use the final test dataset to choose hyperparameters?

**Question 4:** Your training code hasn't changed, but your training dataset has changed. Why might the new model produce different predictions?

**Question 5:** What is the difference between registering a model and deploying it?

**Question 6:** What does a `champion` model alias represent?

**Question 7 — System Design Challenge:**

Your company has an AI incident-classification system serving 50,000 requests daily.

You retrain the model every month.

The current production model has an F1 score of 0.91.

A new candidate has an F1 score of 0.88.

Your manager wants to deploy the newer model because it was trained using more recent data.

Explain how you would evaluate this request.

Consider whether the scores were measured on comparable datasets, whether the newer model performs better on recent incidents, whether critical categories have regressed, and how you would test the candidate safely before a full release.

## Quick answer key

| Question | Answer |
|---|---|
| 1 | MLOps manages data and model dependencies in addition to software. |
| 2 | Parameters configure training; metrics measure results; artifacts are output files. |
| 3 | It introduces selection bias and can make test performance overly optimistic. |
| 4 | Model parameters depend on training data. |
| 5 | Registration manages the artifact; deployment serves it. |
| 6 | A named reference to a designated model version, often the approved production candidate. |

For Question 7, a strong answer would not automatically reject or deploy the candidate based on the two F1 scores alone. It would investigate the evaluation conditions and business impact before making a release decision.

---

# Session 18 Recap

Today, you learned how to move from training individual models to managing repeatable machine learning workflows.

We covered:

| Concept | Key takeaway |
|---|---|
| MLOps | Manages the complete ML lifecycle |
| Experiment tracking | Records training configurations and results |
| MLflow | Tracks experiments and model artifacts |
| Dataset versioning | Identifies training data revisions |
| Model registry | Manages model versions and metadata |
| Model aliases | Provide named references to versions |
| Evaluation gates | Prevent unsuitable models from being released |
| Continuous training | Automates retraining workflows |
| Model rollback | Restores a previously approved version |

The central principle is:

\[
\boxed{
\text{Production AI}
=
\text{Model Quality}
+
\text{Reproducibility}
+
\text{Deployment Reliability}
+
\text{Monitoring}
}
\]

Training a model is only one part of AI engineering.

A Senior AI Engineer must be able to explain how a model was produced, why it was selected, how it was deployed, and how the organization can safely replace or roll it back.

---

# 🔜 Session 19: Advanced LLM Evaluation & Continuous Quality Monitoring

In the next session, we'll connect MLOps with LLMOps.

We'll explore how evaluation changes when a model generates natural-language answers rather than fixed classification labels.

We'll cover automated evaluation pipelines, golden datasets, LLM-as-a-Judge reliability, human evaluation, regression testing, RAG retrieval metrics, agent trajectory evaluation, and quality monitoring after deployment.

**Practical project:** Build an automated evaluation pipeline for a RAG or agent application, record results in MLflow, and prevent deployment when a candidate fails predefined quality checks.

### Your priority before Session 19

Complete the MLflow experiment-tracking exercise and understand the difference between an **experiment run, a registered model version, and a deployed model**.

These three concepts are foundational to production MLOps and will also help you manage LLM fine-tuning and RAG deployments as your AI engineering projects become more complex.