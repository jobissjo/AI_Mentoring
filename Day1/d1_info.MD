Set correctly now.

Your **AI Engineer mentoring guidance** will come every:
- **Monday — 10:00 AM**
- **Tuesday — 10:00 AM**
- **Wednesday — 10:00 AM**
- **Friday — 10:00 AM**

Each session is designed for about **1 hour of learning**, progressing from your Python backend background toward AI/ML, deep learning, LLMs, RAG, agents, fine-tuning, MLOps, deployment, and Senior AI Engineer-level system design.

# 🎓 AI Engineer Mentorship — Session 1
## From Python Backend Developer to AI Engineer
**Today's topic: Neural Networks, PyTorch Autograd & Model Training**

**Duration:** 60 minutes  
**Level:** Intermediate Python → AI Engineering  
**Goal:** Understand how neural networks learn and build your first complete machine-learning training pipeline.

---

## 🎯 Your AI Engineer Roadmap

We're working toward making you capable of designing, developing, and deploying production-grade AI systems—not just calling LLM APIs.

Our learning path will progress through five stages:

| Stage | Topics | Outcome |
|---|---|---|
| 1. ML Foundations | Neural networks, gradients, PyTorch, optimization | Train ML models |
| 2. Deep Learning & LLMs | Embeddings, transformers, attention, tokenization | Understand how LLMs work |
| 3. AI Application Engineering | RAG, vector databases, agents, tool calling | Build intelligent applications |
| 4. Advanced AI | Fine-tuning, LoRA, QLoRA, evaluation | Customize and evaluate models |
| 5. Production AI | MLOps, inference, FastAPI, observability, scaling | Architect production AI systems |

Since you've already explored gradient descent, `dw`, `db`, and the basics of transformer attention, we'll build on those concepts rather than starting from zero.

# ⏱️ Today's 60-Minute Plan

| Time | Topic |
|---|---|
| 0–10 min | How machine learning differs from backend programming |
| 10–20 min | Neural networks and forward propagation |
| 20–30 min | Loss functions, gradients, and backpropagation |
| 30–45 min | Build a neural network using PyTorch |
| 45–50 min | Understand the production ML pipeline |
| 50–60 min | Coding assignment and knowledge assessment |

---

# Part 1: Traditional Programming vs Machine Learning
### ⏱️ 0–10 minutes

As a Python backend developer, you're familiar with writing business rules.

Imagine you have an API that processes customer requests.

You want to predict whether a request will be slow.

### Traditional programming

You might write:

```python
def is_slow_request(payload_kb, db_queries):
    if payload_kb > 500:
        return True

    if db_queries > 20:
        return True

    return False
```

Here, **you explicitly define the rules**.

But what if request performance depends on complicated relationships between:

- Payload size
- Number of database queries
- Server load
- Cache hit ratio
- Query complexity

Writing rules for every combination becomes difficult.

### Machine learning approach

Instead of writing every rule, collect historical examples.

| Payload KB | DB Queries | Slow Request |
|---|---|---|
| 100 | 2 | 0 |
| 500 | 15 | 1 |
| 50 | 25 | 1 |
| 200 | 3 | 0 |
| 800 | 20 | 1 |

Here:

**Features (X):**

```python
X = [
    [100, 2],
    [500, 15],
    [50, 25],
    [200, 3],
    [800, 20]
]
```

**Labels (y):**

```python
y = [0, 1, 1, 0, 1]
```

The model learns a function:

\[
f(X) \rightarrow y
\]

Its job is to discover relationships between inputs and outcomes.

This is called **supervised learning** because we provide examples with known labels.

### Important distinction

A machine-learning model does not necessarily discover the true underlying cause of an outcome.

It learns statistical patterns from training data.

This distinction matters when designing reliable AI systems.

---

# Part 2: How a Neural Network Works
### ⏱️ 10–20 minutes

Let's start with a single artificial neuron.

A neuron computes:

\[
z = w_1x_1 + w_2x_2 + b
\]

Where:

- \(x_1, x_2\): input features
- \(w_1, w_2\): trainable weights
- \(b\): trainable bias
- \(z\): weighted sum

Consider:

```python
payload_size = 500
db_queries = 10

w1 = 0.002
w2 = 0.1
b = -1.5

z = (
    payload_size * w1
    + db_queries * w2
    + b
)

print(z)
```

Output:

```text
0.5
```

But we want to predict a probability.

For binary classification, we can apply the sigmoid function:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

```python
import math

probability = 1 / (1 + math.exp(-z))

print(probability)
```

Output:

```text
0.622459...
```

Our model predicts approximately a **62.2% probability** of a slow request.

Notice something important.

We never wrote:

```python
if payload_size > 500:
```

Instead, the model uses learned weights.

## Why do we need multiple layers?

A single linear neuron can only learn a linear decision boundary.

Real problems are often nonlinear.

A neural network combines multiple neurons and nonlinear activation functions.

For example:

```text
Input Layer
    |
    |-- Payload Size
    |-- DB Queries
    |
    v
Hidden Layer
    |
    |-- Neuron 1
    |-- Neuron 2
    |-- Neuron 3
    |-- Neuron 4
    |
    v
Output Layer
    |
    v
Probability of Slow Request
```

Mathematically:

\[
h = ReLU(W_1X+b_1)
\]

\[
z = W_2h+b_2
\]

\[
P(y=1|X)=\sigma(z)
\]

**ReLU** introduces nonlinearity:

\[
ReLU(x)=\max(0,x)
\]

Without nonlinear activation functions, stacking ordinary linear layers is equivalent to a single linear transformation.

That means depth alone would not provide additional expressive power.

---

# Part 3: How Neural Networks Learn
### ⏱️ 20–30 minutes

This is one of the most important concepts in machine learning.

A neural network starts with initial weights.

Those weights are generally not useful yet.

Training adjusts them to reduce prediction errors.

## Step 1: Forward propagation

Pass the inputs through the model.

```python
prediction = model(inputs)
```

The model produces an output.

## Step 2: Calculate loss

We need to measure how wrong the prediction is.

For binary classification, a common loss function is binary cross-entropy:

\[
L=-[y\log(p)+(1-y)\log(1-p)]
\]

Where:

- \(y\): actual label
- \(p\): predicted probability

A smaller loss generally means predictions fit the training labels better.

## Step 3: Backpropagation

We calculate how the loss changes with respect to each trainable parameter.

For a weight \(w\):

\[
\frac{\partial L}{\partial w}
\]

This derivative is called a **gradient**.

You previously encountered:

```python
dw = 2 * error * x
db = 2 * error
```

Those were gradients for a simple squared-error model.

Neural networks use the same underlying mathematical principle.

The difference is that they contain many more parameters and operations.

PyTorch automatically calculates the gradients using automatic differentiation.

## Step 4: Update weights

Gradient descent follows:

\[
w_{new}=w_{old}-\alpha\frac{\partial L}{\partial w}
\]

Here, \(\alpha\) is the learning rate.

A small learning rate usually produces smaller parameter updates.

A very large learning rate can make training unstable.

### PyTorch implementation

```python
import torch

w = torch.tensor(2.0, requires_grad=True)

x = torch.tensor(3.0)
y = torch.tensor(10.0)

prediction = w * x

loss = (prediction - y) ** 2

loss.backward()

print("Prediction:", prediction.item())
print("Loss:", loss.item())
print("Gradient:", w.grad.item())
```

Expected output:

```text
Prediction: 6.0
Loss: 16.0
Gradient: -24.0
```

Why is the gradient `-24`?

Our loss is:

\[
L=(wx-y)^2
\]

Differentiating with respect to \(w\):

\[
\frac{\partial L}{\partial w}=2(wx-y)x
\]

Substitute:

\[
2(6-10)(3)=-24
\]

A negative gradient tells us that increasing the weight slightly would reduce the loss locally.

This is the foundation of neural-network training.

---

# Part 4: Build a Neural Network with PyTorch
### ⏱️ 30–45 minutes

Now let's build a complete training pipeline.

**Project: Predict whether an API request will be slow.**

We'll use synthetic data so you can run the example without needing a production dataset.

First install PyTorch:

```bash
pip install torch
```

## Step 1: Generate training data

```python
import torch
from torch import nn

torch.manual_seed(42)

n = 2000

# Generate synthetic API request features
payload_kb = torch.rand(n, 1) * 1000

db_queries = torch.randint(
    0, 30, (n, 1)
).float()

X = torch.cat(
    [payload_kb, db_queries],
    dim=1
)

# Synthetic relationship between inputs and latency risk
score = (
    1.5 * (payload_kb[:, 0] / 1000)
    + 2.5 * (db_queries[:, 0] / 30)
    + 1.2 * (payload_kb[:, 0] / 1000)
          * (db_queries[:, 0] / 30)
    - 2.5
    + torch.randn(n) * 0.5
)

y = (score > 0).float().unsqueeze(1)

print("Features:", X.shape)
print("Labels:", y.shape)
```

Output:

```text
Features: torch.Size([2000, 2])
Labels: torch.Size([2000, 1])
```

Think about these dimensions.

`[2000, 2]` means:

- 2,000 training examples
- 2 features per example

The label tensor has shape `[2000, 1]`.

Each request has one binary target.

## Step 2: Split the dataset

We must evaluate the model on data it has not trained on.

```python
indices = torch.randperm(n)

train_idx = indices[:1600]
val_idx = indices[1600:1800]
test_idx = indices[1800:]

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]
```

We now have:

| Dataset | Examples | Purpose |
|---|---:|---|
| Training | 1,600 | Learn model parameters |
| Validation | 200 | Evaluate during development |
| Test | 200 | Final evaluation |

For real API logs, a chronological split may be more appropriate than a random split because traffic patterns change over time.

## Step 3: Normalize features

Payload size and database-query count have different scales.

Normalization helps optimization.

```python
mean = X_train.mean(dim=0)

std = X_train.std(dim=0).clamp_min(1e-8)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std
```

**Important:** We calculate normalization statistics using training data only.

Using validation or test data to calculate them would introduce data leakage.

## Step 4: Create the neural network

```python
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

print(model)
```

Output:

```text
Sequential(
  (0): Linear(in_features=2, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=1, bias=True)
)
```

Let's understand the architecture.

**First layer:**

```python
nn.Linear(2, 16)
```

Receives two input features and produces 16 hidden features.

**Activation:**

```python
nn.ReLU()
```

Introduces nonlinearity.

**Output layer:**

```python
nn.Linear(16, 1)
```

Produces one output called a **logit**.

A logit is an unnormalized score.

We'll convert it into a probability later using sigmoid.

## Step 5: Define loss and optimizer

```python
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)
```

Why `BCEWithLogitsLoss`?

Because this is a binary classification problem.

It combines sigmoid and binary cross-entropy in a numerically stable operation.

Therefore, we do not add sigmoid directly to the model during training.

## Step 6: Train the model

```python
epochs = 100

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    logits = model(X_train)

    loss = criterion(logits, y_train)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 20 == 0:

        model.eval()

        with torch.no_grad():

            val_logits = model(X_val)

            val_loss = criterion(
                val_logits,
                y_val
            )

        print(
            f"Epoch {epoch + 1}, "
            f"Train Loss: {loss.item():.4f}, "
            f"Val Loss: {val_loss.item():.4f}"
        )
```

Let's examine the five most important lines.

```python
model.train()
```

Enables training behavior for layers that need it, such as dropout and batch normalization.

```python
optimizer.zero_grad()
```

Clears previously accumulated gradients.

```python
logits = model(X_train)
```

Performs forward propagation.

```python
loss.backward()
```

Calculates gradients using backpropagation.

```python
optimizer.step()
```

Updates model parameters using the optimizer.

**These operations form the core training loop used in many deep-learning systems.**

## Step 7: Evaluate on unseen data

```python
model.eval()

with torch.no_grad():

    logits = model(X_test)

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= 0.5
    ).float()

    accuracy = (
        predictions == y_test
    ).float().mean()

print("Test Accuracy:", accuracy.item())
```

Your exact result depends on training and data generation.

A higher test accuracy suggests the model has learned patterns that generalize to unseen synthetic examples.

However, accuracy alone is not enough for production monitoring.

For an API latency classifier, we would also examine precision, recall, false-positive rate, and performance across different traffic conditions.

---

# Part 5: Thinking Like a Production AI Engineer
### ⏱️ 45–50 minutes

Training a model is only one part of building an AI system.

Compare a traditional backend workflow:

```text
HTTP Request
    ↓
FastAPI
    ↓
Business Logic
    ↓
Database
    ↓
HTTP Response
```

With an ML-powered backend:

```text
Historical API Logs
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Training Pipeline
    ↓
Model Evaluation
    ↓
Model Artifact
    ↓
Inference Service
    ↓
Monitoring
```

The deployed inference path might look like:

```text
HTTP Request
    ↓
FastAPI
    ↓
Input Validation
    ↓
Feature Transformation
    ↓
Trained PyTorch Model
    ↓
Probability / Prediction
    ↓
HTTP Response
```

Your existing Python backend knowledge is particularly useful for the inference-service portion.

But there are additional responsibilities.

For example, a model might achieve 95% accuracy during testing and perform poorly after deployment because the incoming data distribution changes.

This is called **data drift**.

An AI engineer needs to understand both model behavior and software-system reliability.

### Three production lessons

**1. Never train and evaluate on the same examples.**

Otherwise, you may measure memorization rather than generalization.

**2. Keep preprocessing consistent.**

The normalization used during training must also be used during inference.

**3. Monitor more than application uptime.**

A healthy API can still serve poor predictions.

You need model-quality monitoring alongside conventional latency, error-rate, and infrastructure monitoring.

---

# Part 6: Your Coding Assignment
### ⏱️ 50–60 minutes

Now it's your turn.

I want you to modify today's neural-network project.

## Assignment 1: Understand model parameters

Our network contains:

```python
nn.Linear(2, 16)
nn.ReLU()
nn.Linear(16, 1)
```

**Question:** How many trainable parameters does this model contain?

Remember:

\[
Parameters=(Inputs\times Outputs)+Biases
\]

Calculate the parameters for each linear layer and then add them together.

## Assignment 2: Experiment with learning rates

Train the same model using:

```python
learning_rates = [
    0.0001,
    0.01,
    0.1
]
```

Record the training and validation losses after 100 epochs.

Try to answer:

- Which learning rate converges fastest?
- Does the highest learning rate always perform best?
- What happens when the learning rate is too small?

## Assignment 3: Add a hidden layer

Modify the model to use this architecture:

```text
Input (2)
   ↓
Linear (2 → 32)
   ↓
ReLU
   ↓
Linear (32 → 16)
   ↓
ReLU
   ↓
Linear (16 → 1)
```

Implement it using `nn.Sequential`.

Compare the training and validation losses against the original network.

Does adding more layers always improve performance?

Explain your observation.

---

# 🧠 Knowledge Assessment

Try answering these questions without looking back at the lesson.

**Q1.** What is the fundamental difference between traditional programming and supervised machine learning?

**Q2.** Why do neural networks need nonlinear activation functions such as ReLU?

**Q3.** Consider:

```python
loss.backward()
optimizer.step()
```

What exactly happens during each operation?

**Q4.** Why should normalization statistics be calculated using training data rather than the entire dataset?

**Q5.** Imagine your model achieves:

```text
Training Accuracy:   99%
Validation Accuracy: 65%
```

What problem might this indicate? How would you investigate it?

---

## 🎯 What You Should Understand After Today's Session

By the end of this session, you should be able to explain the relationship between a neuron, a weight, an activation function, a loss function, a gradient, and an optimizer.

You should also be comfortable reading and modifying a basic PyTorch training loop, distinguishing training from evaluation, and recognizing why generalization matters.

For additional hands-on practice, the official [PyTorch Quickstart Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html?utm_source=chatgpt.com) covers datasets, models, optimization, and saving trained models.

### Next Session Preview: Tokenization and Embeddings

Our next lesson will connect neural networks to modern LLM engineering.

We'll explore how text becomes numerical representations, why tokenization matters, how embedding vectors represent information, and how those representations lead into transformer architectures.

We'll also build a small semantic-search example in Python.

**Today's priority: Complete the three coding assignments and answer Q1–Q5.** Understanding the training loop deeply will make transformers, fine-tuning, and production LLM engineering much easier to learn.