# 🎓 AI Engineer Mentorship — Session 8
## Fine-Tuning LLMs: LoRA, QLoRA, Training Data & Model Customization

**Level:** Intermediate → Advanced AI Engineering  
**Duration:** 60 minutes  
**Today's project:** Understand LoRA mathematically, implement it using PyTorch, and prepare a small LLM fine-tuning experiment.

In our previous sessions, we learned how to build RAG applications, develop stateful AI agents, and evaluate their reliability.

Until now, we've primarily used pretrained models without modifying their weights.

Today, we're moving into **model training and customization**.

This is an important transition. Instead of only building applications around LLMs, you'll learn how to modify their behavior through training.

Our central question is:

> When should an AI engineer fine-tune an LLM instead of improving prompts, retrieval, or application logic?

---

# 1. Today's Learning Objectives

By the end of this session, you should understand:

1. The difference between pretraining, supervised fine-tuning, and preference optimization.
2. When fine-tuning is appropriate and when RAG is better.
3. How supervised fine-tuning updates model parameters.
4. Why full fine-tuning requires substantial memory.
5. How Low-Rank Adaptation (LoRA) works mathematically.
6. How QLoRA reduces memory requirements.
7. How to prepare and validate instruction-tuning datasets.
8. How to evaluate a fine-tuned model before deployment.

### ⏱️ Your 60-Minute Learning Plan

| Time | Topic |
|---|---|
| 0–10 min | Pretraining vs fine-tuning vs RAG |
| 10–20 min | Supervised fine-tuning and loss functions |
| 20–35 min | LoRA mathematics and PyTorch implementation |
| 35–45 min | Practical LLM fine-tuning with Hugging Face |
| 45–52 min | QLoRA, memory optimization, and production considerations |
| 52–60 min | Assignment, interview questions, and quiz |

---

# Part 1: What Exactly Is Fine-Tuning?

**Time: 0–10 minutes**

Let's use a scenario from backend development.

Imagine you're building an AI assistant that analyzes Python application errors.

You want it to receive an error message and generate structured output.

**Input:**

```text
django.db.utils.OperationalError:
too many connections
```

**Expected output:**

```json
{
  "category": "database",
  "severity": "high",
  "probable_cause": "Database connection limit reached",
  "recommended_action": "Inspect active database connections"
}
```

You could initially achieve this using prompt engineering and structured-output validation.

But suppose your company processes hundreds of thousands of error reports with specialized categories and formatting requirements.

You discover that the pretrained model repeatedly misclassifies certain internal error patterns.

After collecting enough high-quality examples, fine-tuning might become useful.

## Three stages of model training

### Stage A: Pretraining

During pretraining, an LLM learns broad statistical patterns from large datasets.

For an autoregressive language model, the central objective is next-token prediction.

Given:

```text
PostgreSQL is a relational
```

The model learns to predict a likely next token.

Mathematically:

\[
P(x_t\mid x_1,x_2,\ldots,x_{t-1})
\]

Its parameters are updated across many training examples.

The result is a foundation model capable of handling a wide range of language tasks.

### Stage B: Supervised Fine-Tuning (SFT)

Now we provide examples of desired behavior.

For example:

```json
{
  "prompt": "Explain HTTP status code 404",
  "completion": "HTTP 404 indicates that the requested resource was not found."
}
```

The model learns to produce outputs resembling the target responses.

This process is called **supervised fine-tuning**.

We start with pretrained weights rather than initializing the model from scratch.

### Stage C: Preference Optimization

After supervised fine-tuning, we may want the model to prefer certain responses over others.

For example:

**Prompt:**

> How should I investigate a slow PostgreSQL query?

**Preferred response:**

> Inspect the execution plan using EXPLAIN ANALYZE, review indexes, and measure query execution time.

**Less-preferred response:**

> Restart PostgreSQL immediately.

Preference-training methods such as Direct Preference Optimization (DPO) can use examples of preferred and rejected responses.

We'll study DPO separately in a later advanced lesson.

## When should you fine-tune?

This is an important architectural decision.

| Requirement | Usually start with |
|---|---|
| Answer questions about internal documentation | RAG |
| Access frequently changing information | RAG or tools |
| Produce consistent structured outputs | Prompting and schema validation |
| Improve specialized output behavior | Fine-tuning |
| Learn organization-specific terminology | RAG first; consider fine-tuning if needed |
| Execute external operations | Tool calling |
| Improve a repeated domain-specific task | Fine-tuning may help |
| Enforce authorization | Application code |

**Important:** Fine-tuning is not a replacement for a database.

If your company updates its policies every day, retraining the model after every update is generally less practical than retrieving the latest policy when needed.

### Senior AI Engineer insight

Before fine-tuning, ask:

**What measurable failure are we trying to correct?**

If you cannot identify the failure and measure improvement, you are not ready to fine-tune.

---

# Part 2: How Supervised Fine-Tuning Works Internally

**Time: 10–20 minutes**

Let's connect this to Session 1, where we studied PyTorch, loss functions, and backpropagation.

Remember our training loop:

```python
optimizer.zero_grad()

prediction = model(inputs)

loss = criterion(prediction, labels)

loss.backward()

optimizer.step()
```

LLM fine-tuning uses the same underlying training principles.

The main differences involve model architecture, token-level targets, memory requirements, and training objectives.

## Step 1: Prepare an instruction-response pair

Suppose our dataset contains:

```python
example = {
    "prompt": "What is HTTP 500?",
    "completion": (
        "HTTP 500 indicates an internal server error."
    )
}
```

During preprocessing, the model's tokenizer converts the text into token IDs.

Conceptually:

```text
Prompt:
What is HTTP 500?

Target:
HTTP 500 indicates an internal server error.

              ↓

          Tokenizer

              ↓

          Token IDs

              ↓

             LLM

              ↓

        Predicted Tokens

              ↓

         Cross-Entropy Loss
```

## Step 2: Calculate token-level loss

Suppose the expected output tokens are:

```text
HTTP → 500 → indicates → an → internal → server → error
```

At each position, the model predicts a probability distribution over its vocabulary.

For the correct target token \(y_t\), the loss is:

\[
L_t=-\log P(y_t\mid x,y_{<t})
\]

The total supervised fine-tuning loss is commonly the average over the target tokens:

\[
L=-\frac{1}{N}\sum_{t=1}^{N}\log P(y_t\mid x,y_{<t})
\]

Where:

- \(x\) is the input prompt.
- \(y_t\) is the expected token.
- \(N\) is the number of target tokens contributing to the loss.

For prompt-completion training, we can mask the prompt tokens so that only the completion contributes to the loss.

This is useful because our objective is to teach the model how to respond to an instruction, not necessarily to reproduce the instruction itself.

The [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/main/sft_trainer?utm_source=chatgpt.com) describes prompt-completion training and completion-only loss. [Hugging Face](https://huggingface.co/docs/trl/main/sft_trainer?utm_source=chatgpt.com)

## Step 3: Update parameters

The model computes gradients:

\[
\frac{\partial L}{\partial W}
\]

An optimizer uses those gradients to update trainable parameters.

For ordinary full fine-tuning, this can involve updating billions of weights.

That's where our first major engineering problem appears.

**Memory consumption.**

---

# Part 3: Why Full Fine-Tuning Is Expensive

**Time: 20–25 minutes**

Imagine a model containing 7 billion parameters.

If its weights are stored in FP16, each parameter requires two bytes.

The weight memory alone is approximately:

\[
7,000,000,000\times2
\]

\[
=14,000,000,000\text{ bytes}
\]

That's approximately **14 GB in decimal units**, before accounting for anything else.

Training also requires memory for gradients, optimizer states, activations, and temporary computations.

For example, standard Adam optimization commonly maintains two additional values per trainable parameter.

Depending on the training implementation, master weights and other buffers may also be required.

Consequently, full fine-tuning a 7-billion-parameter model can require substantially more memory than simply loading its weights for inference.

This is one reason parameter-efficient fine-tuning methods are valuable.

## What is parameter-efficient fine-tuning?

Instead of updating every model parameter, we train a relatively small number of additional parameters.

The original pretrained weights remain frozen.

One popular technique is **Low-Rank Adaptation (LoRA)**.

---

# Part 4: Understanding LoRA Mathematically

**Time: 25–35 minutes**

This is the most important technical section today.

Let's begin with an ordinary neural-network layer.

A linear layer performs:

\[
y=Wx+b
\]

Where:

\[
W\in\mathbb{R}^{d_{out}\times d_{in}}
\]

Suppose:

\[
d_{in}=768
\]

\[
d_{out}=768
\]

The weight matrix contains:

\[
768\times768=589,824
\]

trainable weights.

With full fine-tuning, all these weights may be updated.

## LoRA's idea

Instead of modifying the original weight matrix \(W_0\), LoRA introduces a low-rank update.

\[
W'=W_0+\Delta W
\]

The update is represented as:

\[
\Delta W=BA
\]

Where:

\[
A\in\mathbb{R}^{r\times d_{in}}
\]

\[
B\in\mathbb{R}^{d_{out}\times r}
\]

And \(r\) is a relatively small rank.

The forward computation becomes:

\[
y=W_0x+\frac{\alpha}{r}BAx+b
\]

Here, \(\alpha\) controls the scaling of the adapter update.

### Example with rank 8

Suppose:

\[
r=8
\]

Then matrix A contains:

\[
8\times768=6,144
\]

parameters.

Matrix B contains:

\[
768\times8=6,144
\]

parameters.

Total:

\[
6,144+6,144=12,288
\]

Compare:

| Method | Trainable weights in this layer |
|---|---:|
| Full fine-tuning | 589,824 |
| LoRA, rank 8 | 12,288 |

LoRA trains approximately:

\[
\frac{12,288}{589,824}\times100
\approx2.08\%
\]

of the original weight count for this layer.

This is a simplified single-layer comparison; the trainable fraction of an entire model depends on which layers receive adapters.

## Why does this work?

The underlying hypothesis is that many useful task-specific changes to a large weight matrix can be approximated using a lower-rank update.

Instead of learning an arbitrary large matrix, we learn two smaller matrices.

The original pretrained knowledge remains represented by the frozen base weights.

The adapters learn modifications that help with the target task.

### An important limitation

LoRA doesn't guarantee the same quality as full fine-tuning.

Its effectiveness depends on the task, rank, targeted layers, training data, and optimization settings.

---

# Part 5: Implement LoRA From Scratch Using PyTorch

Let's implement the concept without relying on a specialized fine-tuning library.

This is a small educational example, so you can understand the mathematics before using it on a real LLM.

```python
import torch
from torch import nn

torch.manual_seed(42)


class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 2,
        alpha: float = 4.0
    ):
        super().__init__()

        self.base = nn.Linear(
            in_features,
            out_features,
            bias=False
        )

        # Freeze pretrained/base weights
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        # Trainable low-rank matrices
        self.A = nn.Parameter(
            torch.randn(rank, in_features) * 0.01
        )

        self.B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

        self.scaling = alpha / rank

    def forward(self, x):

        base_output = self.base(x)

        lora_output = (
            x @ self.A.T @ self.B.T
        )

        return (
            base_output
            + self.scaling * lora_output
        )


model = LoRALinear(
    in_features=4,
    out_features=3,
    rank=2
)

for name, parameter in model.named_parameters():

    print(
        name,
        parameter.shape,
        parameter.requires_grad
    )
```

Expected parameter information:

```text
A torch.Size([2, 4]) True
B torch.Size([3, 2]) True
base.weight torch.Size([3, 4]) False
```

Notice something important.

The original weight matrix is frozen.

Only A and B are trainable.

## Train the LoRA adapter

Let's create a small synthetic learning task.

```python
# Input data
X = torch.randn(100, 4)

# A synthetic target transformation
target_weights = torch.randn(4, 3)

y = X @ target_weights

optimizer = torch.optim.Adam(
    [
        model.A,
        model.B
    ],
    lr=0.05
)

criterion = nn.MSELoss()

for epoch in range(300):

    optimizer.zero_grad()

    prediction = model(X)

    loss = criterion(prediction, y)

    loss.backward()

    optimizer.step()

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch}, "
            f"Loss: {loss.item():.4f}"
        )
```

The loss should generally decrease as the adapter learns.

Because the target transformation is randomly generated and the adapter rank is restricted, the model may not perfectly reproduce it.

That's expected.

### Why is B initialized to zero?

Look at:

```python
self.B = nn.Parameter(
    torch.zeros(out_features, rank)
)
```

At initialization:

\[
BA=0
\]

Therefore, the adapter initially contributes nothing.

The layer begins with the same output as the frozen base layer.

During training, B receives gradients and begins learning.

A can then receive nonzero gradients as B moves away from zero.

### Check that base weights remain frozen

Add:

```python
print(
    "Base gradient:",
    model.base.weight.grad
)

print(
    "A gradient:",
    model.A.grad is not None
)

print(
    "B gradient:",
    model.B.grad is not None
)
```

Expected:

```text
Base gradient: None
A gradient: True
B gradient: True
```

This demonstrates the central mechanism behind LoRA.

**You are training an additional low-rank transformation without updating the original weight matrix.**

---

# Part 6: Fine-Tuning a Real LLM with Hugging Face

**Time: 35–45 minutes**

Now let's move from our educational PyTorch implementation to an actual language model.

We'll use:

- Transformers for model loading.
- Datasets for preparing examples.
- TRL for supervised fine-tuning.
- PEFT for LoRA adapters.

These libraries are maintained within the [Hugging Face ecosystem](https://huggingface.co/?utm_source=chatgpt.com).

## Step 1: Install dependencies

```bash
pip install -U torch transformers datasets trl peft accelerate
```

For the actual training experiment, an NVIDIA GPU with sufficient VRAM is recommended. CPU training may be impractically slow.

We'll use a small instruction-tuned model:

`Qwen/Qwen2.5-0.5B-Instruct`

This is a learning experiment, not a production model recommendation.

## Step 2: Prepare the dataset

Our use case is backend error classification.

```python
from datasets import Dataset

examples = [
    {
        "prompt": "Classify: PostgreSQL too many connections",
        "completion": "database_connection_error"
    },
    {
        "prompt": "Classify: Redis MISCONF snapshot failure",
        "completion": "cache_persistence_error"
    },
    {
        "prompt": "Classify: HTTP 401 Unauthorized",
        "completion": "authentication_error"
    },
    {
        "prompt": "Classify: HTTP 403 Forbidden",
        "completion": "authorization_error"
    },
    {
        "prompt": "Classify: Celery worker lost",
        "completion": "background_worker_error"
    },
    {
        "prompt": "Classify: Connection refused to PostgreSQL",
        "completion": "database_connection_error"
    },
    {
        "prompt": "Classify: JWT token expired",
        "completion": "authentication_error"
    },
    {
        "prompt": "Classify: Celery task exceeded time limit",
        "completion": "background_worker_error"
    }
]

dataset = Dataset.from_list(examples)

splits = dataset.train_test_split(
    test_size=0.25,
    seed=42
)

train_dataset = splits["train"]
eval_dataset = splits["test"]
```

This dataset is intentionally tiny.

It is useful for verifying that the training pipeline works, but it is not sufficient to establish useful model performance.

For a real project, we would collect a substantially larger and more representative dataset.

## Step 3: Configure LoRA

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)
```

Let's understand the configuration.

| Parameter | Meaning |
|---|---|
| `r=8` | Adapter rank |
| `lora_alpha=16` | Adapter scaling parameter |
| `lora_dropout=0.05` | Dropout applied within the LoRA path |
| `target_modules` | Layers receiving adapters |
| `bias="none"` | Do not train bias parameters |
| `task_type` | Causal language-model training |

We selected attention projection modules for this experiment.

Other configurations may target additional linear layers.

The [PEFT LoRA reference](https://huggingface.co/docs/peft/main/package_reference/lora?utm_source=chatgpt.com) documents the available configuration options, including adapter ranks and target modules. [Hugging Face](https://huggingface.co/docs/peft/main/package_reference/lora?utm_source=chatgpt.com)

## Step 4: Configure supervised fine-tuning

```python
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="./backend-error-adapter",
    max_length=256,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=20,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=10,
    completion_only_loss=True,
    save_strategy="no",
    report_to="none"
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config
)

trainer.train()

trainer.save_model(
    "./backend-error-adapter/final"
)
```

This example follows the current TRL pattern of combining `SFTTrainer`, `SFTConfig`, and `LoraConfig`. [Hugging Face](https://huggingface.co/docs/trl/v1.15.0/sft_trainer?utm_source=chatgpt.com)

The exact library versions and hardware environment can affect compatibility, so record your working dependency versions when you run the experiment.

### What happens internally?

```text
Pretrained LLM
      |
      v
Freeze Base Weights
      |
      v
Insert LoRA Adapters
      |
      v
Load Training Examples
      |
      v
Tokenization
      |
      v
Forward Propagation
      |
      v
Calculate Completion Loss
      |
      v
Backpropagation
      |
      v
Update Adapter Parameters
      |
      v
Save Trained Adapter
```

Notice that we are not training a language model from scratch.

We are adapting an existing model to a particular task.

## Step 5: Understand gradient accumulation

We configured:

```python
per_device_train_batch_size=1

gradient_accumulation_steps=4
```

This means the trainer accumulates gradients across four microbatches before performing an optimizer update.

For one device, the effective batch size is:

\[
1\times4=4
\]

For \(N\) data-parallel devices:

\[
EffectiveBatchSize=
MicrobatchSize\times AccumulationSteps\times N
\]

Gradient accumulation is useful when GPU memory is too limited for a larger microbatch.

However, it does not eliminate the memory needed to store model weights and activations.

---

# Part 7: What Is QLoRA?

**Time: 45–52 minutes**

We now understand LoRA.

But there's another challenge.

Even when the base model's weights are frozen, they still occupy memory.

For a large model, loading the frozen weights can consume substantial VRAM.

**QLoRA addresses this problem by combining quantized base-model weights with trainable low-rank adapters.**

## LoRA vs QLoRA

### LoRA

```text
Base Model Weights
   FP16 / BF16
        |
        v
    Frozen Base
        +
   Trainable LoRA
        |
        v
      Output
```

### QLoRA

```text
Quantized Base Model
     4-bit Storage
          |
          v
      Frozen Base
          +
    Trainable LoRA
          |
          v
        Output
```

The important distinction is that the base model is stored in a quantized representation.

The adapter parameters remain trainable at an appropriate numerical precision.

QLoRA commonly uses a 4-bit representation called NF4, or Normal Float 4.

The [Transformers bitsandbytes documentation](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes?utm_source=chatgpt.com) explains 4-bit loading and NF4 configuration. [Hugging Face](https://huggingface.co/docs/transformers/v4.46.3/en/quantization/bitsandbytes?utm_source=chatgpt.com)

## Simplified memory comparison

For a 7-billion-parameter model:

| Weight representation | Approximate raw weight storage |
|---|---:|
| FP32 | 28 GB |
| FP16 | 14 GB |
| 8-bit | 7 GB |
| 4-bit | 3.5 GB |

These are theoretical raw-weight sizes, not complete training-memory requirements.

Actual memory consumption includes quantization metadata, adapters, activations, temporary buffers, and other components.

## Python configuration

A typical 4-bit loading configuration looks like:

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    quantization_config=quantization_config,
    device_map="auto"
)
```

This example illustrates quantized model loading, not a complete QLoRA training script.

To train with QLoRA, we would combine quantized loading with PEFT adapters and a compatible training configuration.

Hardware compatibility matters. BF16 computation and bitsandbytes acceleration are not supported equally on every device.

### Interview insight

**Does QLoRA train the quantized base weights?**

In the standard QLoRA approach, no.

The quantized base weights remain frozen.

Gradients flow through the model's computations to update the trainable adapters.

---

# Part 8: Data Quality, Evaluation & Production Fine-Tuning

Before you deploy a fine-tuned model, consider a scenario.

Your model achieves:

```text
Training accuracy:   99%
Validation accuracy: 68%
```

What does this suggest?

Possibly overfitting.

The model may have learned patterns specific to its training examples without generalizing to unseen errors.

This is especially likely with very small datasets.

## Data preparation requirements

For a production fine-tuning project, your dataset should be:

- Representative of real user requests.
- Consistently labeled.
- Free from unnecessary secrets and personal information.
- Deduplicated where appropriate.
- Separated into training, validation, and test sets.
- Versioned so experiments are reproducible.

Avoid placing near-identical examples in both training and test datasets.

That can produce misleading evaluation results.

## Evaluation before and after fine-tuning

Suppose you're building an error classifier.

Compare the base model and fine-tuned model using the same held-out test dataset.

| Metric | Base model | Fine-tuned model |
|---|---|---|
| Classification accuracy | Measure | Measure |
| Macro F1 | Measure | Measure |
| Invalid output rate | Measure | Measure |
| Inference latency | Measure | Measure |
| Cost per request | Measure | Measure |

We should also compare against a simple deterministic classifier or conventional ML baseline.

For a fixed set of error categories, fine-tuning an LLM may not be the most efficient solution.

### Production deployment architecture

```text
                 TRAINING PIPELINE

Labeled Dataset
       |
       v
Data Validation
       |
       v
Train / Validation / Test Split
       |
       v
Pretrained Model
       |
       v
LoRA / QLoRA Training
       |
       v
Model Evaluation
       |
       v
Adapter Artifact
       |
       v
Model Registry
       |
       v
Deployment Approval


                  INFERENCE PIPELINE

Client
   |
   v
FastAPI
   |
   v
Input Validation
   |
   v
Model Serving
   |
   v
Base Model + Adapter
   |
   v
Output Validation
   |
   v
Response
```

For production, you should version the base model and adapter together.

A LoRA adapter is not generally a complete standalone language model.

It depends on the compatible base-model architecture and weights.

---

# Part 9: Today's Practical Assignment

**Time: 52–57 minutes**

## Mini-Project: Fine-Tune a Backend Error Classifier

Your assignment is to create a small dataset and evaluate whether fine-tuning improves error classification.

### Task 1: Prepare the dataset

Create at least 50 examples across these categories:

| Category | Example |
|---|---|
| Database | PostgreSQL connection refused |
| Authentication | JWT token expired |
| Authorization | Permission denied |
| Cache | Redis MISCONF |
| Background worker | Celery worker unavailable |
| Network | Connection timeout |
| Application | Unhandled Python exception |

Keep 10 additional examples as a held-out test set.

These numbers are for a learning exercise; a production dataset will usually require broader coverage.

### Task 2: Run a baseline

Before fine-tuning, evaluate the pretrained model using the test examples.

Record accuracy, invalid output rate, and common mistakes.

### Task 3: Fine-tune using LoRA

Use the small-model training example from today's session.

Start with:

```python
r = 8
lora_alpha = 16
learning_rate = 2e-4
```

Treat these as initial experimental values rather than universally optimal settings.

### Task 4: Compare results

Evaluate the fine-tuned model on the same held-out test set.

Record whether accuracy improved, whether the model overfits, and whether it produces valid category labels.

### Task 5: Explain the engineering decision

Write a short technical justification answering:

> Why did we choose LoRA rather than full fine-tuning, and why might a traditional classifier still be preferable for this task?

**If you don't have a suitable GPU:** Complete the PyTorch LoRA implementation and dataset preparation first. You can run the LLM training portion later on appropriate hardware.

---

# Part 10: Senior AI Engineer Interview Questions

**Time: 57–60 minutes**

### Q1. What is the difference between pretraining and fine-tuning?

Pretraining teaches a model broad patterns from large datasets.

Fine-tuning adapts an existing model to a more specific task or desired behavior.

### Q2. What is the difference between RAG and fine-tuning?

RAG supplies external information during inference.

Fine-tuning updates model parameters through training.

They solve different problems and can be combined.

### Q3. Why is LoRA more memory-efficient than full fine-tuning?

LoRA freezes the original weights and trains relatively small low-rank adapter matrices.

This reduces the number of trainable parameters and associated optimizer-state requirements.

### Q4. What is the difference between LoRA and QLoRA?

LoRA uses trainable low-rank adapters.

QLoRA combines low-rank adapters with a quantized, frozen base model to reduce memory requirements further.

### Q5. Does a lower training loss guarantee better model quality?

No.

The model might overfit the training dataset.

You need held-out evaluation data and task-specific quality metrics.

### Advanced interview question

Your company wants to fine-tune a 7-billion-parameter model on 10,000 internal support conversations.

How would you design the training pipeline?

A strong answer should discuss data permissions, sensitive-data removal, train/validation/test splits, baseline evaluation, hardware requirements, LoRA or QLoRA selection, experiment tracking, model versioning, deployment, and rollback.

---

# 🧠 Knowledge Quiz

Try answering these questions without reviewing the lesson.

**Q1.** Suppose a linear layer has 1,024 input features and 1,024 output features.

How many weights does the original matrix contain?

**Q2.** If we apply LoRA with rank 16 to that layer, how many trainable adapter parameters are introduced?

**Q3.** Why do we freeze the original weight matrix during LoRA training?

**Q4.** What is the purpose of `gradient_accumulation_steps`?

**Q5.** Why might fine-tuning be a poor solution for answering questions about frequently updated company documents?

**Q6.** Your fine-tuned model performs well on training data but poorly on unseen examples. What would you investigate?

**Q7 — System Design:** You have an existing FastAPI RAG application. Users report inconsistent JSON formatting, incorrect internal policy answers, and slow inference. Which problems would you address through prompting, retrieval, fine-tuning, or serving optimization?

Explain why you would choose each approach.

---

# Session 8 Recap

Today, we moved from application-level AI engineering into model-level customization.

You learned how supervised fine-tuning uses token-level loss and gradient updates, why full fine-tuning is resource-intensive, and how LoRA introduces small trainable matrices while keeping the pretrained base frozen.

We also examined QLoRA, dataset preparation, evaluation, and production deployment considerations.

The most important mathematical relationship is:

\[
\boxed{W'=W_0+\frac{\alpha}{r}BA}
\]

This equation captures the core idea of LoRA.

The most important engineering principle is:

**Fine-tuning should solve a clearly measured model-behavior problem. It should not be the default response to every limitation in an AI application.**

---

## 🔜 Session 9: LLM Training Pipelines, Experiment Tracking & MLOps

Next, we'll continue from today's fine-tuning project.

We'll explore how to build reproducible training pipelines, track datasets and experiments, compare model versions, manage artifacts, and deploy models safely.

Our practical project will introduce MLflow-style experiment tracking and a model evaluation pipeline, connecting your existing backend and Docker experience with production MLOps.

**Today's priority:** Understand LoRA's matrix operations, implement the PyTorch example, and prepare a clean dataset for the backend error-classification experiment.