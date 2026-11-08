# 🎓 AI Engineer Mentorship — Session 17
## LLM Fine-Tuning: LoRA, QLoRA, PEFT & Training Custom Models

**Level:** Intermediate → Advanced  
**Duration:** 60 minutes  
**Career goal:** Python Backend Developer → Senior AI Engineer  
**Today's project:** Fine-tune a small language model to classify software incidents and generate structured JSON responses.

In Session 16, we learned how to deploy pretrained LLMs using vLLM, Docker, and Kubernetes. We explored GPU memory, KV caching, continuous batching, and model-serving architecture.

Today, we'll answer a different question:

**What happens when a pretrained LLM doesn't perform your specific task well enough?**

Imagine you're building an AI assistant for a software company.

You want it to analyze production incidents and return:

```json
{
  "priority": "P1",
  "team": "backend",
  "reason": "API response times have increased significantly."
}
```

A general-purpose model might return inconsistent JSON, incorrect categories, or unnecessary explanations.

You could improve the prompt. But if prompting alone doesn't achieve your quality targets, you might consider fine-tuning.

Today, we'll learn when fine-tuning is appropriate, how it works, and how to implement it using Python.

---

# ⏱️ Your 60-Minute Learning Plan

| Time | Topic | Learning outcome |
|---|---|---|
| 0–10 min | Prompting vs RAG vs fine-tuning | Choose the right approach |
| 10–20 min | Supervised fine-tuning and training fundamentals | Understand how models learn |
| 20–30 min | LoRA, QLoRA, and PEFT | Understand efficient adaptation |
| 30–45 min | Practical Python fine-tuning project | Train a LoRA adapter |
| 45–52 min | Evaluation and overfitting | Measure improvement |
| 52–57 min | Production fine-tuning architecture | Think like a Senior AI Engineer |
| 57–60 min | Exercise and interview quiz | Test your understanding |

## Learning objectives

By the end of this session, you should be able to explain supervised fine-tuning, LoRA, QLoRA, adapter weights, rank, learning rate, gradient accumulation, training loss, and validation loss.

You should also be able to prepare an instruction-tuning dataset, train a small LoRA adapter, and decide whether fine-tuning is actually necessary for a business problem.

---

# Part 1: Prompting vs RAG vs Fine-Tuning
**Time: 0–10 minutes**

This is one of the most important architectural decisions in AI engineering.

Suppose you're building an AI assistant for your company's project management application.

You have three requirements.

### Requirement A: Follow a particular response format

You want the model to always respond with:

```json
{
  "issue_type": "bug",
  "priority": "high",
  "assigned_team": "backend"
}
```

The first solution should usually be prompting with structured-output constraints and application-level validation.

If that doesn't meet your requirements, fine-tuning may help improve consistency.

### Requirement B: Answer questions about company documents

A user asks:

> What is our company's latest leave policy?

The information is stored in a PDF uploaded yesterday.

Would fine-tuning be the best solution?

Usually not.

**RAG is generally more appropriate** because the information changes and needs to be retrieved from an authoritative source.

### Requirement C: Learn a specialized classification task

Your company has thousands of historical support tickets with carefully reviewed classifications.

You want a model that consistently maps new tickets to your company's incident taxonomy.

Fine-tuning may be useful here.

### Comparison

| Feature | Prompting | RAG | Fine-tuning |
|---|---|---|---|
| Changes model weights | No | No | Yes |
| Uses external knowledge at request time | Optional | Yes | Optional |
| Suitable for changing documents | Limited | Excellent | Usually poor |
| Can improve task-specific behavior | Yes | Sometimes | Yes |
| Requires training data | No | Documents | Yes |
| Requires model training | No | No | Yes |
| Typical complexity | Low | Medium | Medium–High |

**Important:** These approaches are not mutually exclusive.

A production system can use all three:

```text
User Question
     |
     v
Prompt Template
     |
     v
RAG Retrieval
     |
     v
Fine-Tuned LLM
     |
     v
Validated Response
```

### Senior AI Engineer decision rule

Before fine-tuning, establish a baseline using prompting, structured outputs, and RAG where appropriate.

Fine-tune only when you have a measurable problem that training is likely to improve.

Fine-tuning is not a reliable way to keep frequently changing factual information up to date.

---

# Part 2: What Is Supervised Fine-Tuning?
**Time: 10–20 minutes**

Let's connect fine-tuning to the machine learning foundations we've studied.

## 2.1 Pretraining vs fine-tuning

During pretraining, a language model learns statistical patterns from large amounts of text.

For an autoregressive language model, the training objective commonly involves predicting the next token.

Consider:

```text
Input: Python is a programming
Target: language
```

The model produces probabilities for possible next tokens.

For example:

```python
{
    "language": 0.85,
    "framework": 0.08,
    "database": 0.03
}
```

These numbers are illustrative.

The training process compares the model's prediction with the expected token and calculates a loss.

The optimizer uses gradients to update trainable parameters.

### Fine-tuning

Fine-tuning continues training an existing model on a more specific dataset.

For example:

```text
User:
Our payment API is returning HTTP 500.

Assistant:
{"priority": "P1", "team": "backend"}
```

Repeated high-quality examples can help the model learn the desired classification and response style.

## 2.2 What is a training example?

For supervised instruction fine-tuning, one example might contain:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You classify software incidents."
    },
    {
      "role": "user",
      "content": "Our production database is unavailable."
    },
    {
      "role": "assistant",
      "content": "{\"priority\":\"P0\",\"team\":\"infrastructure\"}"
    }
  ]
}
```

The assistant response is the desired output.

During training, the model is optimized to assign higher probability to the expected output tokens.

## 2.3 Understanding training loss

A common language-model training objective is cross-entropy loss.

For a target token:

\[
L=-\log(P(\text{correct token}))
\]

Suppose the correct token is `"P0"`.

If the model assigns:

\[
P(P0)=0.1
\]

The loss is:

\[
L=-\log(0.1)\approx2.303
\]

After training, suppose:

\[
P(P0)=0.9
\]

Then:

\[
L=-\log(0.9)\approx0.105
\]

The loss is lower because the model assigns higher probability to the correct token.

In actual language-model training, loss is calculated across many target tokens and aggregated according to the training configuration.

### What happens during training?

```text
Training Example
       |
       v
   Tokenization
       |
       v
   Forward Pass
       |
       v
   Prediction
       |
       v
   Loss Calculation
       |
       v
   Backpropagation
       |
       v
   Optimizer Step
       |
       v
 Updated Trainable Weights
```

As a backend developer, you can think of this as a repeated optimization process rather than a normal request-response workflow.

The difference is that training modifies model parameters.

Inference doesn't normally modify them.

---

# Part 3: Full Fine-Tuning vs LoRA vs QLoRA
**Time: 20–30 minutes**

Now we reach today's most important technical concept.

## 3.1 The problem with full fine-tuning

Suppose you have a model with seven billion parameters.

If you perform full fine-tuning, all those parameters may receive gradient updates.

Training also requires memory for gradients, optimizer states, activations, and other runtime data.

This can require substantially more memory than inference.

For example, the model's FP16 weights alone occupy approximately 14 GB.

But full fine-tuning requires additional memory beyond those weights.

This makes full fine-tuning expensive for many developers.

## 3.2 What is PEFT?

**PEFT stands for Parameter-Efficient Fine-Tuning.**

Instead of updating every parameter, PEFT methods train a relatively small subset of parameters.

One popular PEFT method is LoRA.

The [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/index?utm_source=chatgpt.com) explains the supported parameter-efficient adaptation techniques.

## 3.3 What is LoRA?

**LoRA stands for Low-Rank Adaptation.**

Imagine a pretrained model contains a weight matrix:

\[
W
\]

During full fine-tuning, we update the original matrix.

LoRA takes a different approach.

It keeps the original weight matrix frozen and learns a low-rank update.

\[
\boxed{W'=W+\Delta W}
\]

Where:

\[
\Delta W=BA
\]

Here:

- \(W\) is the original frozen weight matrix.
- \(A\) and \(B\) are smaller trainable matrices.
- \(\Delta W\) represents the learned adaptation.

### Example

Suppose the original matrix has dimensions:

\[
4096\times4096
\]

Total parameters:

\[
4096\times4096=16,777,216
\]

Now suppose LoRA uses rank:

\[
r=8
\]

The two adapter matrices have dimensions:

\[
A:8\times4096
\]

\[
B:4096\times8
\]

Total LoRA parameters:

\[
(8\times4096)+(4096\times8)
\]

\[
=65,536
\]

Compare:

| Approach | Trainable parameters for this matrix |
|---|---:|
| Full fine-tuning | 16,777,216 |
| LoRA, rank 8 | 65,536 |

The LoRA update uses approximately **0.39% as many trainable parameters** for this matrix.

This is a simplified illustration. Actual model-wide savings depend on which layers receive adapters.

### Python demonstration

```python
input_dim = 4096
output_dim = 4096
rank = 8

full_parameters = input_dim * output_dim

lora_parameters = (
    input_dim * rank
    + rank * output_dim
)

percentage = (
    lora_parameters / full_parameters
) * 100

print("Full parameters:", full_parameters)
print("LoRA parameters:", lora_parameters)
print(f"Percentage: {percentage:.2f}%")
```

Output:

```text
Full parameters: 16777216
LoRA parameters: 65536
Percentage: 0.39%
```

### What does rank mean?

Rank controls the capacity of the learned low-rank update.

A larger rank allows the adapter to represent a wider range of changes but increases trainable parameters and memory requirements.

A smaller rank reduces resource usage but may limit adaptation capacity.

**Interview question:** Does increasing LoRA rank always improve model quality?

No. A larger rank increases capacity, but the improvement depends on the task, dataset, optimization, and evaluation results. It can also increase overfitting risk.

Official reference: [LoRA configuration and implementation](https://huggingface.co/docs/peft/main/package_reference/lora?utm_source=chatgpt.com). [Hugging Face](https://huggingface.co/docs/peft/main/package_reference/lora?utm_source=chatgpt.com)

---

## 3.4 What is QLoRA?

QLoRA combines low-rank adaptation with a quantized base model.

A typical QLoRA setup uses:

- A frozen base model stored in 4-bit precision.
- Trainable LoRA adapters.
- Higher-precision computation for relevant training operations.
- Memory-efficient optimization techniques.

Conceptually:

```text
               Pretrained LLM
                     |
                     v
              4-bit Quantization
                     |
                     v
             Frozen Base Weights
                     |
                     +
                     |
              LoRA Adapters
                     |
                     v
             Train Adapters Only
```

### LoRA vs QLoRA

| Feature | LoRA | QLoRA |
|---|---|---|
| Base weights frozen | Yes | Yes |
| Trainable adapters | Yes | Yes |
| Base model quantized | Not required | Typically 4-bit |
| Training memory | Reduced | Often further reduced |
| Implementation complexity | Lower | Higher |
| Quantization-related constraints | Fewer | More |

QLoRA is useful when GPU memory is limited.

However, 4-bit quantization does not mean the entire training process uses only four bits per parameter. Activations, adapters, optimizer states, and other components require additional memory.

The official [PEFT quantization guide](https://huggingface.co/docs/peft/en/developer_guides/quantization?utm_source=chatgpt.com) explains the integration of quantization and LoRA adapters. [Hugging Face](https://huggingface.co/docs/peft/en/developer_guides/quantization?utm_source=chatgpt.com)

---

# Part 4: Practical Project — Fine-Tune an Incident Classification Model
**Time: 30–45 minutes**

Let's build something relevant to your backend engineering experience.

Our goal is to train a model that classifies production incidents.

### Input

```text
Our PostgreSQL database is unavailable.
```

### Desired output

```json
{
  "priority": "P0",
  "team": "infrastructure"
}
```

We'll use:

- Python.
- PyTorch.
- Transformers.
- TRL.
- PEFT.
- A small Qwen model.

We'll start with LoRA rather than QLoRA because it introduces fewer moving parts.

A CUDA GPU is recommended for the training portion. Dataset preparation and validation can be completed without one.

## Step 1: Install dependencies

```bash
pip install -U torch transformers datasets trl peft accelerate
```

For reproducible production work, record the exact resolved package versions in a lockfile.

We'll use:

`Qwen/Qwen2.5-0.5B-Instruct`

This is a small instruction-tuned model suitable for learning the workflow, though its performance will not represent that of a larger production model.

The [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?utm_source=chatgpt.com) provides model information and usage examples.

---

## Step 2: Prepare the training dataset

Create `train.py`.

```python
import json

from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = """
You are a software incident classifier.

Return a JSON object containing exactly:
- priority: P0, P1, or P2
- team: backend, infrastructure, or security

Do not include explanations outside the JSON.
""".strip()


training_examples = [
    (
        "Production PostgreSQL is completely unavailable.",
        "P0",
        "infrastructure"
    ),
    (
        "The main Redis cluster has stopped responding.",
        "P0",
        "infrastructure"
    ),
    (
        "Our payment API returns HTTP 500 for many users.",
        "P1",
        "backend"
    ),
    (
        "The login endpoint has become significantly slower.",
        "P1",
        "backend"
    ),
    (
        "A minor validation error appears in the profile API.",
        "P2",
        "backend"
    ),
    (
        "A noncritical background job is running slowly.",
        "P2",
        "backend"
    ),
    (
        "An unauthorized user accessed confidential records.",
        "P0",
        "security"
    ),
    (
        "Suspicious login attempts are increasing rapidly.",
        "P1",
        "security"
    ),
    (
        "A development API key was accidentally exposed.",
        "P1",
        "security"
    ),
    (
        "CPU usage is elevated on one noncritical worker.",
        "P2",
        "infrastructure"
    )
]


validation_examples = [
    (
        "Our primary database is offline.",
        "P0",
        "infrastructure"
    ),
    (
        "The API returns occasional internal server errors.",
        "P1",
        "backend"
    )
]


def create_example(text, priority, team):

    return {
        "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": json.dumps({
                    "priority": priority,
                    "team": team
                })
            }
        ]
    }


train_dataset = Dataset.from_list([
    create_example(*example)
    for example in training_examples
])

eval_dataset = Dataset.from_list([
    create_example(*example)
    for example in validation_examples
])


print("Training samples:", len(train_dataset))
print("Validation samples:", len(eval_dataset))
```

### Explanation

Notice the dataset contains two fields:

```python
{
    "prompt": [...],
    "completion": [...]
}
```

This is a conversational prompt-completion dataset.

The prompt contains the instructions and user request.

The completion contains the expected assistant response.

This structure allows us to train specifically on the expected completion tokens.

The [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer?utm_source=chatgpt.com) supports conversational prompt-completion datasets and completion-only loss. [Hugging Face](https://huggingface.co/docs/trl/sft_trainer?utm_source=chatgpt.com)

### Important limitation

Ten training examples are nowhere near sufficient for a serious production classifier.

This is a demonstration dataset.

For a real project, you would need a larger, representative, consistently labeled dataset with sufficient examples for each class and a separate test set.

You should also define clear incident severity rules before collecting labels.

---

## Step 3: Configure LoRA

Add the following to `train.py`:

```python
from peft import LoraConfig


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM"
)
```

Let's understand each parameter.

### `r=8`

This is the LoRA rank.

It controls the size of the low-rank adaptation matrices.

### `lora_alpha=16`

This controls the scaling of the LoRA update.

With the conventional LoRA scaling rule:

\[
Scaling=\frac{\alpha}{r}
\]

In our case:

\[
Scaling=\frac{16}{8}=2
\]

### `lora_dropout=0.05`

Applies dropout in the LoRA adaptation path during training.

It may help regularization.

### `target_modules="all-linear"`

Applies LoRA to supported linear layers in the model, excluding certain output layers according to PEFT's implementation.

This is convenient because different model architectures use different internal layer names.

### `task_type="CAUSAL_LM"`

Specifies that we're adapting a causal language model.

---

## Step 4: Configure supervised fine-tuning

Now add:

```python
from trl import SFTConfig, SFTTrainer


training_args = SFTConfig(
    output_dir="./incident-lora",

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    max_length=256,

    completion_only_loss=True,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_steps=1,

    report_to="none",

    seed=42
)
```

### Understanding the important settings

**`num_train_epochs=3`**

The model makes three passes through the training dataset.

**`per_device_train_batch_size=1`**

One training example is processed per device in each microbatch.

**`gradient_accumulation_steps=4`**

Gradients are accumulated across four microbatches before an optimizer update.

With one device:

\[
EffectiveBatchSize=1\times4=4
\]

This is the effective batch size for optimizer updates, assuming one device.

**`learning_rate=2e-4`**

Controls the optimizer's update step size.

Adapter training often uses a higher learning rate than full-model fine-tuning, but the appropriate value must be validated experimentally.

**`max_length=256`**

Limits the tokenized training sequence length.

Longer examples may be truncated, so verify that your target completions remain intact.

**`completion_only_loss=True`**

Trains on the completion tokens rather than calculating loss across the entire prompt.

This is useful because we want the model to learn the expected assistant output.

---

## Step 5: Train the LoRA adapter

Add:

```python
trainer = SFTTrainer(
    model=MODEL_NAME,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config
)

trainer.train()

trainer.save_model("./incident-lora-final")
```

Run:

```bash
python train.py
```

The trainer will load the base model, attach LoRA adapters, tokenize the dataset, and begin training.

You should see training logs containing values such as loss, learning rate, and epoch progress.

### What is actually being saved?

Because we're using PEFT, the saved model artifacts primarily contain the trained adapter configuration and adapter weights rather than a complete independent copy of all the base model weights.

The base model is still required when loading the adapter for inference.

Conceptually:

```text
Qwen Base Model
       |
       +
       |
Trained LoRA Adapter
       |
       v
Adapted Incident Classifier
```

This is one reason LoRA can be useful for maintaining multiple task-specific adaptations of the same base model.

---

# Part 5: Run Inference with the Fine-Tuned Model
**Time: 45–48 minutes**

Create `inference.py`.

```python
import json

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./incident-lora-final"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()


messages = [
    {
        "role": "system",
        "content": (
            "Classify the software incident. "
            "Return only JSON with priority and team."
        )
    },
    {
        "role": "user",
        "content": (
            "Our production database has stopped responding."
        )
    }
]


inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
)

inputs = inputs.to(model.device)


with torch.inference_mode():

    outputs = model.generate(
        inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )


generated_tokens = outputs[0][
    inputs.shape[-1]:
]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print(response)

try:
    parsed = json.loads(response)
    print("Valid JSON:", parsed)

except json.JSONDecodeError:
    print("The model returned invalid JSON")
```

### Expected behavior

We want the model to return something like:

```json
{
  "priority": "P0",
  "team": "infrastructure"
}
```

But this is a target response, not a guaranteed result.

The small training dataset may not produce reliable behavior.

Even after fine-tuning, production applications should validate generated JSON and enforce allowed field values.

Fine-tuning improves statistical behavior; it does not create a deterministic guarantee.

---

# Part 6: Understanding QLoRA in Python
**Time: 48–52 minutes**

Let's look at how the configuration changes when we use QLoRA.

Instead of loading the base model in ordinary floating-point precision, we configure 4-bit quantization.

```python
import torch

from transformers import BitsAndBytesConfig


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

Then, with a compatible TRL version, the trainer can be configured using:

```python
trainer = SFTTrainer(
    model=MODEL_NAME,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    quantization_config=quantization_config
)
```

This replaces the non-quantized trainer construction in the earlier example.

It requires a supported hardware and software configuration, including an appropriate `bitsandbytes` installation.

The current TRL documentation supports passing a quantization configuration together with a PEFT configuration for QLoRA training. [Hugging Face](https://huggingface.co/docs/trl/sft_trainer?utm_source=chatgpt.com)

### What is NF4?

NF4 stands for NormalFloat4.

It is a 4-bit quantization format designed for certain weight distributions.

### What is double quantization?

Double quantization reduces memory consumption by quantizing some of the quantization metadata itself.

### When should you choose QLoRA?

Suppose you want to adapt a 7B model but have limited GPU memory.

QLoRA may make training feasible where a higher-precision LoRA setup would not fit.

However, compatibility, model architecture, sequence length, batch size, and GPU capabilities all matter.

**Interview insight:** QLoRA primarily reduces the memory required for storing the frozen base weights. It doesn't eliminate the memory needed for activations, adapters, and training operations.

---

# Part 7: How Do We Know Fine-Tuning Worked?
**Time: 52–55 minutes**

This is where your previous lessons on evaluation become important.

A lower training loss doesn't automatically mean a better production model.

Suppose your results are:

| Epoch | Training loss | Validation loss |
|---|---:|---:|
| 1 | 1.8 | 1.9 |
| 2 | 1.1 | 1.4 |
| 3 | 0.5 | 1.7 |

These are hypothetical values.

Notice that training loss continues decreasing, but validation loss starts increasing.

This may indicate overfitting.

The model is learning the training examples without generalizing as well to unseen data.

## What should we measure?

For our incident classification project, useful metrics include:

| Metric | Purpose |
|---|---|
| Priority accuracy | Correct severity classification |
| Team accuracy | Correct team assignment |
| Exact-match accuracy | Both fields correct |
| Macro F1 | Performance across classes |
| JSON validity rate | Syntactically valid output |
| Schema compliance | Correct keys and allowed values |
| Inference latency | Production responsiveness |
| Cost per request | Financial efficiency |

### Python evaluation example

```python
def exact_match_accuracy(predictions, expected):

    correct = sum(
        prediction == target
        for prediction, target in zip(
            predictions,
            expected
        )
    )

    return correct / len(expected)


predictions = [
    {"priority": "P0", "team": "infrastructure"},
    {"priority": "P1", "team": "backend"},
    {"priority": "P2", "team": "backend"}
]

expected = [
    {"priority": "P0", "team": "infrastructure"},
    {"priority": "P1", "team": "backend"},
    {"priority": "P1", "team": "backend"}
]

accuracy = exact_match_accuracy(
    predictions,
    expected
)

print(f"Accuracy: {accuracy:.2%}")
```

Output:

```text
Accuracy: 66.67%
```

### Important production principle

Always compare:

**Base model + baseline prompt**

against:

**Fine-tuned model + equivalent evaluation conditions**

Use the same held-out test cases and comparable decoding settings.

Otherwise, you cannot confidently attribute improvements to fine-tuning.

Also, keep the final test set separate from the training and validation data.

---

# Part 8: Senior AI Engineer System Design
**Time: 55–57 minutes**

Let's design a production fine-tuning platform.

Imagine your company has 100,000 historical support tickets.

You want to create and deploy a specialized classification model.

A production architecture might look like:

```text
                 Historical Tickets
                         |
                         v
                  Data Extraction
                         |
                         v
                Data Quality Checks
                         |
                         v
                Sensitive Data Removal
                         |
                         v
                 Dataset Versioning
                         |
                         v
               Train / Validation Split
                         |
                         v
                 Fine-Tuning Pipeline
                         |
                         v
                  LoRA / QLoRA
                         |
                         v
                 Evaluation Pipeline
                         |
                ┌────────┴────────┐
                |                 |
                v                 v
           Quality Pass       Quality Fail
                |                 |
                v                 v
           Model Registry      Reject Model
                |
                v
           Staging Deployment
                |
                v
           Production Canary
                |
                v
           Monitoring & Feedback
```

### What must a Senior AI Engineer consider?

**Data quality:** Poor labels can teach the model incorrect behavior. Review class definitions, duplicate records, conflicting labels, and data leakage.

**Security:** Training datasets may contain passwords, API keys, customer information, or confidential documents. Apply appropriate redaction, authorization, and retention controls.

**Reproducibility:** Record the base model revision, dataset version, training configuration, random seed, library versions, and adapter artifact.

**Evaluation:** Measure quality on unseen examples. Include rare but important cases, such as security incidents and critical production outages.

**Deployment:** Release the adapter to staging first. Compare it against the current production model before increasing traffic.

**Rollback:** Preserve the previous adapter and base-model configuration so a bad deployment can be reversed.

### A useful production artifact

Each training run should produce a record resembling:

```json
{
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "dataset_version": "incidents-v3",
  "adapter_version": "incident-lora-v2",
  "training_method": "LoRA",
  "rank": 8,
  "learning_rate": 0.0002,
  "epochs": 3,
  "evaluation_status": "pending"
}
```

The `evaluation_status` should only change to a passing state after the model meets predefined quality requirements.

This connects fine-tuning directly to MLOps.

---

# 🛠️ Today's Practical Assignment

## Mini-Project: Train and Evaluate a Custom Incident Classifier

Build on the Python project from today's lesson.

Your goal is to create an incident classification model that outputs:

```json
{
  "priority": "P1",
  "team": "backend"
}
```

Complete these tasks:

1. Expand the demonstration dataset to at least 30 carefully labeled examples, including examples for every priority and team category.
2. Create separate training, validation, and test sets. Avoid placing duplicate or near-duplicate incidents across splits.
3. Evaluate the pretrained model before fine-tuning.
4. Train a LoRA adapter using the provided code if you have suitable hardware.
5. Evaluate the adapted model on the same held-out test set and compare exact-match accuracy, JSON validity, and latency.
6. Save the training configuration and evaluation results alongside the adapter.

**Bonus:** Repeat the experiment using two LoRA ranks, such as 8 and 16. Compare trainable parameter counts, memory usage, and test performance.

If you don't have a compatible GPU, complete the dataset, validation, and baseline-evaluation portions first. Those are meaningful AI engineering tasks independently of GPU training.

### Expected deliverables

```text
incident-finetuning/
    |
    ├── train.py
    ├── inference.py
    ├── evaluate.py
    ├── datasets/
    │   ├── train.jsonl
    │   ├── validation.jsonl
    │   └── test.jsonl
    |
    ├── experiments/
    │   └── results.json
    |
    └── adapters/
        └── incident-lora/
```

You can implement the missing evaluation script as part of the exercise.

---

# 🎯 Senior AI Engineer Interview Questions

### Q1. What is the difference between RAG and fine-tuning?

RAG retrieves external information and provides it to the model at inference time.

Fine-tuning modifies trainable model parameters using task-specific training data.

RAG is usually preferable for frequently changing knowledge, while fine-tuning may improve specialized behavior or task performance.

### Q2. What is LoRA?

LoRA is a parameter-efficient fine-tuning technique that freezes the pretrained weights and learns low-rank updates through smaller trainable matrices.

### Q3. What is the difference between LoRA and QLoRA?

LoRA trains low-rank adapters while keeping the base model frozen.

QLoRA typically uses a 4-bit quantized frozen base model to reduce memory requirements further.

### Q4. What is the purpose of LoRA rank?

Rank controls the dimensionality and capacity of the low-rank update.

Higher rank increases the number of trainable parameters and may allow more expressive adaptations.

### Q5. What is gradient accumulation?

Gradient accumulation combines gradients from multiple microbatches before performing an optimizer step.

It allows a larger effective batch size without requiring all examples to fit into memory simultaneously.

### Q6. Why can validation loss increase while training loss decreases?

This may indicate overfitting.

The model is becoming better at predicting training examples but worse at generalizing to unseen validation examples.

### Q7. When should you avoid fine-tuning?

Avoid fine-tuning when simpler techniques already meet your requirements, when you lack reliable training data, or when the main problem is access to frequently changing information.

### Q8. Can LoRA adapters be reused across different base models?

Not arbitrarily.

An adapter is trained against a particular base model architecture and parameter configuration. Compatibility must be verified, and the correct base model revision should be preserved.

---

# 🧠 Session 17 Quiz

Try answering these questions before checking the answers.

**Question 1:** What is the difference between pretraining and supervised fine-tuning?

**Question 2:** Why does LoRA require fewer trainable parameters than full fine-tuning?

**Question 3:** A weight matrix has dimensions `2048 × 2048`. If LoRA rank is 16, how many trainable parameters are required for its two low-rank matrices?

**Question 4:** What is the difference between LoRA and QLoRA?

**Question 5:** Suppose:

```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
```

What is the effective batch size when training on one GPU?

**Question 6:** Your training loss keeps decreasing, but validation loss increases. What might be happening?

**Question 7 — System Design:**

Your company wants to customize an LLM for its internal project management application.

The application has 500,000 historical issues containing descriptions, priorities, assigned teams, and resolution details.

Explain how you would prepare the dataset, choose between prompting and fine-tuning, select LoRA or QLoRA, evaluate the model, deploy the adapter, and monitor quality after deployment.

## Quick answer key

| Question | Answer |
|---|---|
| 1 | Pretraining learns broad patterns; SFT adapts the model using supervised examples. |
| 2 | LoRA trains smaller low-rank matrices instead of updating the full weight matrix. |
| 3 | \(2048\times16+16\times2048=65,536\) parameters. |
| 4 | QLoRA adds quantization of the frozen base model. |
| 5 | 16 examples per optimizer update. |
| 6 | Possible overfitting; investigate validation data and training configuration. |

For Question 7, a strong answer should prioritize data quality, baseline evaluation, reproducibility, security, and controlled deployment rather than immediately choosing the largest model.

---

# Session 17 Recap

Today, you learned how pretrained language models can be adapted for specialized tasks.

The main concepts were:

| Concept | Key idea |
|---|---|
| Supervised fine-tuning | Train on examples with desired outputs |
| Full fine-tuning | Update the model's original parameters |
| PEFT | Train a smaller subset of parameters |
| LoRA | Learn low-rank weight updates |
| QLoRA | Train adapters over a quantized frozen base |
| Rank | Controls adapter capacity |
| Gradient accumulation | Increases effective batch size |
| Validation loss | Helps assess generalization |
| Model evaluation | Determines whether fine-tuning improved results |

The most important architectural lesson is:

\[
\boxed{\text{Fine-tuning is an optimization choice, not a default requirement.}}
\]

As a Senior AI Engineer, your responsibility is not simply to train a model. It's to determine whether training produces enough measurable value to justify the additional complexity, infrastructure, and maintenance costs.

---

## 🔜 Session 18: MLOps — Experiment Tracking, Dataset Versioning, Model Registry & CI/CD

Our next session will connect today's fine-tuning work to production machine learning operations.

We'll cover experiment tracking, dataset lineage, model versioning, reproducible training pipelines, model registries, automated evaluation gates, and deployment workflows.

We'll also explore [MLflow](https://mlflow.org/?utm_source=chatgpt.com) and build a practical Python experiment-tracking workflow that records training configurations, metrics, and model artifacts.

**Your priority for today:** Understand the LoRA equation \(W'=W+BA\), prepare the incident classification dataset, and study how `SFTTrainer` trains and saves a LoRA adapter. These concepts will be essential when we move into production MLOps.