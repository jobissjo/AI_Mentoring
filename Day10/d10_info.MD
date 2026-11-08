# 🎓 AI Engineer Mentorship — Session 10
## Production Model Serving with FastAPI, Docker & LLM Inference Optimization

**Level:** Intermediate → Advanced AI Engineering  
**Duration:** 60 minutes  
**Today's project:** Deploy the ML model from Session 9 as a production-oriented inference API and understand how serving changes when moving from traditional ML models to large language models.

---

## 1. Where We Are in Your AI Engineering Journey

We've progressively covered:

| Session | Topic |
|---|---|
| 1–2 | ML foundations, neural networks, and deep learning |
| 3–4 | Embeddings, vector search, and RAG |
| 5 | AI agents and function calling |
| 6 | Stateful agents and LangGraph |
| 7 | Agent evaluation and observability |
| 8 | LLM fine-tuning, LoRA, and QLoRA |
| 9 | MLOps, experiment tracking, and model registries |
| **10 — Today** | **Model serving and inference optimization** |

In Session 9, we trained a backend error-classification model and registered it using MLflow.

We learned how to track experiments, compare models, and manage model versions.

But registering a model doesn't make it available to applications.

**Today, we'll learn how to expose that model through an API and make it reliable under concurrent traffic.**

This is particularly relevant to your Python backend experience because you'll be applying familiar concepts—FastAPI, Docker, concurrency, deployment, and monitoring—to machine learning inference.

---

# 2. Learning Objectives

By the end of this session, you should be able to:

1. Explain the difference between model training and model serving.
2. Build an inference API using FastAPI.
3. Load a registered model efficiently during application startup.
4. Understand synchronous versus asynchronous inference.
5. Explain batching, concurrency, throughput, and latency.
6. Containerize a model-serving application using Docker.
7. Understand CPU versus GPU inference.
8. Explain KV caching, quantization, and continuous batching for LLMs.
9. Design a scalable production model-serving architecture.

### ⏱️ Today's 60-Minute Agenda

| Time | Topic |
|---|---|
| 0–10 min | Model serving fundamentals |
| 10–25 min | Build a FastAPI inference service |
| 25–35 min | Concurrency, batching, and performance |
| 35–43 min | Docker deployment and model lifecycle |
| 43–52 min | LLM inference architecture and optimization |
| 52–56 min | Production system design |
| 56–60 min | Practical assignment and interview quiz |

---

# Part 1: What Is Model Serving?

**Time: 0–10 minutes**

Let's start with a simple distinction.

## Training vs inference

During training, the model learns from data.

```text
Training Dataset
       |
       v
     Model
       |
       v
   Predictions
       |
       v
  Calculate Loss
       |
       v
 Backpropagation
       |
       v
 Update Weights
```

During inference, the model uses learned parameters to generate predictions.

```text
User Input
    |
    v
Preprocessing
    |
    v
Trained Model
    |
    v
Prediction
    |
    v
API Response
```

**Inference does not normally update model weights.**

Its purpose is to use an already-trained model to produce results.

## What is model serving?

Model serving means making a trained model available to other applications through a reliable interface.

For example, our backend error classifier receives:

```json
{
  "message": "PostgreSQL connection pool exhausted"
}
```

It returns:

```json
{
  "category": "database",
  "model_version": "1",
  "latency_ms": 12.5
}
```

The latency is illustrative.

Your actual latency depends on the model, hardware, and workload.

### Model serving architecture

```text
             CLIENT APPLICATION
                     |
                     v
                  FastAPI
                     |
                     v
              Input Validation
                     |
                     v
                Preprocessing
                     |
                     v
                Trained Model
                     |
                     v
                Prediction
                     |
                     v
              Output Validation
                     |
                     v
                API Response
```

For a traditional ML model, this architecture can be relatively simple.

For a large language model, it becomes significantly more complex because of GPU memory requirements, autoregressive token generation, batching, and scheduling.

We'll study both.

---

# Part 2: Build a FastAPI Model-Serving Application

**Time: 10–25 minutes**

## Project: Backend Error Classification API

We'll continue directly from Session 9.

Previously, we registered a model named:

`backend-error-classifier`

We also assigned the alias:

`candidate`

Today, we'll load that model into a FastAPI application.

### Step 1: Project structure

Create:

```text
ai-model-serving/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

For this exercise, your MLflow tracking database and model artifacts must remain accessible to the application.

During local development, you can use the same project directory as Session 9 or configure the tracking URI to point to your existing MLflow server.

## Step 2: Install dependencies

```bash
pip install fastapi uvicorn mlflow scikit-learn
```

For a reproducible deployment, use tested, pinned package versions in your final requirements file.

## Step 3: Understand the wrong approach

A beginner might implement inference like this:

```python
@app.post("/predict")
def predict(message: str):

    model = load_model()

    return model.predict([message])
```

What's wrong?

**The model is loaded for every request.**

Suppose loading the model takes two seconds.

Even if prediction takes only 20 milliseconds, every request may spend approximately two seconds loading the model.

For larger models, loading could take much longer.

This is inefficient.

### Better approach

Load the model once during application startup.

Then reuse it across requests.

FastAPI supports application startup and shutdown management through its lifespan mechanism. [FastAPI](https://fastapi.tiangolo.com/pt/advanced/events/?utm_source=chatgpt.com)

## Step 4: Implement the inference service

Create `app/main.py`.

```python
import asyncio
import logging
import os

from contextlib import asynccontextmanager
from time import perf_counter

import mlflow
import mlflow.sklearn

from mlflow import MlflowClient

from fastapi import (
    FastAPI,
    HTTPException,
    Request
)

from pydantic import BaseModel, Field

from starlette.concurrency import run_in_threadpool


logger = logging.getLogger(__name__)

MODEL_NAME = "backend-error-classifier"

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "candidate"
)

MAX_CONCURRENT_PREDICTIONS = 8


class PredictionRequest(BaseModel):
    message: str = Field(
        min_length=3,
        max_length=2000
    )


class PredictionResponse(BaseModel):
    category: str
    model_version: str
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db"
    )

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # Resolve the alias to a concrete version.
    version_info = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            MODEL_ALIAS
        )
    )

    model_version = str(version_info.version)

    # Pin the model loaded by this process.
    model_uri = (
        f"models:/{MODEL_NAME}/{model_version}"
    )

    model = mlflow.sklearn.load_model(
        model_uri
    )

    app.state.model = model

    app.state.model_version = model_version

    app.state.inference_semaphore = (
        asyncio.Semaphore(
            MAX_CONCURRENT_PREDICTIONS
        )
    )

    logger.info(
        "Loaded model version %s",
        model_version
    )

    yield

    app.state.model = None


app = FastAPI(
    title="AI Model Serving API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health/live")
def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
def readiness(request: Request):

    ready = (
        getattr(
            request.app.state,
            "model",
            None
        ) is not None
    )

    if not ready:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable"
        )

    return {"status": "ready"}


@app.post(
    "/v1/predict",
    response_model=PredictionResponse
)
async def predict(
    payload: PredictionRequest,
    request: Request
):

    start = perf_counter()

    async with (
        request.app.state.inference_semaphore
    ):

        try:
            prediction = await run_in_threadpool(
                request.app.state.model.predict,
                [payload.message]
            )

        except Exception:
            logger.exception(
                "Model prediction failed"
            )

            raise HTTPException(
                status_code=503,
                detail="Prediction unavailable"
            )

    duration_ms = (
        perf_counter() - start
    ) * 1000

    return PredictionResponse(
        category=str(prediction[0]),
        model_version=(
            request.app.state.model_version
        ),
        latency_ms=round(
            duration_ms,
            2
        )
    )
```

Let's understand the important engineering decisions.

### Decision 1: Load the model during startup

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
```

The model is loaded before the application begins handling requests.

If loading fails, startup fails rather than allowing the service to accept prediction requests without a model.

This is generally preferable to silently falling back to an unknown model.

### Decision 2: Pin the model version

Notice:

```python
version_info = client.get_model_version_by_alias(
    MODEL_NAME,
    MODEL_ALIAS
)
```

We resolve the alias to a specific version.

Then load that version.

Why?

Because registry aliases can change.

Suppose `candidate` points to Version 3 when the application starts.

Later, somebody updates it to Version 4.

Our running application should not unexpectedly switch models halfway through processing traffic.

Instead, it continues using the version loaded during startup.

A controlled restart or deployment can load the new version.

MLflow supports loading models by explicit version or alias. [MLflow AI Platform](https://www.mlflow.org/docs/latest/model-registry/?utm_source=chatgpt.com)

### Decision 3: Avoid blocking the event loop

We use:

```python
await run_in_threadpool(
    request.app.state.model.predict,
    [payload.message]
)
```

This is important.

Our Scikit-learn prediction function is synchronous.

If we call a slow synchronous function directly inside an `async def` endpoint, it can block the event loop.

Using a thread pool allows the event loop to continue handling other asynchronous operations.

However, this does not automatically make CPU-intensive Python code execute in parallel.

The Python GIL, native numerical libraries, and thread scheduling all influence actual throughput.

### Decision 4: Limit concurrency

```python
asyncio.Semaphore(8)
```

This restricts the number of simultaneous prediction operations inside one application process.

Why?

Because unlimited concurrency can overload CPU resources, memory, or downstream inference services.

The semaphore is process-local.

If you run four worker processes, each process has its own semaphore.

For a shared GPU or external model server, concurrency should also be controlled at the shared resource.

---

## Step 5: Run the application

Set the tracking URI to your Session 9 MLflow store.

For a local SQLite setup, run from the directory containing the tracking database and associated model artifacts:

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

export MODEL_ALIAS="candidate"
```

Start FastAPI:

```bash
uvicorn app.main:app \
    --host 127.0.0.1 \
    --port 8000
```

Test the endpoint:

```bash
curl -X POST \
  http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "message": "PostgreSQL connection pool exhausted"
  }'
```

An illustrative response:

```json
{
  "category": "database",
  "model_version": "1",
  "latency_ms": 8.42
}
```

The actual prediction and latency depend on your registered model.

### Exercise 1

Modify the request to:

```json
{
  "message": ""
}
```

What happens?

FastAPI uses Pydantic validation to reject the invalid input.

The endpoint should return HTTP 422 because the message doesn't satisfy the minimum length.

**This is an example of enforcing an API contract independently of the ML model.**

---

# Part 3: Concurrency, Throughput & Batching

**Time: 25–35 minutes**

This is one of the most important sections for a Senior AI Engineer.

## 3.1 Latency vs throughput

These terms are frequently confused.

**Latency** is the time required to complete an individual request.

**Throughput** is the number of requests completed per unit of time.

Suppose:

- One prediction takes 100 milliseconds.
- The service executes predictions sequentially.

The theoretical maximum throughput is approximately:

\[
\frac{1}{0.1}=10
\]

requests per second.

Now suppose the server can efficiently process four requests concurrently.

Its throughput may increase.

But the exact improvement depends on available CPU cores, memory, model implementation, and contention.

**Increasing concurrency does not guarantee increasing throughput.**

Eventually, the system becomes saturated.

## 3.2 Why p95 latency matters

Suppose your API processes five requests with latencies:

```text
20 ms
25 ms
30 ms
35 ms
500 ms
```

The average is:

\[
\frac{20+25+30+35+500}{5}=122ms
\]

But most requests completed much faster.

The slowest request took 500 ms.

Production systems often monitor latency percentiles.

| Metric | Meaning |
|---|---|
| p50 | Median request latency |
| p95 | Approximately 95% of requests complete within this duration |
| p99 | Approximately 99% complete within this duration |

With very small samples, percentile estimates are unstable.

You should use representative load tests and sufficient observations.

## 3.3 What is batching?

Suppose your model receives four requests.

Without batching:

```text
Request 1 → Model → Prediction
Request 2 → Model → Prediction
Request 3 → Model → Prediction
Request 4 → Model → Prediction
```

With batching:

```text
Request 1 ─┐
Request 2 ─┤
Request 3 ─┼──> Model ──> Four Predictions
Request 4 ─┘
```

Instead of calling:

```python
model.predict(["Error 1"])
model.predict(["Error 2"])
model.predict(["Error 3"])
```

We call:

```python
predictions = model.predict([
    "Error 1",
    "Error 2",
    "Error 3"
])
```

Batching can reduce per-item overhead and improve hardware utilization.

But there is a trade-off.

**Waiting for a batch to fill increases queueing latency.**

### Static batching vs dynamic batching

Static batching processes a predefined group of inputs.

Dynamic batching collects incoming requests for a short period or until a batch-size limit is reached.

For example:

```text
Maximum batch size: 16
Maximum batching wait: 10 ms
```

The serving engine collects up to 16 requests or waits up to 10 milliseconds before processing the batch.

These values are examples, not universal recommendations.

### Exercise 2

Suppose you have 100 requests.

Processing them individually takes 20 milliseconds each.

A batch of 10 takes 70 milliseconds.

Ignoring queueing and network overhead:

**Individual processing:**

\[
100\times20=2000ms
\]

**Batched processing:**

\[
10\times70=700ms
\]

Batching reduces total processing time in this example.

However, individual request latency may increase because requests wait for batching.

---

# Part 4: Dockerize the Inference Service

**Time: 35–43 minutes**

Now we'll package the application.

## Step 1: Create requirements.txt

```text
fastapi
uvicorn
mlflow
scikit-learn
```

For production, replace these unpinned dependencies with versions validated against your saved model and recorded in a lockfile.

## Step 2: Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir \
    -r requirements.txt

COPY app/ ./app/

RUN useradd --create-home appuser

USER appuser

EXPOSE 8000

CMD [
  "uvicorn",
  "app.main:app",
  "--host",
  "0.0.0.0",
  "--port",
  "8000",
  "--workers",
  "1"
]
```

For this container, we'll use an external MLflow tracking server and accessible artifact storage.

The Docker image contains the inference application, not the model registry database.

### Why use one worker initially?

Because every worker process loads its own model.

Suppose a model requires 2 GB of RAM.

Four independent worker processes may require approximately 8 GB for their model instances alone, depending on memory sharing and loading behavior.

For GPU inference, blindly increasing worker count can exhaust VRAM.

Start with a measured configuration rather than assuming more workers are always better.

## Step 3: Build the image

```bash
docker build \
  -t backend-error-inference:1.0 .
```

## Step 4: Run the container

Assuming your MLflow server is reachable at `mlflow.internal:5000`:

```bash
docker run --rm \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow.internal:5000 \
  -e MODEL_ALIAS=candidate \
  backend-error-inference:1.0
```

Replace the hostname with your actual MLflow server address.

The container must be able to access both the registry and the underlying model artifacts.

For local Docker development, you may need to configure host networking, a Docker network, or an accessible MLflow artifact server.

### Production consideration

For reliable deployments, you have two common options.

**Option A: Load the model from a remote registry.**

The inference service downloads the approved model when starting.

**Option B: Package the model artifact into the deployment image.**

The model becomes part of an immutable deployment artifact.

Option A provides flexibility.

Option B can improve deployment reproducibility and reduce runtime dependency on the model registry.

Neither approach is universally superior.

---

# Part 5: How LLM Serving Is Different

**Time: 43–52 minutes**

So far, we've served a conventional machine learning classifier.

Now let's discuss large language models.

Suppose you're serving a 7-billion-parameter LLM.

The architecture is different.

```text
                   CLIENT
                     |
                     v
                  FastAPI
                     |
                     v
               LLM Gateway
                     |
                     v
             Inference Scheduler
                     |
              ┌──────┴──────┐
              |             |
              v             v
          Request A     Request B
              |             |
              └──────┬──────┘
                     |
                     v
               GPU Inference
                     |
                     v
              Token Generation
                     |
                     v
              Streaming Response
```

Why do LLMs need specialized serving infrastructure?

Because autoregressive LLMs generate tokens sequentially.

## 5.1 Prefill and decode

LLM inference commonly involves two phases.

### Prefill

The model processes the input prompt.

For example:

> Explain why PostgreSQL connection pooling improves performance.

The model processes the prompt tokens and prepares attention-related state.

### Decode

The model generates output tokens one at a time.

```text
Token 1 → Token 2 → Token 3 → Token 4 ...
```

Each newly generated token depends on previously generated tokens.

This makes LLM serving different from a classifier that produces one output in a single forward pass.

## 5.2 What is KV caching?

Remember the attention mechanism from our transformer lessons.

During autoregressive generation, the model repeatedly attends to previous tokens.

Without caching, it would need to recompute previously generated keys and values.

KV caching stores the key and value representations from previous tokens.

Conceptually:

```text
Prompt Tokens
      |
      v
Attention Computation
      |
      v
Store Keys and Values
      |
      v
Generate Next Token
      |
      v
Reuse Cached KV
      |
      v
Generate Next Token
```

This reduces redundant computation.

However, KV caching consumes GPU memory.

The cache grows with the number of active sequences and their context lengths.

### Why is this important?

Suppose you serve 100 concurrent requests.

Each request has a different prompt length and generates a different number of tokens.

Your serving engine must manage memory for all active sequences.

This is one reason LLM serving requires sophisticated memory management and scheduling.

## 5.3 What is continuous batching?

Traditional static batching often waits for an entire batch to finish before admitting new requests.

But LLM requests have variable generation lengths.

Consider:

```text
Request A: 20 output tokens
Request B: 200 output tokens
Request C: 50 output tokens
```

If all requests are tightly coupled in one static batch, shorter requests can be affected by longer-running requests.

Continuous batching allows the serving engine to add or remove requests as generation progresses.

Conceptually:

```text
Iteration 1: A + B + C
Iteration 2: A + B + C
...
A finishes
...
Iteration N: B + C + D
...
C finishes
...
Iteration M: B + D + E
```

This can improve GPU utilization and overall throughput.

## 5.4 vLLM

vLLM is an inference engine designed for efficient LLM serving.

It supports capabilities such as optimized attention-memory management, request scheduling, and an OpenAI-compatible API server. [vLLM](https://docs.vllm.ai/en/v0.14.1/serving/openai_compatible_server/?utm_source=chatgpt.com)

Official documentation: [vLLM OpenAI-compatible serving](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/?utm_source=chatgpt.com).

A typical serving command is:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8001
```

This requires a compatible inference environment and sufficient hardware resources.

For a protected deployment, configure authentication and avoid exposing the inference endpoint publicly without access controls.

Now your FastAPI backend can call the inference server using an OpenAI-compatible client.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8001/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[
        {
            "role": "user",
            "content": (
                "Explain PostgreSQL "
                "connection pooling."
            )
        }
    ],
    max_tokens=150
)

print(
    response.choices[0].message.content
)
```

The `EMPTY` key is appropriate only for a local, trusted server that does not require authentication. Production deployments should use real access controls.

### Important architecture distinction

For a traditional ML model:

```text
FastAPI
   |
   v
Scikit-learn Model
```

For a large LLM:

```text
FastAPI
   |
   v
Inference Gateway
   |
   v
vLLM Server
   |
   v
GPU
```

Separating API orchestration from GPU inference makes scaling and resource management easier.

---

# Part 6: LLM Inference Optimization Techniques

Let's compare the main techniques.

| Technique | Primary benefit | Trade-off |
|---|---|---|
| Quantization | Reduces model memory usage | May affect quality or performance |
| KV caching | Avoids repeated attention computation | Consumes memory |
| Continuous batching | Improves request scheduling and throughput | Requires sophisticated scheduling |
| Streaming | Improves perceived responsiveness | Does not necessarily reduce total computation |
| Tensor parallelism | Distributes large models across GPUs | Communication overhead |
| Prefix caching | Reuses computation for shared prompt prefixes | Cache management and memory overhead |
| Speculative decoding | Can accelerate token generation | Additional model or verification overhead |

## Quantization

We introduced quantization in Session 8 while studying QLoRA.

But there is an important difference.

**Training-time quantization and inference-time quantization are related but distinct use cases.**

During QLoRA training, we use a quantized frozen base model and train adapters.

During inference, we may quantize model weights to reduce serving memory requirements.

For example:

| Model | Approximate raw weight memory |
|---|---:|
| 7B parameters, FP16 | 14 GB |
| 7B parameters, INT8 | 7 GB |
| 7B parameters, 4-bit | 3.5 GB |

These figures exclude KV cache, temporary buffers, quantization metadata, and runtime overhead.

Quantization does not always make inference faster.

Performance depends on hardware support, quantization format, and kernel implementation.

## Time to First Token vs Total Latency

For LLM serving, two measurements are particularly important.

**Time to First Token (TTFT):** Time between submitting a request and receiving the first generated token.

**Time per Output Token:** The average time spent generating subsequent tokens, measured using a clearly defined methodology.

Suppose:

```text
Request submitted
       |
       | 400 ms
       v
First token received
       |
       | 50 ms per token
       v
Additional tokens streamed
```

Users may perceive the application as responsive because they begin seeing output quickly.

However, total generation time may still be substantial.

This is why streaming is useful for chat applications.

---

# Part 7: Senior AI Engineer System Design

**Time: 52–56 minutes**

Imagine your company has developed an AI assistant that performs RAG and agent workflows.

It must support:

- 5,000 active users.
- Document retrieval.
- Multiple LLM models.
- Tool execution.
- Streaming responses.
- Request tracing.
- Model version management.

How would you design the architecture?

A possible solution:

```text
                   FRONTEND
                      |
                      v
                 API GATEWAY
                      |
                      v
               Authentication
                      |
                      v
                    FastAPI
                      |
            ┌─────────┼─────────┐
            |         |         |
            v         v         v
          RAG       Agent     Model Router
            |         |         |
            v         v         |
       Vector DB   Tool APIs    |
            |         |         |
            └─────────┼─────────┘
                      |
                      v
                 LLM Gateway
                      |
             ┌────────┴────────┐
             |                 |
             v                 v
         vLLM Server A     vLLM Server B
             |                 |
             v                 v
           GPU A             GPU B
                      |
                      v
               Response Stream


              SUPPORTING SYSTEMS

       PostgreSQL → Application State
       Redis      → Cache / Coordination
       MLflow     → Model Registry
       Telemetry  → Traces / Metrics / Logs
       Object Storage → Model Artifacts
```

This architecture separates concerns.

FastAPI handles authentication, request validation, and application orchestration.

The inference servers manage model execution.

The vector database handles retrieval.

The registry manages model versions.

The observability infrastructure collects traces and metrics.

### Production security

A model-serving API needs more than input validation.

It should also enforce authentication, authorization, rate limits, request-size limits, and resource quotas.

For multi-tenant systems, tenant isolation must be enforced before retrieving documents or invoking tools.

A malicious prompt must never be able to bypass backend authorization.

### Production observability

Measure:

| Metric | Why it matters |
|---|---|
| Request rate | Understand traffic |
| p50/p95/p99 latency | Detect performance degradation |
| Error rate | Detect serving failures |
| Queue wait time | Identify saturation |
| GPU utilization | Understand hardware usage |
| GPU memory usage | Prevent out-of-memory failures |
| Tokens per second | Measure generation throughput |
| Time to first token | Measure interactive responsiveness |
| Model version | Correlate regressions |
| Cost per request | Control operating expenses |

For a real deployment, you can integrate Prometheus and Grafana.

A Senior AI Engineer should be able to explain not only how a model works, but also how it behaves under load and how to recover when its serving infrastructure fails.

---

# Part 8: Today's Practical Assignment

**Time: 56–58 minutes**

## Mini-Project: Production-Oriented Model Serving

Build on the FastAPI application from today's session.

### Task 1 — Deploy the registered model

Load the model from your MLflow registry.

Verify that the API returns the correct model version.

### Task 2 — Add batch inference

Implement:

```http
POST /v1/predict/batch
```

Request:

```json
{
  "messages": [
    "PostgreSQL connection refused",
    "JWT token expired",
    "Redis memory limit exceeded"
  ]
}
```

Expected response structure:

```json
{
  "predictions": [
    "database",
    "authentication",
    "cache"
  ],
  "model_version": "1"
}
```

Use a bounded batch size.

For example, allow between 1 and 32 messages.

### Task 3 — Measure performance

Compare:

- 100 individual predictions.
- Ten batches containing ten predictions each.

Measure total processing time and average per-item latency.

Remember that this experiment measures your specific model and environment. It does not prove batching will always be faster.

### Task 4 — Test concurrency

Use an HTTP load-testing tool or a small asynchronous Python client.

Test concurrency levels of:

```text
1
5
10
20
```

Record:

| Concurrency | Throughput | p95 latency | Error rate |
|---|---|---|---|
| 1 | Measure | Measure | Measure |
| 5 | Measure | Measure | Measure |
| 10 | Measure | Measure | Measure |
| 20 | Measure | Measure | Measure |

Identify the point where increasing concurrency stops improving throughput or begins increasing latency significantly.

### Bonus challenge

Containerize the service and implement a readiness endpoint that reports unhealthy when the model is unavailable.

Run the same load tests inside Docker.

---

# Part 9: Senior AI Engineer Interview Questions

### Q1. Why shouldn't you load a model inside every prediction request?

Because model loading can involve disk access, deserialization, memory allocation, and initialization.

Loading once at startup avoids repeating these operations for every request.

### Q2. Does using `async def` make ML inference non-blocking?

No.

An asynchronous endpoint can still block the event loop if it directly executes expensive synchronous operations.

You need an appropriate execution strategy, such as a thread pool, process pool, background worker, or dedicated inference server.

### Q3. What is the difference between latency and throughput?

Latency measures how long an individual request takes.

Throughput measures how many requests the system completes per unit of time.

### Q4. Why is batching useful for GPU inference?

Batching allows hardware to process multiple inputs together, potentially improving computational efficiency and throughput.

However, batching can introduce queueing delays and increase memory consumption.

### Q5. What is KV caching?

KV caching stores previously computed attention keys and values so autoregressive generation can reuse them when producing subsequent tokens.

It reduces redundant computation at the cost of additional memory.

### Q6. Why is continuous batching useful for LLMs?

Because requests generate different numbers of tokens.

Continuous batching allows the serving engine to schedule active requests dynamically instead of requiring all requests in a static batch to complete together.

### Q7. How would you scale a model-serving system?

A strong answer should discuss traffic characteristics, model size, CPU/GPU requirements, batching, concurrency limits, horizontal scaling, queueing, load balancing, observability, and cost.

For GPU-backed LLMs, scaling is more complicated than simply increasing the number of FastAPI workers.

---

# 🧠 Today's Knowledge Quiz

Try answering these questions before checking the answer key.

**Question 1**

Your model takes 500 milliseconds to load and 10 milliseconds to predict.

You receive 1,000 sequential requests.

Approximately how much time would you spend if you loaded the model for every request?

How much would you spend if you loaded it only once?

Ignore all other overhead.

**Question 2**

Suppose your API has an average latency of 50 milliseconds but a p99 latency of 2 seconds.

What does this tell you?

What would you investigate?

**Question 3**

What is the difference between static batching and continuous batching?

**Question 4**

Why can running four FastAPI workers cause problems when serving a large GPU model?

**Question 5**

Does quantization always improve inference speed?

Explain your answer.

**Question 6 — Advanced System Design**

You need to serve a 7B-parameter LLM to 1,000 concurrent users.

The application requires streaming responses, RAG, tool calling, and a p95 time-to-first-token target of less than two seconds.

How would you design the system?

Consider GPU capacity, model serving, batching, request queues, rate limits, autoscaling, and observability.

### Quick answer key

**Q1:** Loading every time takes approximately 510 seconds. Loading once takes approximately 10.5 seconds.

**Q2:** A small fraction of requests experiences severe latency. Investigate queueing, resource saturation, slow dependencies, and workload differences.

**Q3:** Static batching groups requests for execution; continuous batching dynamically schedules requests as generation progresses.

**Q4:** Each worker may load its own model and exhaust GPU memory.

**Q5:** No. Performance depends on hardware, kernels, model architecture, and quantization format.

For Question 6, there is no single correct architecture. The important part is explaining your capacity assumptions and trade-offs.

---

# Session 10 Recap

Today, we moved from training and registering models to serving them in production.

We learned how to build a FastAPI inference API, load models during startup, control concurrency, measure latency, and package a service using Docker.

We also explored the additional complexity of LLM serving, including KV caching, continuous batching, quantization, streaming, and GPU memory management.

The complete lifecycle now looks like:

\[
\boxed{
\text{Training}
\rightarrow
\text{Evaluation}
\rightarrow
\text{Registry}
\rightarrow
\text{Serving}
\rightarrow
\text{Monitoring}
}
\]

**The key lesson:** A model that performs well during training is not automatically a model that can serve thousands of users reliably. Production AI engineering requires careful management of latency, throughput, memory, concurrency, and deployment reliability.

---

## 🔜 Session 11: Vector Databases & Production RAG Architecture

Next, we'll revisit RAG at a more advanced, production-engineering level.

We'll focus on the problems that arise when your document collection grows from a few hundred documents to millions of chunks.

We'll study approximate nearest-neighbor search, HNSW indexing, metadata filtering, vector database selection, ingestion pipelines, incremental updates, document versioning, multi-tenant isolation, and retrieval performance.

**Practical project:** Build a scalable document-ingestion and retrieval service using FastAPI, PostgreSQL with pgvector, and asynchronous background processing.

This will connect your backend development experience with the design of production-grade AI knowledge systems.

**Mentorship schedule note:** I also disabled the overlapping mentorship schedule to avoid duplicate lessons. Your preferred schedule remains active.