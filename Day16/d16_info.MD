# 🎓 AI Engineer Mentorship — Session 16
## Production LLM Deployment: vLLM, GPU Inference, Docker, Kubernetes & Model Serving

**Level:** Intermediate → Advanced AI Engineering  
**Duration:** 60 minutes  
**Target role:** Senior AI Engineer  
**Today's practical project:** Deploy an LLM inference service and connect it to a production-style FastAPI gateway.

---

# 1. Where We Are in Your AI Engineering Journey

We've progressively covered RAG retrieval, evaluation, observability, and security.

| Session | Topic |
|---|---|
| 11 | Vector databases, pgvector, and HNSW |
| 12 | Hybrid retrieval, BM25, RRF, and reranking |
| 13 | RAG evaluation and hallucination detection |
| 14 | OpenTelemetry, tracing, and LLM monitoring |
| 15 | Prompt injection, tenant isolation, and AI security |
| **16 — Today** | **LLM serving, GPU infrastructure, and deployment** |

Until now, we've generally treated the LLM as an external service.

For example:

```python
response = await llm.ainvoke(prompt)
```

But what happens behind that API call?

How does the model load into memory?

How does a GPU process multiple requests?

What happens when 100 users simultaneously request responses?

How do we deploy a 7-billion-parameter model?

And how do we prevent the inference server from crashing under load?

These are important questions for production AI engineering.

Today, we'll focus on the infrastructure that runs LLMs.

---

# 2. Learning Objectives

By the end of this session, you should understand:

1. The difference between model training and model inference.
2. Why LLM serving differs from traditional API serving.
3. How GPU memory affects model deployment.
4. What prefill, decoding, and KV caching mean.
5. How continuous batching improves inference throughput.
6. How to deploy an LLM using vLLM.
7. How to connect FastAPI to an inference server.
8. How to containerize an AI application.
9. How Kubernetes schedules GPU workloads.
10. How to scale and monitor production inference services.

## ⏱️ 60-Minute Agenda

| Time | Topic | Goal |
|---|---|---|
| 0–8 min | Model serving fundamentals | Understand inference architecture |
| 8–18 min | GPU memory, prefill, decoding, and KV cache | Understand resource requirements |
| 18–28 min | vLLM and continuous batching | Learn inference optimization |
| 28–43 min | FastAPI inference gateway | Build practical Python code |
| 43–50 min | Docker and vLLM deployment | Run the application |
| 50–56 min | Kubernetes and production scaling | Learn deployment architecture |
| 56–60 min | Exercise, interview questions, and quiz | Validate understanding |

---

# Part 1: What Is Model Serving?
**Time: 0–8 minutes**

Let's compare traditional backend development with AI engineering.

## 1.1 Traditional FastAPI application

Suppose you build this endpoint:

```python
@app.get("/calculate")
def calculate(a: int, b: int):
    return {"result": a + b}
```

A request comes in.

Python performs a small calculation.

The server returns the response.

```text
Client
   |
   v
FastAPI
   |
   v
Python Function
   |
   v
Response
```

The computation is lightweight.

You can usually scale the application by running more workers or replicas, subject to CPU, memory, and other resource constraints.

## 1.2 LLM application

Now imagine this endpoint:

```python
@app.post("/generate")
async def generate(question: str):

    response = await model.generate(question)

    return {"answer": response}
```

The apparent simplicity hides substantial computation.

An LLM may contain billions of parameters.

Generating an answer involves repeatedly executing neural network operations.

The architecture looks more like:

```text
                     Client
                       |
                       v
                    FastAPI
                       |
                       v
                 Inference Server
                       |
                       v
                  Model Weights
                       |
                       v
                  GPU Computation
                       |
                       v
                  Token Generation
                       |
                       v
                     Response
```

The inference server is responsible for loading model weights, allocating memory, scheduling requests, and generating tokens.

### Important distinction

**Model training** updates model parameters using training data and optimization algorithms.

**Model inference** uses the learned parameters to generate predictions or responses.

Training is generally much more computationally expensive than an individual inference request.

However, serving an LLM to thousands of users can also require substantial infrastructure.

### Senior AI Engineer insight

A production architecture often separates the application API from the model inference service.

```text
                  User
                   |
                   v
              API Gateway
                   |
                   v
                 FastAPI
                   |
          ┌────────┴────────┐
          |                 |
          v                 v
     Authentication      RAG Service
                            |
                            v
                       LLM Gateway
                            |
                            v
                       vLLM Server
                            |
                            v
                           GPU
```

This separation allows the backend API and inference infrastructure to scale independently.

---

# Part 2: Understanding GPU Memory Requirements
**Time: 8–18 minutes**

This is an essential topic for Senior AI Engineer interviews.

Suppose you want to deploy a 7-billion-parameter LLM.

How much GPU memory does it need?

## 2.1 Model parameters

A model parameter is a numerical value learned during training.

For example:

```python
weight = 0.7532
```

A neural network contains many such values.

A 7B model contains approximately seven billion parameters.

The memory required to store those parameters depends on their numerical representation.

## 2.2 FP32, FP16, and INT8

Different numerical formats require different amounts of memory.

| Format | Approximate bytes per parameter | 7B model weights |
|---|---:|---:|
| FP32 | 4 | 28 GB |
| FP16 | 2 | 14 GB |
| BF16 | 2 | 14 GB |
| INT8 | 1 | 7 GB |
| INT4 | 0.5 | 3.5 GB |

These are approximate **weight-storage requirements**, not total inference memory requirements.

Quantized models may also require memory for scales, metadata, and other supporting structures.

### Calculation

For FP16:

\[
Memory = Parameters \times BytesPerParameter
\]

\[
Memory = 7,000,000,000 \times 2
\]

\[
Memory = 14,000,000,000\text{ bytes}
\]

Approximately 14 GB in decimal units.

But this does not mean a 14 GB GPU can necessarily serve the model.

Why?

Because inference requires additional memory.

---

## 2.3 Where does the additional memory go?

An inference server uses GPU memory for several purposes.

```text
                GPU Memory
                    |
        ┌───────────┼───────────┐
        |           |           |
        v           v           v
    Model Weights  KV Cache   Temporary Buffers
        |           |           |
        └───────────┼───────────┘
                    |
                    v
              GPU Operations
```

The total memory requirement is approximately:

\[
TotalMemory =
Weights + KVCache + RuntimeOverhead
\]

Other factors include CUDA allocations, temporary tensors, framework overhead, and model-specific components.

### Example

Suppose a model requires:

```text
Model weights:         14 GB
KV cache:               5 GB
Runtime overhead:       2 GB
----------------------------
Estimated total:       21 GB
```

These are illustrative values, not measurements for a particular model.

A GPU with 24 GB of memory might accommodate this workload, but actual capacity depends on context length, concurrency, architecture, and serving configuration.

---

# Part 3: Prefill, Decoding, and KV Cache
**Time: 18–23 minutes**

When an LLM generates a response, two important computational phases occur.

## 3.1 Prefill phase

Suppose the user submits:

> Explain how PostgreSQL connection pooling works.

The model must process the input tokens.

During prefill, it processes the prompt and computes the attention state needed for generation.

Conceptually:

```text
Input Prompt
     |
     v
Tokenization
     |
     v
Transformer Layers
     |
     v
Attention Computation
     |
     v
KV Cache Created
```

Longer prompts generally require more processing and memory.

This is one reason RAG applications can become slow when they retrieve too many document chunks.

## 3.2 Decode phase

After processing the prompt, the model generates output tokens.

For an autoregressive LLM, tokens are generated sequentially.

```text
Input: Explain Redis

Generated tokens:

Redis
Redis is
Redis is an
Redis is an in-memory
Redis is an in-memory database
```

Each new token depends on the previous context.

This sequential dependency creates a performance challenge.

## 3.3 What is KV caching?

Transformer attention uses query, key, and value representations.

During autoregressive generation, previously computed key and value representations can be reused.

Without caching, the model would repeatedly recompute attention projections for earlier tokens.

With KV caching, it stores the relevant previous key and value states.

```text
Token 1 ──> K1, V1
Token 2 ──> K2, V2
Token 3 ──> K3, V3
Token 4 ──> K4, V4
```

When generating the next token, the model can reuse cached states.

This reduces redundant computation.

However, the cache consumes memory.

### Why does this matter in production?

Suppose your inference server handles 100 simultaneous users.

Each request may have a different context length.

The inference engine must allocate and manage KV cache memory for those requests.

Poor memory management can reduce concurrency or cause out-of-memory failures.

---

# Part 4: Continuous Batching and vLLM
**Time: 23–28 minutes**

Now let's introduce vLLM.

vLLM is an inference and serving engine designed to efficiently run large language models.

It provides an OpenAI-compatible API server and supports optimizations such as continuous batching and efficient KV cache management.

Its official documentation includes deployment instructions and an OpenAI-compatible serving interface. [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart/?utm_source=chatgpt.com)

## 4.1 Traditional batching

Imagine three requests arrive.

```text
Request A: 100 output tokens
Request B:  20 output tokens
Request C:  60 output tokens
```

With a rigid batch, requests may remain grouped until the batch finishes.

Short requests can waste computation or wait unnecessarily while longer requests continue.

## 4.2 Continuous batching

Continuous batching allows the inference scheduler to dynamically manage active requests.

When a request finishes, another waiting request can enter the batch.

Conceptually:

```text
Time ───────────────────────────────>

Step 1: [A] [B] [C]
Step 2: [A] [B] [C]
Step 3: [A]     [C] [D]
Step 4: [A]     [C] [D]
Step 5: [A]         [D] [E]
```

This improves hardware utilization when multiple requests are being served.

However, increased batching can also affect latency, particularly under heavy load.

The goal is not simply to maximize the number of concurrent requests.

It is to balance throughput, latency, memory usage, and cost.

## 4.3 PagedAttention

vLLM also uses efficient KV cache management techniques associated with PagedAttention.

The central idea is to organize KV cache memory into blocks rather than requiring one large contiguous allocation for each sequence.

This helps reduce memory fragmentation and improves the ability to manage requests with different sequence lengths. [vLLM](https://docs.vllm.ai/en/v0.9.1/design/automatic_prefix_caching.html?utm_source=chatgpt.com)

### Interview question

**What is the difference between continuous batching and KV caching?**

KV caching avoids recomputing previously calculated key and value states during token generation.

Continuous batching schedules multiple requests efficiently so the GPU can process active sequences together.

They solve different problems.

---

# Part 5: Practical Project — Build an LLM Inference Gateway
**Time: 28–43 minutes**

We'll now build a FastAPI application that communicates with an inference server.

Our architecture:

```text
                    Client
                      |
                      v
                FastAPI Gateway
                      |
                      v
                HTTPX Async Client
                      |
                      v
              OpenAI-Compatible API
                      |
                      v
                Inference Server
                      |
                      v
                    Model
```

We'll first use a mock inference server so you can run everything on your existing Python development environment without a GPU.

Afterward, we'll replace the mock server with vLLM.

## Step 1: Create the project

```text
llm-serving/
    |
    ├── gateway.py
    ├── mock_model.py
    ├── requirements.txt
    ├── Dockerfile
    └── compose.yaml
```

Create `requirements.txt`:

```text
fastapi
uvicorn
httpx
```

Install:

```bash
pip install -r requirements.txt
```

## Step 2: Implement the FastAPI gateway

Create `gateway.py`.

```python
import os
import secrets
from contextlib import asynccontextmanager

import httpx

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


MODEL_URL = os.environ["MODEL_URL"].rstrip("/")
MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_API_KEY = os.environ["MODEL_API_KEY"]
GATEWAY_API_KEY = os.environ["GATEWAY_API_KEY"]


@asynccontextmanager
async def lifespan(app: FastAPI):

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(
            30.0,
            connect=3.0
        )
    ) as client:

        app.state.model_client = client

        yield


app = FastAPI(
    title="LLM Inference Gateway",
    lifespan=lifespan
)


class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=4000
    )

    max_tokens: int = Field(
        default=128,
        ge=1,
        le=512
    )


def verify_api_key(key: str | None):

    if (
        key is None
        or not secrets.compare_digest(
            key,
            GATEWAY_API_KEY
        )
    ):

        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )


@app.get("/health/live")
async def liveness():

    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():

    try:

        response = await app.state.model_client.get(
            f"{MODEL_URL}/v1/models",
            headers={
                "Authorization": (
                    f"Bearer {MODEL_API_KEY}"
                )
            },
            timeout=3.0
        )

        response.raise_for_status()

    except httpx.HTTPError:

        raise HTTPException(
            status_code=503,
            detail="Model service unavailable"
        )

    return {"status": "ready"}


@app.post("/chat")
async def chat(
    payload: ChatRequest,
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key"
    )
):

    verify_api_key(x_api_key)

    request_data = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful technical assistant."
                )
            },
            {
                "role": "user",
                "content": payload.question
            }
        ],
        "max_tokens": payload.max_tokens,
        "temperature": 0.2
    }

    try:

        response = await app.state.model_client.post(
            f"{MODEL_URL}/v1/chat/completions",
            json=request_data,
            headers={
                "Authorization": (
                    f"Bearer {MODEL_API_KEY}"
                )
            }
        )

        response.raise_for_status()

        result = response.json()

        answer = result["choices"][0][
            "message"
        ]["content"]

    except httpx.TimeoutException:

        raise HTTPException(
            status_code=504,
            detail="Model inference timed out"
        )

    except httpx.HTTPStatusError as exc:

        if exc.response.status_code in (429, 503):

            raise HTTPException(
                status_code=503,
                detail="Model service overloaded"
            )

        raise HTTPException(
            status_code=502,
            detail="Model service returned an error"
        )

    except (
        httpx.RequestError,
        ValueError,
        KeyError,
        IndexError,
        TypeError
    ):

        raise HTTPException(
            status_code=502,
            detail="Invalid model service response"
        )

    return {
        "answer": answer,
        "model": MODEL_NAME,
        "usage": result.get("usage", {})
    }
```

### Understanding the architecture

Notice that we use:

```python
httpx.AsyncClient()
```

instead of creating a new HTTP client for every request.

The shared client supports connection reuse and avoids unnecessary client initialization overhead.

Also notice:

```python
timeout=httpx.Timeout(30.0, connect=3.0)
```

This establishes a general timeout and a shorter connection timeout.

A production service may require separate connect, read, write, and pool timeouts.

For long-running generation, the read timeout must account for expected model latency.

### Why separate liveness and readiness?

The liveness endpoint indicates that the gateway process is running.

The readiness endpoint checks whether the gateway can communicate with the model service.

This distinction is important when deploying to Kubernetes.

A temporary model-service outage should not necessarily cause Kubernetes to restart every gateway container.

---

## Step 3: Create a mock inference server

Create `mock_model.py`.

```python
import os
import secrets

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


app = FastAPI()

API_KEY = os.environ["MODEL_API_KEY"]


class CompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int = 128


def authenticate(authorization: str | None):

    expected = f"Bearer {API_KEY}"

    if (
        authorization is None
        or not secrets.compare_digest(
            authorization,
            expected
        )
    ):

        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )


@app.get("/v1/models")
def list_models(
    authorization: str | None = Header(default=None)
):

    authenticate(authorization)

    return {
        "data": [
            {
                "id": "demo-model",
                "object": "model"
            }
        ]
    }


@app.post("/v1/chat/completions")
def completion(
    payload: CompletionRequest,
    authorization: str | None = Header(default=None)
):

    authenticate(authorization)

    return {
        "model": payload.model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "This is a simulated LLM response."
                    )
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30
        }
    }
```

This mock implements the subset of the OpenAI-compatible API needed by our gateway.

It doesn't perform real inference.

The token counts are simulated.

The advantage is that we can verify our networking, authentication, request validation, and deployment behavior before introducing GPU dependencies.

---

# Part 6: Containerize the Application
**Time: 43–50 minutes**

Now we'll use Docker to run both services.

## Step 1: Create a Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY gateway.py mock_model.py ./

EXPOSE 8000

CMD [
    "uvicorn",
    "gateway:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]
```

For production, pin dependency versions, use an appropriately restricted runtime user, and scan the resulting image.

## Step 2: Create Docker Compose configuration

Create `compose.yaml`:

```yaml
services:

  mock-model:
    build: .
    command:
      - uvicorn
      - mock_model:app
      - --host
      - 0.0.0.0
      - --port
      - "8001"
    environment:
      MODEL_API_KEY: local-model-key

  gateway:
    build: .
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      MODEL_URL: http://mock-model:8001
      MODEL_NAME: demo-model
      MODEL_API_KEY: local-model-key
      GATEWAY_API_KEY: local-gateway-key
    depends_on:
      - mock-model
```

The credentials here are disposable local-development values. Use a secrets manager or another secure secret-delivery mechanism for real deployments.

Start the application:

```bash
docker compose up --build
```

Test the gateway:

```bash
curl -X POST \
  http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: local-gateway-key" \
  -d '{
    "question": "Explain KV caching",
    "max_tokens": 100
  }'
```

Expected response:

```json
{
  "answer": "This is a simulated LLM response.",
  "model": "demo-model",
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 10,
    "total_tokens": 30
  }
}
```

We've now built a containerized inference gateway.

The next step is replacing the mock model with a real LLM.

---

# Part 7: Deploy a Real Model Using vLLM

If you have a compatible NVIDIA GPU, you can run an actual model using vLLM.

For this exercise, we'll use a small model:

`Qwen/Qwen3-0.6B`

The official vLLM Docker documentation demonstrates this model with the `vllm/vllm-openai` container image. [vLLM](https://docs.vllm.ai/en/latest/deployment/docker/?utm_source=chatgpt.com)

### Prerequisites

You need a compatible GPU, working GPU drivers, Docker GPU support, and sufficient GPU memory.

On Windows with WSL2, GPU passthrough must also be correctly configured.

First, verify GPU access:

```bash
nvidia-smi
```

If the GPU is not visible, resolve the driver/runtime configuration before proceeding.

## Run vLLM

Stop the mock application first:

```bash
docker compose down
```

Then start vLLM:

```bash
docker run --gpus all \
  --ipc=host \
  -p 127.0.0.1:8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --api-key local-model-key
```

This is a local development example.

For production, pin a tested image version rather than using `latest`.

The model will be downloaded if it isn't already cached.

Model loading may take some time.

## Test the inference server

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-model-key" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {
        "role": "user",
        "content": "Explain connection pooling simply."
      }
    ],
    "max_tokens": 100
  }'
```

The server should return a generated response if the model loads successfully.

### Connect your FastAPI gateway

Run the gateway locally with these environment variables:

```bash
export MODEL_URL=http://127.0.0.1:8001
export MODEL_NAME=Qwen/Qwen3-0.6B
export MODEL_API_KEY=local-model-key
export GATEWAY_API_KEY=local-gateway-key

uvicorn gateway:app --port 8000
```

Now your gateway uses a real inference server instead of the mock.

The important architectural benefit is that **the gateway code did not need to change**.

Only its configuration changed.

This is a useful pattern when designing AI systems that may switch between model providers or inference backends.

Official reference: [vLLM Docker Deployment Guide](https://docs.vllm.ai/en/latest/deployment/docker/?utm_source=chatgpt.com).

---

# Part 8: Kubernetes GPU Deployment
**Time: 50–56 minutes**

Now imagine your application receives thousands of requests per day.

Running one Docker container manually isn't enough for a resilient production deployment.

You may need orchestration, resource scheduling, health checks, and controlled rollouts.

This is where Kubernetes becomes useful.

## 8.1 Why is GPU scheduling different?

Traditional workloads request CPU and memory.

An LLM inference workload may also require a GPU.

Kubernetes supports GPU scheduling through vendor device plugins.

For example, a properly configured NVIDIA GPU node exposes the `nvidia.com/gpu` resource. [Kubernetes](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/?utm_source=chatgpt.com)

A container can request a GPU using:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

This requires compatible GPU nodes and an installed device plugin.

A GPU limit alone does not install drivers or configure the cluster.

---

## 8.2 Example Kubernetes deployment

Here's an illustrative deployment for our small model.

```yaml
apiVersion: apps/v1
kind: Deployment

metadata:
  name: llm-inference

spec:
  replicas: 1

  selector:
    matchLabels:
      app: llm-inference

  template:
    metadata:
      labels:
        app: llm-inference

    spec:
      containers:
        - name: vllm

          image: vllm/vllm-openai:latest

          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --port
            - "8000"

          ports:
            - containerPort: 8000

          resources:
            requests:
              cpu: "2"
              memory: 4Gi

            limits:
              cpu: "4"
              memory: 8Gi
              nvidia.com/gpu: 1

          startupProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 10
            failureThreshold: 60

          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 5

          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 30
```

This is a starting example, not a production-ready deployment.

The CPU and memory allocations must be adjusted for your model, GPU, and workload.

For production, also configure authentication, restricted network access, model caching, secret management, rollout strategy, and a pinned image version.

The vLLM Kubernetes guide documents the `/health` endpoint and deployment considerations. [vLLM](https://docs.vllm.ai/en/latest/deployment/k8s/?utm_source=chatgpt.com)

### Why use a startup probe?

Loading model weights can take significantly longer than starting a conventional API.

Without a startup probe, Kubernetes might repeatedly restart a container that is still loading its model.

A startup probe gives the process time to initialize before regular liveness and readiness checks begin.

Kubernetes documents this distinction in its [health probe guide](https://kubernetes.io/docs/concepts/workloads/pods/probes/?utm_source=chatgpt.com). [Kubernetes](https://kubernetes.io/docs/concepts/workloads/pods/probes/?utm_source=chatgpt.com)

---

## 8.3 How do we scale LLM inference?

Suppose one GPU server handles 20 concurrent requests while meeting your latency target.

Your application now receives 100 concurrent requests.

Should you simply increase FastAPI workers?

No.

More FastAPI workers do not increase the compute capacity of the inference GPU.

You need to examine the actual bottleneck.

A simplified scaling architecture:

```text
                      Clients
                         |
                         v
                    API Gateway
                         |
                         v
                  Request Router
                         |
            ┌────────────┼────────────┐
            |            |            |
            v            v            v
         vLLM Pod 1   vLLM Pod 2   vLLM Pod 3
            |            |            |
            v            v            v
          GPU 1        GPU 2        GPU 3
```

Each replica loads a model and serves requests.

However, this consumes additional GPU memory and infrastructure resources.

### Two important scaling approaches

**Horizontal scaling:** Add more inference replicas, each with the required GPU resources.

**Model parallelism:** Split a model across multiple GPUs when it cannot fit efficiently on one GPU or when the deployment benefits from parallel execution.

These approaches are not interchangeable.

Horizontal scaling primarily increases the number of independently served requests.

Model parallelism distributes computation for a model across devices.

---

# Part 9: Production Performance and Capacity Planning

Let's connect today's lesson to Session 14's observability work.

A production inference system should monitor:

| Metric | Why it matters |
|---|---|
| Time to first token | Initial response latency |
| Tokens per second | Generation throughput |
| p95 inference latency | Slow-request performance |
| Queue waiting time | Capacity pressure |
| GPU utilization | Compute usage |
| GPU memory utilization | Memory pressure |
| KV cache utilization | Available sequence capacity |
| Active requests | Current concurrency |
| Waiting requests | Scheduler backlog |
| Error and OOM rates | Reliability |

vLLM exposes production metrics through its `/metrics` endpoint, including request activity and KV cache-related measurements. [vLLM](https://docs.vllm.ai/en/latest/usage/metrics/?utm_source=chatgpt.com)

Official reference: [vLLM Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/?utm_source=chatgpt.com).

### Capacity-planning example

Suppose your inference server achieves 1,000 aggregate output tokens per second under a particular benchmark workload.

Your average request generates 200 output tokens.

A simple throughput estimate is:

\[
RequestsPerSecond =
\frac{1000}{200}
\]

\[
RequestsPerSecond = 5
\]

This is only a rough estimate.

Real throughput also depends on prompt processing, scheduling, request arrival patterns, output lengths, and latency requirements.

For example, a system might achieve high aggregate throughput while producing unacceptable p95 latency.

**Senior AI Engineer principle:** Capacity planning must consider throughput and latency together.

---

# Part 10: Practical Exercise — Production Model Serving

## Mini-Project: Deploy and Benchmark an LLM Service

Your assignment is to extend today's inference gateway.

### Task 1 — Run the application

Run the FastAPI gateway and mock inference service using Docker Compose.

Verify authentication, readiness, and response handling.

### Task 2 — Replace the mock

If a compatible GPU is available, deploy a small model using vLLM.

Otherwise, continue with the mock server and focus on the gateway and deployment architecture.

### Task 3 — Measure inference latency

Add a timing measurement around the HTTPX inference request.

For example:

```python
import time

start = time.perf_counter()

response = await client.post(
    model_url,
    json=payload
)

elapsed = time.perf_counter() - start

print(f"Inference latency: {elapsed:.3f}s")
```

Use the actual variables from your application when integrating this example.

### Task 4 — Test concurrency

Create a small asynchronous benchmark:

```python
import asyncio
import time

import httpx


async def send_request(client, request_id):

    response = await client.post(
        "http://127.0.0.1:8000/chat",
        headers={
            "X-API-Key": "local-gateway-key"
        },
        json={
            "question": f"Explain Redis request {request_id}",
            "max_tokens": 100
        }
    )

    return response.status_code


async def main():

    async with httpx.AsyncClient(
        timeout=60.0
    ) as client:

        start = time.perf_counter()

        results = await asyncio.gather(
            *[
                send_request(client, i)
                for i in range(20)
            ]
        )

        elapsed = time.perf_counter() - start

        print("Status codes:", results)

        print(
            f"Total time: {elapsed:.2f}s"
        )


asyncio.run(main())
```

Run the benchmark against the mock server first.

Then compare its behavior with a real inference server.

Remember that the mock doesn't perform GPU computation, so its performance is not representative of real model inference.

### Task 5 — Design scaling

Assume your application receives 50,000 AI requests daily.

Estimate the required inference capacity using measured token usage, arrival patterns, and latency targets.

Explain how you would scale the system without unnecessarily increasing GPU costs.

---

# Part 11: Senior AI Engineer Interview Questions

### Question 1: What is the difference between model training and inference?

Training updates model parameters using data and optimization.

Inference uses the trained parameters to generate predictions or responses.

### Question 2: Why can't we simply increase FastAPI workers to improve LLM inference throughput?

Because the primary bottleneck may be GPU computation or GPU memory.

Increasing API workers does not automatically increase inference capacity and may overload the model server.

### Question 3: What is KV caching?

KV caching stores previously computed attention key and value states so an autoregressive transformer can reuse them during token generation.

This reduces redundant computation but consumes memory.

### Question 4: What is continuous batching?

Continuous batching dynamically schedules active inference requests, allowing new requests to enter processing as capacity becomes available.

It can improve GPU utilization and aggregate throughput.

### Question 5: What is the difference between horizontal scaling and tensor parallelism?

Horizontal scaling adds independent model-serving replicas.

Tensor parallelism distributes parts of a model's computation across multiple devices.

### Question 6: Why are Kubernetes readiness probes important for LLM serving?

Because model loading and initialization can take time.

Readiness probes help prevent traffic from reaching a model server before it is ready to handle requests.

### Question 7: What would you do if an LLM server experiences GPU out-of-memory errors?

I would inspect model weight memory, KV cache usage, context lengths, concurrent requests, and runtime allocations.

Potential solutions include reducing concurrency or context limits, using compatible quantization, selecting a smaller model, or increasing available GPU memory.

I would benchmark each change to ensure it meets quality and latency requirements.

---

# 🧠 Session 16 Quiz

Try answering these without reviewing the explanations.

**Q1.** A model has 7 billion parameters stored in FP16. Approximately how much memory is required for the model weights alone?

**Q2.** Why does a model require more GPU memory than its weight-storage size?

**Q3.** What is the difference between prefill and decoding?

**Q4.** What problem does continuous batching solve?

**Q5.** Your inference server is overloaded, but FastAPI CPU usage is only 15%. Would increasing FastAPI workers necessarily improve performance? Why?

**Q6.** What is the difference between a Kubernetes startup probe and a readiness probe?

**Q7 — Senior System Design Challenge**

You need to deploy a 7B model for an enterprise AI assistant.

Requirements:

| Requirement | Target |
|---|---|
| Daily requests | 100,000 |
| Average input tokens | 1,500 |
| Average output tokens | 300 |
| p95 latency | Under 5 seconds |
| Availability | 99.9% |
| Deployment | Kubernetes |
| Inference | Self-hosted GPU |

Design the architecture.

Explain how you would select GPUs, estimate memory, configure vLLM, handle concurrent requests, scale inference replicas, manage model versions, and monitor production performance.

Also explain how you would deploy a new model version without immediately directing all production traffic to it.

---

# Session 16 Recap

Today, we moved from building AI applications to understanding how LLMs run in production.

We covered GPU memory requirements, prefill, decoding, KV caching, continuous batching, vLLM, FastAPI inference gateways, Docker, and Kubernetes.

The key architectural pattern is:

\[
\boxed{
\text{Client}
\rightarrow
\text{FastAPI Gateway}
\rightarrow
\text{Inference Server}
\rightarrow
\text{GPU}
\rightarrow
\text{Response}
}
\]

A backend developer knows how to expose an API.

A production AI engineer must additionally understand how model computation, GPU memory, request scheduling, latency, and infrastructure costs interact.

**The most important takeaway:** Scaling an AI application is not the same as scaling a traditional REST API. You must understand the inference engine and its resource constraints.

---

## 🔜 Session 17: LLM Fine-Tuning — LoRA, QLoRA, PEFT & Training Custom Models

In the next session, we'll move from deploying pretrained models to adapting models for specific tasks.

We'll cover the difference between prompting, RAG, and fine-tuning; supervised fine-tuning; LoRA and QLoRA; quantization-aware memory considerations; training datasets; and evaluating fine-tuned models.

We'll also examine when fine-tuning is unnecessary and why retrieval may be a better solution for frequently changing knowledge.

**Practical project:** Prepare an instruction-tuning dataset and implement a parameter-efficient fine-tuning workflow using Python.

For now, prioritize understanding **KV caching, continuous batching, and the separation between the FastAPI gateway and the inference server**. These three concepts form the foundation for designing scalable LLM-serving systems.