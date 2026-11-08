# 🎓 AI Engineer Mentorship — Session 14
## Production AI Observability: OpenTelemetry, Distributed Tracing, LLM Monitoring & Cost Optimization

**Level:** Intermediate → Advanced  
**Duration:** 60 minutes  
**Target role:** Senior AI Engineer  
**Today's project:** Build an observable RAG API using Python, FastAPI, OpenTelemetry, and Prometheus.

---

## 1. Where We Are in Your AI Engineering Journey

In our previous sessions, we progressively developed a production-oriented RAG application.

| Session | What we learned |
|---|---|
| 11 | Vector databases, pgvector, HNSW indexing |
| 12 | Hybrid search, BM25, RRF, cross-encoder reranking |
| 13 | RAG evaluation, hallucination detection, LLM-as-a-Judge |
| **14 — Today** | **Production observability, tracing, monitoring, and debugging** |

In Session 13, we learned how to evaluate whether an AI-generated answer is correct and supported by evidence.

Today, we'll solve a different problem.

Imagine you've deployed your RAG application to production.

A user reports:

> Your AI chatbot takes 15 seconds to answer a simple question.

You check your FastAPI logs:

```text
POST /chat HTTP/1.1 200 OK
```

The request succeeded.

But why did it take 15 seconds?

Was the problem caused by:

- Slow PostgreSQL vector search?
- Cross-encoder reranking?
- LLM inference?
- Network latency?
- An overloaded GPU?
- Too many input tokens?
- A downstream service retry?

Traditional HTTP access logs cannot answer all these questions.

This is where **AI observability** becomes essential.

A Senior AI Engineer must understand not only how to build AI applications, but also how to diagnose their behavior in production.

---

# 2. Learning Objectives

By the end of this session, you should be able to:

1. Explain the difference between logging, metrics, and distributed tracing.
2. Understand traces, spans, trace IDs, and parent-child relationships.
3. Instrument FastAPI applications using OpenTelemetry.
4. Measure latency across retrieval, reranking, and LLM inference.
5. Collect LLM token usage and estimate inference costs.
6. Expose application metrics using Prometheus.
7. Understand p50, p95, and p99 latency.
8. Debug production AI performance issues using traces.
9. Design observability for distributed AI applications.
10. Understand how observability connects to evaluation and LLMOps.

## ⏱️ Today's 60-Minute Agenda

| Time | Topic |
|---|---|
| 0–8 min | Understanding AI observability |
| 8–18 min | Distributed tracing and OpenTelemetry |
| 18–28 min | AI-specific metrics and monitoring |
| 28–45 min | Build an observable FastAPI RAG application |
| 45–53 min | Production debugging and cost optimization |
| 53–57 min | Senior AI Engineer system design |
| 57–60 min | Practical exercise, interview questions, and quiz |

---

# Part 1: What Is Observability?

**Time: 0–8 minutes**

Let's begin with a simple backend example.

Suppose you have a FastAPI application:

```python
@app.get("/users/{user_id}")
def get_user(user_id: int):
    return database.get_user(user_id)
```

If something fails, you might write:

```python
logger.error("Failed to fetch user")
```

Logging tells you that something happened.

But what if the database query takes five seconds without raising an error?

You need performance metrics.

And what if your request passes through five microservices?

You need distributed tracing.

These are the three major pillars of observability.

## 1.1 Logs

Logs record individual events.

Example:

```text
2026-10-23 10:00:01 INFO  Request received
2026-10-23 10:00:02 INFO  Documents retrieved
2026-10-23 10:00:03 INFO  Reranking completed
2026-10-23 10:00:08 INFO  LLM response generated
```

Logs are useful for answering:

**What happened?**

Examples include authentication failures, API exceptions, worker crashes, and unexpected model responses.

However, logs alone may not provide a complete view of a distributed request.

---

## 1.2 Metrics

Metrics represent numerical measurements collected over time.

Examples:

```text
Total requests: 10,000
Average response time: 1.8 seconds
Error rate: 2%
LLM input tokens: 850,000
LLM output tokens: 120,000
```

Metrics help answer:

**How is the system performing?**

For example, you might want to know whether the error rate increased during the last 30 minutes.

Or whether LLM token consumption doubled after a prompt change.

Metrics are especially useful for dashboards and alerts.

---

## 1.3 Distributed Traces

A trace represents the execution path of a request across operations and services.

Consider this RAG pipeline:

```text
                    User Request
                         |
                         v
                    FastAPI
                         |
                         v
                 Authentication
                         |
                         v
                  Query Embedding
                         |
                         v
                  Vector Search
                         |
                         v
                    Reranking
                         |
                         v
                   LLM Inference
                         |
                         v
                    Response
```

A distributed trace records the execution of these operations as related spans.

Example:

```text
Trace ID: abc123

POST /chat                  8,000 ms
│
├── Authentication             20 ms
│
├── Query Embedding           100 ms
│
├── Vector Search             180 ms
│
├── Reranking                 400 ms
│
└── LLM Inference           7,300 ms
```

Now the performance problem becomes clearer.

Most of the request time is spent generating the LLM response.

Optimizing the PostgreSQL index probably won't produce a meaningful improvement.

### Important distinction

| Observability component | Primary purpose |
|---|---|
| Logs | Understand individual events |
| Metrics | Monitor system behavior over time |
| Traces | Understand the execution path and latency of individual requests |

These components complement one another.

**Interview question:** What is the difference between monitoring and observability?

Monitoring typically involves tracking known indicators and alerting when they cross expected thresholds.

Observability is broader: it helps engineers investigate system behavior, including problems they did not anticipate when creating their dashboards.

---

# Part 2: Understanding Distributed Tracing

**Time: 8–18 minutes**

Before writing code, we need to understand four concepts.

## 2.1 What is a trace?

A trace represents the execution of one logical operation.

For example:

```http
POST /chat
```

The request may involve multiple services.

```text
FastAPI
   |
   v
Embedding Service
   |
   v
PostgreSQL
   |
   v
Reranking Service
   |
   v
LLM Server
```

All related operations should belong to the same trace.

Each trace has a unique trace ID.

Conceptually:

```text
trace_id = 4bf92f3577b34da6a3ce929d0e0e4736
```

This ID helps correlate operations belonging to the same request.

---

## 2.2 What is a span?

A span represents one operation within a trace.

For example:

```text
Trace: User asks a question

    Span 1: POST /chat
    Span 2: Generate query embedding
    Span 3: Search vector database
    Span 4: Rerank documents
    Span 5: Generate LLM response
```

Each span can contain:

- Span name.
- Start time.
- End time.
- Duration.
- Attributes.
- Events.
- Error status.
- Parent span ID.

Example:

```json
{
  "name": "rag.vector_search",
  "attributes": {
    "retrieval.top_k": 10,
    "retrieval.result_count": 8,
    "retrieval.strategy": "hybrid"
  },
  "status": "OK"
}
```

The duration is normally represented by the span's start and end timestamps rather than a manually assigned attribute.

---

## 2.3 Parent-child relationships

Suppose your FastAPI endpoint calls a retrieval function.

The HTTP request span is the parent.

The retrieval span is a child.

```text
POST /chat
    |
    ├── rag.retrieve
    |
    ├── rag.rerank
    |
    └── rag.generate
```

This hierarchy is important because it allows engineers to understand how an individual request spent its time.

For distributed services, the trace context must be propagated across service boundaries.

The W3C `traceparent` header is commonly used for HTTP trace-context propagation.

---

## 2.4 What is OpenTelemetry?

OpenTelemetry is an open-source observability framework that provides APIs, SDKs, instrumentation, and tools for collecting telemetry.

It supports traces, metrics, and logs, although the maturity of language-specific implementations varies.

Its Python SDK allows us to create spans and export telemetry to observability systems. [OpenTelemetry](https://opentelemetry.io/docs/languages/python/instrumentation/?utm_source=chatgpt.com)

Official guide: [OpenTelemetry Python Instrumentation](https://opentelemetry.io/docs/languages/python/instrumentation/?utm_source=chatgpt.com).

### Typical production architecture

```text
                   FastAPI
                      |
                      v
               OpenTelemetry SDK
                      |
                      v
                OTLP Exporter
                      |
                      v
             OpenTelemetry Collector
                      |
             ┌────────┴────────┐
             |                 |
             v                 v
         Trace Backend     Metrics Backend
             |                 |
             v                 v
         Trace Viewer       Dashboards
```

The OpenTelemetry Collector receives telemetry, processes it, and exports it to supported backends.

Using a Collector is a recommended approach for production deployments because it separates telemetry processing and routing from application code. [OpenTelemetry](https://opentelemetry.io/docs/languages/python/exporters/?utm_source=chatgpt.com)

---

# Part 3: AI-Specific Observability Metrics

**Time: 18–28 minutes**

Traditional backend applications commonly monitor:

- Request latency.
- CPU usage.
- Memory consumption.
- Database query duration.
- Error rate.
- Request throughput.

AI applications require additional metrics.

## 3.1 LLM inference latency

Consider an LLM request:

```python
response = llm.generate(prompt)
```

The total time might be:

```text
Prompt preparation:       100 ms
Network overhead:          50 ms
Model queueing:           800 ms
Prompt processing:        600 ms
Token generation:       3,000 ms
```

Total:

\[
T=100+50+800+600+3000
\]

\[
T=4550\text{ ms}
\]

That's approximately 4.55 seconds.

But the user experiences only the total duration unless we instrument the internal operations.

A production LLM system should ideally distinguish queueing, prompt processing, generation, and network time where the serving infrastructure exposes those measurements.

---

## 3.2 Time to First Token

**Time to First Token (TTFT)** measures how long it takes before the first output token becomes available.

Suppose:

```text
Request started:       10:00:00.000
First token generated: 10:00:01.200
Response completed:    10:00:05.000
```

Then:

\[
TTFT=1.2\text{ seconds}
\]

Total generation request latency:

\[
T=5\text{ seconds}
\]

TTFT is especially important for streaming applications.

Users may perceive an application as more responsive when the first token arrives quickly, even if the complete answer takes several seconds.

### Interview insight

**Why is TTFT different from total response latency?**

TTFT measures the initial wait before output begins.

Total latency measures the duration until the response completes.

A system can have good TTFT but poor total latency if it generates a very long response slowly.

---

## 3.3 Tokens per second

Token generation throughput is another important metric.

Suppose the model generates 200 output tokens in four seconds.

\[
Throughput=\frac{200}{4}
\]

\[
Throughput=50\text{ tokens/second}
\]

This is a simple average.

For streaming models, engineers often distinguish output-token throughput after the first token from total request throughput.

Throughput depends on hardware, model size, batch size, sequence length, concurrency, and inference configuration.

---

## 3.4 Token usage

LLM providers and serving frameworks may expose:

```json
{
  "input_tokens": 1200,
  "output_tokens": 350,
  "total_tokens": 1550
}
```

Why is this important?

Because token consumption affects inference cost and performance.

For example, a prompt change might increase average input tokens from 1,200 to 4,000.

Even if answer quality remains unchanged, operating costs and latency may increase.

---

## 3.5 LLM cost estimation

A simple usage-based cost formula is:

\[
Cost=
\frac{InputTokens}{1,000,000}\times InputRate
+
\frac{OutputTokens}{1,000,000}\times OutputRate
\]

For illustration, assume a hypothetical model charges:

- $0.50 per million input tokens.
- $1.50 per million output tokens.

For 1,200 input tokens and 350 output tokens:

\[
InputCost=\frac{1200}{1,000,000}\times0.50
\]

\[
InputCost=\$0.0006
\]

\[
OutputCost=\frac{350}{1,000,000}\times1.50
\]

\[
OutputCost=\$0.000525
\]

Total:

\[
\boxed{Cost=\$0.001125}
\]

These prices are hypothetical and are not quotations for any particular provider.

Actual billing can also involve cached tokens, reasoning tokens, batch discounts, infrastructure costs, and other provider-specific rules.

---

## 3.6 p50, p95, and p99 latency

This is an important Senior Engineer concept.

Suppose your API processes 1,000 requests.

The average response time is two seconds.

Does that mean most users receive answers in two seconds?

Not necessarily.

Consider:

| Metric | Example |
|---|---|
| p50 | 1.2 seconds |
| p95 | 4.5 seconds |
| p99 | 12 seconds |

**p50:** Approximately 50% of requests complete within this duration.

**p95:** Approximately 95% complete within this duration.

**p99:** Approximately 99% complete within this duration.

High percentiles help reveal slow requests that an average can hide.

For production AI applications, p95 and p99 latency are often more informative than average latency alone.

Prometheus supports histogram metrics, which can be used to estimate latency percentiles from bucket observations. [Prometheus](https://prometheus.github.io/client_python/instrumenting/histogram/?utm_source=chatgpt.com)

---

# Part 4: Build an Observable FastAPI RAG Application

**Time: 28–45 minutes**

Now we'll implement a working example.

We'll create a simplified RAG API with three stages:

```text
POST /chat
    |
    v
Retrieve Documents
    |
    v
Rerank Documents
    |
    v
Generate Answer
    |
    v
Return Response
```

We'll instrument each stage using OpenTelemetry.

We'll also expose Prometheus metrics.

**Important:** We'll simulate retrieval and generation so the project runs without an LLM API key, GPU, or vector database. After understanding the instrumentation, you can replace these functions with the real components from Sessions 11 and 12.

## Step 1: Project structure

Create:

```text
ai-observability/
    |
    ├── app.py
    └── requirements.txt
```

## Step 2: Install dependencies

Create `requirements.txt`:

```text
fastapi
uvicorn
opentelemetry-api
opentelemetry-sdk
opentelemetry-instrumentation-fastapi
prometheus-client
```

Install:

```bash
pip install -r requirements.txt
```

---

## Step 3: Configure OpenTelemetry

Create `app.py`.

```python
import time
import logging

from contextlib import contextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from opentelemetry import trace

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    ConsoleSpanExporter
)

from opentelemetry.trace import (
    Status,
    StatusCode
)

from opentelemetry.instrumentation.fastapi import (
    FastAPIInstrumentor
)

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from fastapi.responses import Response


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("rag-api")


resource = Resource.create({
    "service.name": "rag-observability-demo",
    "service.version": "1.0.0"
})

provider = TracerProvider(
    resource=resource
)

provider.add_span_processor(
    SimpleSpanProcessor(
        ConsoleSpanExporter()
    )
)

trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

app = FastAPI(
    title="Observable RAG API"
)
```

### Explanation

```python
TracerProvider(resource=resource)
```

Configures the service that produces traces.

```python
trace.get_tracer(__name__)
```

Provides a tracer for creating spans.

```python
ConsoleSpanExporter()
```

Prints completed spans to the console.

This is useful for local development.

For production, we'll replace the console exporter with a batched OTLP exporter.

---

## Step 4: Create Prometheus metrics

Add:

```python
REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total RAG requests",
    ["status"]
)

STAGE_DURATION = Histogram(
    "rag_stage_duration_seconds",
    "RAG pipeline stage duration",
    ["stage"],
    buckets=(
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        5,
        10
    )
)

STAGE_ERRORS = Counter(
    "rag_stage_errors_total",
    "Errors by RAG stage",
    ["stage"]
)

TOKEN_USAGE = Counter(
    "rag_llm_tokens_total",
    "LLM tokens processed",
    ["kind", "model"]
)
```

Let's understand the metrics.

**REQUEST_COUNT** tracks successful and failed requests.

**STAGE_DURATION** records the duration of each pipeline stage.

**STAGE_ERRORS** tracks failures in retrieval, reranking, or generation.

**TOKEN_USAGE** records input and output tokens by model.

### Why use labels?

Consider:

```python
STAGE_DURATION.labels(
    stage="retrieval"
)
```

Labels let us distinguish metrics belonging to different operations.

But be careful.

Do not use user IDs, request IDs, raw questions, or document IDs as Prometheus labels.

Those values can create extremely high metric cardinality.

Use a controlled set of values such as:

```text
retrieval
reranking
generation
```

---

## Step 5: Create a reusable tracing helper

Instead of manually measuring every function, we'll create a context manager.

```python
@contextmanager
def observe_stage(stage_name: str):

    start_time = time.perf_counter()

    with tracer.start_as_current_span(
        f"rag.{stage_name}"
    ) as span:

        try:
            yield span

        except Exception as exc:

            span.record_exception(exc)

            span.set_status(
                Status(StatusCode.ERROR)
            )

            STAGE_ERRORS.labels(
                stage=stage_name
            ).inc()

            raise

        finally:

            duration = (
                time.perf_counter() - start_time
            )

            STAGE_DURATION.labels(
                stage=stage_name
            ).observe(duration)
```

Now we can instrument any operation:

```python
with observe_stage("retrieval"):
    documents = retrieve_documents(query)
```

This automatically creates a span and records the operation's duration.

If the operation raises an exception, the helper also records the error.

### Why use `time.perf_counter()`?

As we discussed in your Python interview preparation, `perf_counter()` is appropriate for measuring elapsed durations.

It uses a monotonic performance counter and is not affected by wall-clock adjustments in the way `time.time()` can be.

---

## Step 6: Implement a simulated retrieval function

Add:

```python
DOCUMENTS = [
    {
        "id": "pg-pool",
        "content": (
            "PostgreSQL connection pool exhaustion "
            "occurs when all available connections "
            "are occupied."
        )
    },
    {
        "id": "pg-debug",
        "content": (
            "Inspect active database connections, "
            "long-running queries, and connection "
            "pool configuration."
        )
    }
]


def retrieve_documents(query: str):

    time.sleep(0.15)

    if any(
        term in query.lower()
        for term in ("postgres", "database", "connection")
    ):
        return DOCUMENTS

    return []
```

The `sleep()` call simulates retrieval latency.

In your actual application, replace this with the hybrid retrieval implementation from Session 12.

---

## Step 7: Implement simulated reranking

Add:

```python
def rerank_documents(query, documents):

    time.sleep(0.25)

    query_terms = set(
        query.lower().split()
    )

    return sorted(
        documents,
        key=lambda document: len(
            query_terms.intersection(
                document["content"].lower().split()
            )
        ),
        reverse=True
    )
```

This is a simple lexical overlap function, not a real cross-encoder.

It allows us to demonstrate how reranking latency appears in a trace.

In production, this function would call your cross-encoder model or reranking service.

---

## Step 8: Implement simulated LLM generation

Add:

```python
def generate_answer(query, documents):

    time.sleep(0.8)

    if not documents:

        answer = (
            "I don't have enough information "
            "to answer from the available documents."
        )

    else:

        answer = documents[0]["content"]

    # Simulated usage for demonstration.
    # Replace with actual provider-reported usage.
    usage = {
        "input_tokens": 120,
        "output_tokens": 25
    }

    return {
        "answer": answer,
        "usage": usage,
        "model": "mock-llm"
    }
```

We're using a mock model so the application works without an external service.

The token counts are illustrative.

In a real LLM integration, record the token usage returned by the provider or inference server.

---

## Step 9: Create the observable RAG endpoint

Now combine everything.

```python
class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000
    )


@app.post("/chat")
def chat(payload: ChatRequest):

    try:

        with observe_stage("retrieval") as span:

            documents = retrieve_documents(
                payload.question
            )

            span.set_attribute(
                "rag.retrieved_count",
                len(documents)
            )

        with observe_stage("reranking") as span:

            ranked_documents = rerank_documents(
                payload.question,
                documents
            )

            span.set_attribute(
                "rag.reranked_count",
                len(ranked_documents)
            )

        with observe_stage("generation") as span:

            result = generate_answer(
                payload.question,
                ranked_documents
            )

            usage = result["usage"]

            span.set_attribute(
                "gen_ai.request.model",
                result["model"]
            )

            span.set_attribute(
                "gen_ai.usage.input_tokens",
                usage["input_tokens"]
            )

            span.set_attribute(
                "gen_ai.usage.output_tokens",
                usage["output_tokens"]
            )

            for kind, count in (
                ("input", usage["input_tokens"]),
                ("output", usage["output_tokens"])
            ):

                TOKEN_USAGE.labels(
                    kind=kind,
                    model=result["model"]
                ).inc(count)

        REQUEST_COUNT.labels(
            status="success"
        ).inc()

        current_span = trace.get_current_span()

        trace_id = format(
            current_span.get_span_context().trace_id,
            "032x"
        )

        logger.info(
            "RAG request completed trace_id=%s",
            trace_id
        )

        return {
            "answer": result["answer"],
            "sources": [
                doc["id"]
                for doc in ranked_documents
            ],
            "trace_id": trace_id
        }

    except Exception:

        REQUEST_COUNT.labels(
            status="error"
        ).inc()

        logger.exception(
            "RAG request failed"
        )

        raise
```

### What happens when the endpoint runs?

```text
POST /chat
    |
    v
OpenTelemetry HTTP Span
    |
    ├── rag.retrieval
    |       |
    |       └── Record retrieval duration
    |
    ├── rag.reranking
    |       |
    |       └── Record reranking duration
    |
    └── rag.generation
            |
            ├── Record generation duration
            └── Record token usage
```

Each stage produces a child span.

The HTTP instrumentation supplies the parent request span.

---

## Step 10: Enable FastAPI instrumentation

Add this after defining your routes:

```python
FastAPIInstrumentor.instrument_app(app)
```

This instruments incoming FastAPI requests.

The manual spans created inside the endpoint become children of the request span.

The instrumentation package is part of the OpenTelemetry Python ecosystem and supports FastAPI request tracing. [PyPI](https://pypi.org/project/opentelemetry-instrumentation-fastapi/0.61b0/?utm_source=chatgpt.com)

---

## Step 11: Expose Prometheus metrics

Add:

```python
@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

This exposes the metrics collected by the application.

In production, restrict access to the metrics endpoint to your monitoring infrastructure.

Also note that this simple example assumes a single application process. Multiple Uvicorn workers require an appropriate multiprocess metrics strategy or a different collection architecture.

---

## Step 12: Run the application

Start FastAPI:

```bash
uvicorn app:app --port 8000
```

Send a request:

```bash
curl -X POST \
  http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Why are my PostgreSQL connections exhausted?"
  }'
```

Example response:

```json
{
  "answer": "PostgreSQL connection pool exhaustion occurs when all available connections are occupied.",
  "sources": [
    "pg-pool",
    "pg-debug"
  ],
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736"
}
```

The actual trace ID will be generated dynamically.

Your terminal will also show exported spans.

Conceptually:

```text
POST /chat
    |
    ├── rag.retrieval      ~150 ms
    |
    ├── rag.reranking      ~250 ms
    |
    └── rag.generation     ~800 ms
```

Total application processing time will be approximately 1.2 seconds plus framework overhead.

These are expected illustrative timings, not benchmark results.

---

## Step 13: Inspect Prometheus metrics

Open:

```text
http://localhost:8000/metrics
```

You should see metrics similar to:

```text
rag_requests_total{status="success"} 1.0

rag_llm_tokens_total{
    kind="input",
    model="mock-llm"
} 120.0

rag_llm_tokens_total{
    kind="output",
    model="mock-llm"
} 25.0
```

The actual Prometheus exposition format uses each sample on a single line.

You will also see histogram bucket, sum, and count series for stage durations.

### Example PromQL query

To calculate p95 retrieval latency over five minutes:

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(
      rag_stage_duration_seconds_bucket{
        stage="retrieval"
      }[5m]
    )
  )
)
```

This estimates the 95th-percentile retrieval latency from histogram buckets.

It requires Prometheus to scrape the metrics endpoint over time.

For a meaningful result, the system must have enough observations.

---

# Part 5: Production Debugging and Cost Optimization

**Time: 45–53 minutes**

Let's move beyond the demonstration.

Imagine you're working as a Senior AI Engineer.

Your company reports that AI chatbot latency increased significantly after a deployment.

Previously:

```text
p95 latency: 2.5 seconds
```

After deployment:

```text
p95 latency: 8.0 seconds
```

How would you investigate?

## 5.1 Examine the traces

Suppose a representative slow request shows:

| Operation | Before | After |
|---|---:|---:|
| Query embedding | 100 ms | 120 ms |
| Vector search | 150 ms | 180 ms |
| Reranking | 250 ms | 300 ms |
| LLM generation | 1,800 ms | 7,100 ms |

The biggest increase is in LLM generation.

That gives us a direction for investigation.

It doesn't yet tell us the root cause.

We should examine:

- Input token count.
- Output token count.
- Model version.
- GPU utilization.
- Request concurrency.
- Queueing duration.
- KV cache utilization.
- Retry behavior.

### Example finding

Suppose the previous prompt contained:

```text
1,000 input tokens
```

After deployment:

```text
8,000 input tokens
```

The developer may have increased the number of retrieved chunks or changed the prompt template.

That could explain increased prompt-processing time and token costs.

The next step would be to compare the prompt construction and retrieval configuration between deployments.

### Important nuance

Do not simply add the p95 latencies of individual stages and assume the result equals end-to-end p95.

Percentiles are not generally additive.

Also, operations running concurrently may overlap.

Distributed traces help you inspect the actual execution timeline.

---

## 5.2 Implement LLM cost tracking

Now let's create a reusable cost-calculation function.

```python
def calculate_llm_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float
) -> float:

    input_cost = (
        input_tokens / 1_000_000
    ) * input_price_per_million

    output_cost = (
        output_tokens / 1_000_000
    ) * output_price_per_million

    return input_cost + output_cost
```

Example:

```python
cost = calculate_llm_cost(
    input_tokens=1200,
    output_tokens=350,
    input_price_per_million=0.50,
    output_price_per_million=1.50
)

print(f"Estimated cost: ${cost:.6f}")
```

Output:

```text
Estimated cost: $0.001125
```

### Production considerations

Your cost-tracking service should maintain versioned pricing configurations.

Different models may have different input and output rates.

Some providers also distinguish cached tokens from uncached tokens.

For self-hosted models, you may need to allocate infrastructure expenses across requests instead of relying on per-token provider pricing.

---

## 5.3 Metrics that matter for LLM applications

| Metric | Why it matters |
|---|---|
| End-to-end latency | User experience |
| Time to first token | Streaming responsiveness |
| Output tokens/second | Generation throughput |
| Input tokens | Prompt size and cost |
| Output tokens | Response length and cost |
| Retrieval latency | Search performance |
| Reranking latency | Ranking overhead |
| GPU utilization | Hardware efficiency |
| Queueing latency | Capacity pressure |
| Error rate | Reliability |
| Timeout rate | Availability |
| Estimated cost/request | Financial efficiency |
| Groundedness on evaluated samples | Answer reliability |

Notice the final metric.

This connects directly to Session 13.

**Observability tells you what your system is doing. Evaluation helps determine whether the AI output is good.**

A mature production AI platform needs both.

---

# Part 6: Production Observability Architecture

**Time: 53–57 minutes**

Let's design an observability architecture for a real enterprise RAG system.

Suppose your application uses:

- FastAPI.
- PostgreSQL and pgvector.
- Redis.
- Celery workers.
- An embedding service.
- A cross-encoder reranker.
- A self-hosted LLM.
- Kubernetes.

Your architecture might look like this:

```text
                         USER
                           |
                           v
                      API GATEWAY
                           |
                           v
                         FastAPI
                           |
                ┌──────────┴──────────┐
                |                     |
                v                     v
          Retrieval Service      Ingestion API
                |                     |
                v                     v
              pgvector             Redis Queue
                |                     |
                v                     v
          Reranking Service       Celery Worker
                |
                v
             LLM Server
                |
                v
             Response


              OBSERVABILITY LAYER

       FastAPI ───────────────┐
       Retrieval Service ─────┤
       Celery Workers ────────┤
       Reranking Service ─────┤
       LLM Server ────────────┤
                             |
                             v
                    OpenTelemetry SDK
                             |
                             v
                    OpenTelemetry Collector
                             |
               ┌─────────────┼─────────────┐
               |             |             |
               v             v             v
             Traces        Metrics         Logs
               |             |             |
               v             v             v
             Jaeger       Prometheus    Log Backend
               |             |             |
               └─────────────┼─────────────┘
                             |
                             v
                       Dashboards
                             |
                             v
                          Alerts
```

Here, Jaeger can be used for distributed trace visualization, while Prometheus stores and queries time-series metrics.

For production, telemetry should normally flow through an OpenTelemetry Collector.

### Exporting traces through OTLP

Install:

```bash
pip install opentelemetry-exporter-otlp-proto-http
```

Replace the console span processor with:

```python
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor
)

from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter
)


provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint=(
                "http://localhost:4318/v1/traces"
            )
        )
    )
)
```

Use this instead of the earlier console processor when exporting to a locally running OTLP HTTP Collector.

The Collector must be configured to receive OTLP HTTP traffic on port 4318.

The exporter follows the OpenTelemetry Python OTLP implementation. [OpenTelemetry Python](https://opentelemetry-python.readthedocs.io/en/latest/exporter/otlp/otlp.html?utm_source=chatgpt.com)

Official guide: [OpenTelemetry Exporters](https://opentelemetry.io/docs/languages/python/exporters/?utm_source=chatgpt.com).

---

## 6.1 Distributed tracing across Celery workers

Here's a scenario particularly relevant to your Python backend background.

Your FastAPI application receives a document upload.

It sends an ingestion task to Celery.

```text
FastAPI
    |
    v
Redis Queue
    |
    v
Celery Worker
    |
    v
Document Parsing
    |
    v
Embedding Generation
    |
    v
pgvector
```

If you trace only the FastAPI request, you won't automatically see the entire asynchronous ingestion process.

You need trace-context propagation across the queue boundary.

This typically involves injecting trace context into task metadata when publishing and extracting it in the worker.

OpenTelemetry instrumentation can automate parts of this process, but propagation must be verified for your Celery and messaging configuration.

### Important distinction

A background task may continue long after the HTTP request has completed.

The task's execution span should represent the worker's actual processing time, not artificially extend the HTTP request.

For long-running asynchronous workflows, span links or carefully designed parent-child relationships can preserve causal relationships without misrepresenting execution.

---

## 6.2 Observability security

AI applications may process confidential information.

Consider:

```python
span.set_attribute(
    "user.question",
    payload.question
)
```

This might accidentally store:

- Personal information.
- Internal company documents.
- Passwords or API keys.
- Confidential business data.

A safer default is to record operational metadata rather than raw content.

For example:

```python
span.set_attribute(
    "rag.question_length",
    len(payload.question)
)

span.set_attribute(
    "rag.retrieved_count",
    len(documents)
)
```

Avoid recording complete prompts, retrieved documents, and generated answers by default.

If content-level tracing is necessary, apply explicit access controls, redaction, retention limits, and appropriate data-handling policies.

---

# Part 7: Today's Practical Exercise

**Time: 57–60 minutes**

## Mini-Project: Build an AI Observability Dashboard

Your assignment is to extend the RAG application we created today.

### Task 1: Instrument your real RAG pipeline

Replace the simulated retrieval and reranking functions with your implementations from Sessions 11 and 12.

Create spans for:

```text
query_embedding
bm25_search
vector_search
rrf_fusion
reranking
llm_generation
```

Record the duration of each operation.

### Task 2: Track token usage

Record input and output tokens for every LLM request.

Calculate estimated cost using a versioned pricing configuration.

### Task 3: Add metrics

Expose:

```text
rag_requests_total
rag_stage_duration_seconds
rag_stage_errors_total
rag_llm_tokens_total
```

### Task 4: Add an artificial delay

Temporarily add:

```python
time.sleep(3)
```

inside the reranking function.

Send multiple requests.

Inspect the traces and determine which operation caused the slowdown.

### Task 5: Design a dashboard

Create dashboard panels for:

| Panel | Metric |
|---|---|
| Request volume | Requests per second |
| API latency | p50, p95, p99 |
| Retrieval latency | p95 |
| Reranking latency | p95 |
| LLM latency | p95 |
| Error rate | Failed requests / total requests |
| Token usage | Input and output tokens |
| Cost | Estimated cost per request |

### Bonus challenge

Instrument a Celery task.

Propagate trace context from FastAPI to the worker and verify that the asynchronous ingestion job can be correlated with the originating request.

---

# Part 8: Senior AI Engineer Interview Preparation

### Q1. What is observability?

Observability is the ability to understand a system's internal behavior using information such as logs, metrics, and traces.

It helps engineers investigate performance problems, failures, and unexpected system behavior.

### Q2. What is the difference between a trace and a span?

A trace represents an end-to-end operation.

A span represents an individual operation within that trace.

Multiple related spans form a trace.

### Q3. Why is distributed tracing important for AI applications?

Because AI requests often involve multiple components, including embedding services, vector databases, rerankers, model servers, and external APIs.

Distributed tracing helps identify where time is spent and where failures occur.

### Q4. What is OpenTelemetry?

OpenTelemetry is an open-source framework for generating, collecting, and exporting telemetry such as traces, metrics, and logs.

It helps standardize instrumentation across services.

### Q5. What is p95 latency?

p95 latency is the duration at or below which approximately 95% of measured requests complete.

It helps reveal slow requests that average latency may hide.

### Q6. What is TTFT?

Time to First Token measures the delay between submitting a generation request and receiving the first output token.

It is particularly useful for monitoring streaming LLM applications.

### Q7. How would you debug slow LLM responses?

I would inspect distributed traces, token usage, queueing time, prompt-processing latency, output generation speed, model version, hardware utilization, and request concurrency.

I would identify the dominant bottleneck before attempting optimizations.

### Q8. How is AI observability different from traditional application monitoring?

AI observability includes traditional metrics such as latency and error rates, but also monitors model-specific behavior.

Examples include token consumption, TTFT, generation throughput, retrieval quality, model version, estimated inference cost, and evaluated answer quality.

---

# 🧠 Today's Knowledge Quiz

Try answering these questions before checking the lesson again.

**Question 1**

What are the three major pillars of observability?

Explain the purpose of each.

**Question 2**

Your RAG API takes eight seconds to respond.

A trace shows:

```text
Embedding:       100 ms
Vector Search:   200 ms
Reranking:       300 ms
LLM Generation: 7,200 ms
```

Which component would you investigate first?

What additional metrics would you examine?

**Question 3**

What is the difference between a trace ID and a span ID?

**Question 4**

Your LLM generates 300 output tokens in six seconds.

What is the average output-token throughput?

**Question 5**

Why is `time.perf_counter()` preferable to `time.time()` for measuring elapsed execution durations?

**Question 6**

Why should you avoid using user IDs and raw questions as Prometheus metric labels?

**Question 7 — Advanced System Design**

Your company operates an AI assistant handling 100,000 requests per day.

The architecture contains:

```text
FastAPI
   |
   v
Embedding Service
   |
   v
PostgreSQL + pgvector
   |
   v
Reranking Service
   |
   v
LLM Inference Server
```

Users report occasional 20-second response times.

However, the average response time is only three seconds.

As a Senior AI Engineer, explain:

1. Which latency metrics you would examine.
2. How distributed tracing would help.
3. Which LLM-specific metrics you would collect.
4. How you would identify GPU or queueing bottlenecks.
5. How you would detect cost increases after a deployment.
6. How you would protect sensitive user data in traces.
7. How you would configure alerts without generating excessive false alarms.

---

# Session 14 Recap

Today, we learned how to make AI applications observable in production.

We covered logs, metrics, distributed traces, OpenTelemetry, Prometheus, latency percentiles, token monitoring, and cost estimation.

We also built a FastAPI application that generates traces for retrieval, reranking, and LLM generation.

Our observability architecture is:

\[
\boxed{
\text{AI Application}
\rightarrow
\text{Instrumentation}
\rightarrow
\text{Telemetry Collection}
\rightarrow
\text{Analysis}
\rightarrow
\text{Debugging}
}
\]

The most important lesson is:

**A production AI application must be measurable and diagnosable.**

When a user reports that an AI assistant is slow, expensive, or unreliable, a Senior AI Engineer should be able to investigate using traces, metrics, logs, and evaluation results rather than relying on guesswork.

---

## 🔜 Session 15: Production AI Security — Prompt Injection, Data Leakage, Tool Permissions & Secure RAG Architecture

Our next session will focus on securing AI applications.

We'll study prompt injection attacks, direct versus indirect prompt injection, insecure agent tool execution, RAG data leakage, tenant isolation, secret management, output validation, and defense-in-depth.

**Practical project:** Build a secure FastAPI-based RAG endpoint with document-level authorization, restricted tool access, input validation, and security-focused tests.

We'll also examine how to design AI agents that can safely interact with external APIs, databases, and internal company systems.

**Today's priority:** Run the observable FastAPI application, inspect the generated spans, and understand how tracing reveals the source of latency. This is a fundamental skill for operating production AI systems.