# 🎓 AI Engineer Mentorship — Session 7
## Agent Evaluation, Observability, Tracing & Reliability Engineering

**Level:** Intermediate → Advanced  
**Duration:** 60 minutes  
**Today's project:** Build an evaluation framework to test whether AI agents execute the correct tools, follow security rules, and recover from failures.

In our previous session, we built a stateful incident-investigation workflow using LangGraph.

We implemented checkpointing, human approval, conditional routing, and idempotent ticket creation.

Now imagine deploying that agent to production.

It works correctly during development, but after deployment, users report that it sometimes selects the wrong tools, repeatedly checks the same service, or creates incident tickets unnecessarily.

Your FastAPI application returns HTTP 200. Your database is healthy. Your workers are running.

**But your AI agent is behaving incorrectly.**

How do you identify and prevent these failures?

That's what we're learning today.

---

# 1. Learning Objectives

By the end of this session, you should be able to:

1. Explain why traditional software testing is insufficient for AI agents.
2. Understand agent trajectory evaluation.
3. Measure tool-selection accuracy and task-completion rates.
4. Build a Python evaluation framework.
5. Understand distributed tracing for LLM and agent applications.
6. Test agent behavior under simulated failures.
7. Design production evaluation and observability pipelines.
8. Establish measurable quality requirements before deployment.

### Today's 60-Minute Agenda

| Time | Topic |
|---|---|
| 0–10 min | Why AI agents need specialized evaluation |
| 10–20 min | Agent trajectories and evaluation metrics |
| 20–40 min | Build an agent evaluation framework in Python |
| 40–48 min | Observability, tracing, and debugging |
| 48–53 min | Failure injection and production reliability |
| 53–60 min | Practical assignment, interview questions, and quiz |

---

# Part 1: Why Traditional Testing Is Not Enough

**Time: 0–10 minutes**

As a Python backend developer, you're familiar with unit tests.

Consider:

```python
def calculate_total(price, quantity):
    return price * quantity
```

We can test this easily:

```python
def test_calculate_total():
    assert calculate_total(100, 5) == 500
```

For the same inputs, the function should produce the same result.

This is deterministic behavior.

Now consider an AI agent.

**User:**

> Our PostgreSQL service is responding slowly. Investigate the issue.

Your agent has three tools:

```python
get_service_health()
get_recent_errors()
search_documentation()
```

On the first execution, the agent might call:

```text
1. get_service_health("postgres")
2. get_recent_errors("postgres")
3. search_documentation("postgres connection pool")
```

On another execution, it might call:

```text
1. search_documentation("postgres slow queries")
2. get_service_health("postgres")
3. get_recent_errors("postgres")
```

Both sequences might be reasonable.

This introduces a testing challenge.

**How do we evaluate behavior when multiple execution paths can be valid?**

We need to evaluate more than the final answer.

## Three levels of agent evaluation

| Level | What we evaluate | Example |
|---|---|---|
| Output | Was the final answer correct? | Correctly identifies connection pressure |
| Trajectory | Did the agent perform appropriate steps? | Inspected PostgreSQL metrics |
| Safety | Did the agent obey execution boundaries? | Did not create a ticket without approval |

These levels are related but independent.

An agent might produce a correct answer while performing unnecessary operations.

It might also execute the correct tools but misunderstand their results.

### Example

Suppose an agent receives:

```json
{
  "service": "postgres",
  "connection_usage_percent": 94,
  "status": "warning"
}
```

It responds:

> PostgreSQL is completely unavailable.

This is incorrect.

The tool returned a warning, not an outage.

The tool execution succeeded, but the agent's interpretation failed.

**Senior AI Engineer principle:** Evaluate the execution process and the final outcome separately.

---

# Part 2: Agent Trajectories and Evaluation Metrics

**Time: 10–20 minutes**

## What is an agent trajectory?

A trajectory is a record of the actions and observations produced during an agent execution.

For example:

```text
User Question
     |
     v
LLM selects get_service_health
     |
     v
Tool returns connection usage: 94%
     |
     v
LLM selects get_recent_errors
     |
     v
Tool returns DB_POOL_TIMEOUT: 25 occurrences
     |
     v
LLM generates investigation summary
     |
     v
Final Answer
```

A structured representation might look like:

```python
trajectory = [
    {
        "step": 1,
        "type": "tool_call",
        "tool": "get_service_health",
        "arguments": {"service": "postgres"}
    },
    {
        "step": 2,
        "type": "tool_result",
        "status": "success"
    },
    {
        "step": 3,
        "type": "tool_call",
        "tool": "get_recent_errors",
        "arguments": {"service": "postgres"}
    },
    {
        "step": 4,
        "type": "final_answer",
        "status": "completed"
    }
]
```

Now we can evaluate what happened.

## Metric 1: Task Success Rate

Measures how frequently an agent successfully completes assigned tasks.

\[
SuccessRate=
\frac{\text{Successfully completed tasks}}
{\text{Total evaluated tasks}}
\]

Suppose you test 100 incident investigations.

If 87 meet your success criteria:

\[
SuccessRate=\frac{87}{100}=87\%
\]

But we need a precise definition of success.

For example, a successful investigation might require an evidence-supported conclusion and no unauthorized actions.

Merely generating an answer is not enough.

## Metric 2: Tool Selection Accuracy

Measures whether the agent selects appropriate tools.

Suppose the user asks:

> Check Redis health.

Expected tool:

```python
get_service_health("redis")
```

But the agent selects:

```python
search_documentation("redis health")
```

That may be useful background research, but it doesn't inspect the current service state.

We can evaluate whether the expected tool was called.

For a dataset with one required tool per query:

\[
ToolSelectionAccuracy=
\frac{\text{Cases selecting the required tool}}
{\text{Total cases}}
\]

For more complex tasks, a single expected-tool metric is insufficient. We may need to evaluate sets of required tools, their arguments, and execution dependencies.

## Metric 3: Trajectory Correctness

Consider two trajectories.

**Expected:**

```text
Check Service
    ↓
Check Errors
    ↓
Generate Summary
```

**Actual:**

```text
Check Errors
    ↓
Check Service
    ↓
Generate Summary
```

Should the agent fail?

Not necessarily.

If the tools are independent, both orders may be acceptable.

This leads to three evaluation approaches.

| Approach | Description |
|---|---|
| Exact match | Require an identical sequence of actions |
| Order-aware match | Require certain actions to occur in a specified order |
| Constraint-based | Require or forbid actions without demanding one exact sequence |

For production agents, constraint-based evaluation is often more practical.

For example:

```text
Required:
- Check service health.
- Inspect recent errors.

Forbidden:
- Create a ticket without approval.

Allowed:
- Search documentation at any point.
```

This evaluates meaningful behavior without requiring one rigid execution path.

## Metric 4: Safety Violation Rate

Measures how frequently the agent violates a safety requirement.

Examples include unauthorized tool execution, accessing another tenant's information, or performing a sensitive operation without approval.

\[
ViolationRate=
\frac{\text{Executions with safety violations}}
{\text{Total executions}}
\]

For critical actions, the target should be zero observed violations in testing.

However, zero failures in a finite test dataset does not prove that a system is completely safe.

## Metric 5: Latency and Cost

Agents can execute several model calls and tools.

Therefore, total latency may be much greater than that of a single LLM request.

\[
T_{total}\approx
T_{model}+T_{tools}+T_{orchestration}+T_{waiting}
\]

For sequential execution, these components contribute to the critical path. Parallel tool execution changes how durations combine.

Useful measurements include p50 latency, p95 latency, token usage, model cost, tool-call count, and retry count.

---

# Part 3: Build an Agent Evaluation Framework

**Time: 20–40 minutes**

We're going to create a Python evaluation harness for our incident-investigation agent.

For this exercise, we'll use a deterministic simulator instead of calling a real LLM.

Why?

Because we want to verify that our evaluation framework works before connecting it to unpredictable model behavior.

Once the harness is working, we can connect it to the LangGraph agent from Session 6.

## Step 1: Define the execution result

Create a file named `agent_evaluation.py`.

```python
from dataclasses import dataclass
from time import perf_counter


@dataclass
class ToolStep:
    name: str
    success: bool
    duration_ms: float


@dataclass
class AgentResult:
    status: str
    steps: list[ToolStep]
    answer: str
    total_ms: float
```

This defines a structured representation of an agent execution.

For example:

```python
AgentResult(
    status="completed",
    steps=[
        ToolStep(
            name="get_service_health",
            success=True,
            duration_ms=25.0
        )
    ],
    answer="Simulated service health checked.",
    total_ms=30.0
)
```

In a real application, you would also store a run ID, trace ID, model version, prompt version, tool arguments, and error categories.

Avoid recording credentials or sensitive data unnecessarily.

## Step 2: Create a simulated agent

```python
def run_simulated_agent(
    intent: str,
    approved: bool = False,
    fail_tool: str | None = None
) -> AgentResult:

    start = perf_counter()
    steps = []

    plans = {
        "health": [
            "get_service_health"
        ],
        "diagnose": [
            "get_service_health",
            "get_recent_errors",
            "search_documentation"
        ],
        "ticket": [
            "get_service_health",
            "get_recent_errors"
        ]
    }

    if intent not in plans:
        raise ValueError("Unsupported intent")

    plan = plans[intent].copy()

    if intent == "ticket" and approved:
        plan.append("create_ticket")

    for tool_name in plan:

        tool_start = perf_counter()

        # Simulate a tool failure when requested.
        success = tool_name != fail_tool

        duration_ms = (
            perf_counter() - tool_start
        ) * 1000

        steps.append(
            ToolStep(
                name=tool_name,
                success=success,
                duration_ms=duration_ms
            )
        )

        if not success:
            return AgentResult(
                status="failed",
                steps=steps,
                answer=(
                    "Investigation incomplete: "
                    f"{tool_name} failed."
                ),
                total_ms=(
                    perf_counter() - start
                ) * 1000
            )

    if intent == "ticket" and not approved:
        status = "awaiting_approval"
        answer = "Ticket creation requires approval."
    else:
        status = "completed"
        answer = "Simulated investigation completed."

    return AgentResult(
        status=status,
        steps=steps,
        answer=answer,
        total_ms=(
            perf_counter() - start
        ) * 1000
    )
```

This is deliberately not an LLM-powered agent.

It simulates execution paths so we can test our evaluation logic.

Notice that it supports three scenarios: checking service health, investigating an issue, and creating a ticket after approval.

It also supports injecting a failure into a selected tool.

The simulator does not perform real monitoring or ticket creation. Its timing measurements are therefore not representative of production agent latency.

---

## Step 3: Create an evaluation dataset

Now define expected behavior.

```python
TEST_CASES = [
    {
        "name": "Health check",
        "intent": "health",
        "approved": False,
        "fail_tool": None,
        "required": [
            "get_service_health"
        ],
        "forbidden": [
            "create_ticket"
        ],
        "expected_status": "completed"
    },
    {
        "name": "Incident investigation",
        "intent": "diagnose",
        "approved": False,
        "fail_tool": None,
        "required": [
            "get_service_health",
            "get_recent_errors"
        ],
        "forbidden": [
            "create_ticket"
        ],
        "expected_status": "completed"
    },
    {
        "name": "Ticket without approval",
        "intent": "ticket",
        "approved": False,
        "fail_tool": None,
        "required": [
            "get_service_health"
        ],
        "forbidden": [
            "create_ticket"
        ],
        "expected_status": "awaiting_approval"
    },
    {
        "name": "Approved ticket",
        "intent": "ticket",
        "approved": True,
        "fail_tool": None,
        "required": [
            "get_service_health",
            "create_ticket"
        ],
        "forbidden": [],
        "expected_status": "completed"
    },
    {
        "name": "Tool failure",
        "intent": "diagnose",
        "approved": False,
        "fail_tool": "get_recent_errors",
        "required": [
            "get_service_health",
            "get_recent_errors"
        ],
        "forbidden": [
            "create_ticket"
        ],
        "expected_status": "failed"
    }
]
```

Each test defines an execution scenario and the behavior we expect.

This is similar to a unit-test dataset, but instead of checking only a return value, we're checking the agent's actions.

---

## Step 4: Implement trajectory evaluation

First, create a function to verify whether required tools appeared in the expected order.

```python
def is_subsequence(
    required: list[str],
    actual: list[str]
) -> bool:

    iterator = iter(actual)

    return all(
        any(tool == expected for tool in iterator)
        for expected in required
    )
```

For example:

```python
required = [
    "get_service_health",
    "get_recent_errors"
]

actual = [
    "get_service_health",
    "search_documentation",
    "get_recent_errors"
]

print(is_subsequence(required, actual))
```

Output:

```text
True
```

The additional documentation search is permitted.

But this would fail:

```python
actual = [
    "get_recent_errors",
    "get_service_health"
]
```

Because the required order was not followed.

Remember that order requirements should reflect actual dependencies. Don't impose an ordering constraint when independent tool calls can safely execute in either order.

## Step 5: Evaluate an execution

```python
def evaluate_case(case: dict) -> dict:

    result = run_simulated_agent(
        intent=case["intent"],
        approved=case["approved"],
        fail_tool=case["fail_tool"]
    )

    executed_tools = [
        step.name
        for step in result.steps
    ]

    required_ok = is_subsequence(
        case["required"],
        executed_tools
    )

    forbidden_ok = not any(
        tool in executed_tools
        for tool in case["forbidden"]
    )

    status_ok = (
        result.status == case["expected_status"]
    )

    passed = (
        required_ok
        and forbidden_ok
        and status_ok
    )

    return {
        "test": case["name"],
        "passed": passed,
        "status": result.status,
        "tools": executed_tools,
        "latency_ms": round(
            result.total_ms, 3
        )
    }
```

This evaluator checks three things.

First, whether required tools were executed in the expected order.

Second, whether forbidden tools were avoided.

Third, whether the workflow reached the expected status.

For simplicity, the harness doesn't yet evaluate tool arguments or answer correctness.

We'll add those later.

## Step 6: Run all evaluations

```python
def run_evaluation_suite():

    results = []

    for case in TEST_CASES:

        result = evaluate_case(case)

        results.append(result)

        print(
            f"{result['test']}: "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )

        print("Status:", result["status"])
        print("Tools:", result["tools"])
        print()

    passed = sum(
        result["passed"]
        for result in results
    )

    total = len(results)

    success_rate = passed / total

    print(
        f"Evaluation Pass Rate: "
        f"{success_rate:.1%}"
    )


if __name__ == "__main__":
    run_evaluation_suite()
```

Run:

```bash
python agent_evaluation.py
```

Expected output:

```text
Health check: PASS
Incident investigation: PASS
Ticket without approval: PASS
Approved ticket: PASS
Tool failure: PASS

Evaluation Pass Rate: 100.0%
```

**Important:** This 100% result only means our deterministic simulator satisfied five predefined tests.

It does not demonstrate that a real LLM agent is 100% reliable.

The purpose of this exercise is to build the evaluation infrastructure.

---

# Part 4: Connect Evaluation to a Real LangGraph Agent

**Time: 40–48 minutes**

Now let's think about our Session 6 LangGraph application.

Previously, we executed the graph using:

```python
result = graph.invoke(
    {
        "service": "postgres",
        "status": "new"
    },
    config={
        "configurable": {
            "thread_id": "incident-1001"
        }
    }
)
```

But how do we inspect every node execution?

One approach is to use LangGraph's streaming capabilities.

For example:

```python
config = {
    "configurable": {
        "thread_id": "evaluation-run-001"
    }
}

events = []

for update in graph.stream(
    {
        "service": "postgres",
        "status": "new"
    },
    config=config,
    stream_mode="updates"
):
    events.append(update)

print(events)
```

This produces state-update events from the graph.

You can extract node names and build a simplified execution trace.

For example:

```python
executed_nodes = []

for event in events:
    executed_nodes.extend(event.keys())

print(executed_nodes)
```

A possible result:

```python
[
    "inspect_service",
    "analyze_results"
]
```

The workflow may then pause for approval.

You can evaluate whether the expected nodes executed before the interrupt.

**Important distinction:** State-update streams are useful for examining graph execution, but they are not a complete distributed trace. Detailed LLM calls, tool arguments, nested operations, and timing measurements require additional instrumentation.

## What is distributed tracing?

In traditional backend development, you might have:

```text
FastAPI
   ↓
Django Service
   ↓
PostgreSQL
```

A distributed trace helps you follow a request across services.

In AI engineering, the execution path becomes more complicated.

```text
FastAPI Request
       |
       v
Agent Execution
       |
       ├── LLM Call
       |
       ├── Tool: Search Documentation
       |          |
       |          v
       |      Vector Database
       |
       ├── LLM Call
       |
       ├── Tool: Get Service Health
       |
       └── Final LLM Response
```

Each operation can be represented by a **span**.

A span typically contains an operation name, start and end times, status, and attributes.

For example:

```text
Trace ID: abc123

agent.run                    4200 ms
    |
    ├── llm.generate         1200 ms
    |
    ├── search_documents      180 ms
    |
    ├── llm.generate         1600 ms
    |
    ├── get_service_health     90 ms
    |
    └── llm.generate         1130 ms
```

The numbers are illustrative.

This trace allows you to identify which operations contribute most to latency.

## OpenTelemetry

OpenTelemetry provides common standards and tooling for collecting traces, metrics, and logs.

Its semantic conventions define consistent names for telemetry attributes across systems. Generative-AI-specific conventions also cover model operations and token usage, although some of these conventions are still evolving. [OpenTelemetry](https://opentelemetry.io/docs/concepts/semantic-conventions/?utm_source=chatgpt.com)

You can explore the [OpenTelemetry documentation](https://opentelemetry.io/docs/?utm_source=chatgpt.com).

## LangSmith

LangSmith is another option for tracing and evaluating LLM applications.

It supports evaluation datasets, experiment comparisons, and evaluators for model outputs and agent behavior. [LangChain Reference Docs](https://reference.langchain.com/python/langsmith/evaluation?utm_source=chatgpt.com)

See the [LangSmith evaluation reference](https://reference.langchain.com/python/langsmith/evaluation?utm_source=chatgpt.com).

### What should we record?

For each agent execution, consider collecting:

| Field | Purpose |
|---|---|
| Run ID | Identify the execution |
| Trace ID | Correlate operations |
| Model version | Identify the model used |
| Prompt version | Detect prompt regressions |
| Tool name | Identify executed tools |
| Tool status | Detect failures |
| Duration | Measure latency |
| Input/output tokens | Track model consumption |
| Estimated cost | Monitor spending |
| Final status | Measure task completion |

Be careful when recording prompts and tool responses.

They may contain credentials, customer information, internal documentation, or other sensitive data.

Use redaction, access controls, and appropriate retention policies.

---

# Part 5: Failure Injection and Reliability Engineering

**Time: 48–53 minutes**

A production agent must handle failures safely.

Let's consider our incident-investigation workflow.

```text
Check Service Health
        ↓
Check Recent Errors
        ↓
Search Documentation
        ↓
Generate Report
```

What happens if `get_recent_errors()` times out?

A poorly designed agent might repeatedly call the failing tool.

Another agent might ignore the failure and invent an error summary.

Both behaviors are problematic.

## Failure injection

Failure injection means deliberately introducing controlled failures to verify how the system responds.

For example:

```python
result = run_simulated_agent(
    intent="diagnose",
    fail_tool="get_recent_errors"
)

print(result.status)
print(result.answer)
```

Expected:

```text
failed
Investigation incomplete: get_recent_errors failed.
```

Our simulator stops rather than pretending the investigation succeeded.

In a more sophisticated system, you might implement bounded retries or continue with partial evidence.

### Retry design

Not every failure should trigger the same response.

| Failure | Possible handling |
|---|---|
| Temporary network timeout | Retry with bounded exponential backoff |
| Invalid tool arguments | Validate and optionally request corrected arguments |
| Permission denied | Stop the unauthorized operation |
| Rate limit | Respect retry guidance and rate limits |
| External service unavailable | Return partial results or fail gracefully |
| Approval missing | Pause execution |
| Duplicate ticket request | Return existing operation result |

### Why idempotency matters

Suppose the agent creates an incident ticket.

The ticket service succeeds, but the response is lost.

Your application retries the request.

Without idempotency, two tickets might be created.

A safer design uses a stable operation identifier:

```python
idempotency_key = (
    f"{workflow_id}:create_incident_ticket"
)
```

The same key should be reused when retrying the same logical operation.

If the external API supports idempotency keys, send it with the request.

Otherwise, implement an operation ledger and reconciliation mechanism in your backend.

### Senior Engineer Insight

An agent's failure rate is not only about whether the LLM produces a useful answer.

It also includes operational failures, invalid actions, authorization violations, and unsafe retries.

These should be measured separately.

---

# Part 6: Production Evaluation Architecture

**Time: 53–55 minutes**

Now let's design a system that continuously evaluates your agent before and after deployment.

```text
                 DEVELOPMENT
                      |
                      v
              Agent Code Change
                      |
                      v
                CI Pipeline
                      |
                      v
              Evaluation Dataset
                      |
                      v
                Agent Runner
                      |
                      v
              Evaluation Metrics
                      |
                ┌─────┴─────┐
                |           |
                v           v
              PASS         FAIL
                |           |
                v           v
          Deploy Candidate  Block Release
                |
                v
               STAGING
                |
                v
          Production Deployment
                |
                v
          Agent Execution Traces
                |
                v
         Monitoring & Evaluation
                |
                v
         Regression Detection
```

This is part of **LLMOps**.

The objective is to make AI-system quality measurable throughout development and deployment.

## Offline vs online evaluation

**Offline evaluation** uses a fixed dataset before deployment.

For example, run 500 incident-investigation scenarios against a new prompt or model.

This helps identify regressions before users are affected.

**Online evaluation** examines actual production behavior.

For example, monitor agent failures, tool-call counts, latency, and user-reported errors.

Production evaluation should use privacy-conscious logging and representative sampling.

### Example release criteria

A team might establish requirements such as:

| Metric | Illustrative release criterion |
|---|---|
| Task success rate | ≥ 95% |
| Required-tool selection | ≥ 98% |
| Unauthorized actions | 0 observed |
| p95 latency | < 10 seconds |
| Average cost per task | Within approved budget |
| Critical regression tests | 100% pass |

These are example engineering targets, not universal standards.

The appropriate thresholds depend on application risk, business requirements, and the representativeness of the evaluation dataset.

A critical safety failure should generally block deployment even if the aggregate success rate is high.

---

# Part 7: Today's Practical Assignment

**Time: 55–58 minutes**

## Mini-Project: Evaluate Your Real Incident Agent

Extend the evaluation framework we built today.

Your task is to connect it to the LangGraph incident agent from Session 6.

### Task 1: Capture real trajectories

Record the nodes executed during each investigation.

For example:

```python
{
    "run_id": "incident-1001",
    "executed_nodes": [
        "inspect_service",
        "analyze_results",
        "request_approval"
    ],
    "status": "awaiting_approval"
}
```

### Task 2: Add approval-safety tests

Verify that `create_ticket` is never executed before approval.

Test approved, rejected, and missing-approval scenarios.

### Task 3: Add failure injection

Simulate a monitoring-service timeout.

Verify that the workflow either retries within a configured limit or terminates with a clear failure status.

It must not invent monitoring results.

### Task 4: Generate an evaluation report

Produce a report containing:

```text
Total Tests: 20
Passed: 18
Failed: 2

Task Success Rate: 90%
Safety Violations: 0
Average Tool Calls: 2.8
```

These numbers are illustrative. Your report should use actual measurements.

### Bonus challenge

Store evaluation results in PostgreSQL.

Create tables for evaluation runs, test cases, and individual results.

Include the model version, prompt version, and execution trace ID.

This will allow you to compare agent behavior across deployments.

---

# 8. Senior AI Engineer Interview Preparation

### Q1. Why is evaluating only the final answer insufficient for an AI agent?

Because an agent may produce a correct answer after executing unnecessary, unsafe, or unauthorized actions.

We must evaluate the execution trajectory as well as the final response.

### Q2. What is trajectory evaluation?

Trajectory evaluation examines the sequence of actions, tool calls, and observations produced during agent execution.

It helps identify incorrect tool selection, missing steps, repeated operations, and safety violations.

### Q3. What is the difference between tracing and evaluation?

**Tracing records what happened.**

**Evaluation determines whether what happened was acceptable.**

Tracing provides evidence that evaluators can inspect.

### Q4. How would you evaluate an agent when several execution paths are valid?

Use constraint-based evaluation.

Define required actions, forbidden actions, argument requirements, and necessary dependencies rather than requiring one exact sequence.

### Q5. How would you detect a regression after changing the LLM model?

Run the old and new models against the same evaluation dataset.

Compare task success, tool selection, safety violations, latency, and cost.

Use repeated runs where model variability matters, and investigate meaningful differences before deployment.

### Advanced interview question

Your production agent has a 96% task-success rate.

However, 0.2% of executions create incident tickets without approval.

Would you deploy it?

A strong answer should explain why aggregate success rate cannot compensate for a critical authorization failure.

The system needs application-enforced approval boundaries, regression tests, and evidence that the unsafe execution path has been eliminated.

---

# 🧠 Knowledge Quiz

Try answering these questions without looking back.

**Q1.** What is the difference between task-success rate and tool-selection accuracy?

**Q2.** Why might exact trajectory matching be unsuitable for evaluating an LLM-powered agent?

**Q3.** Consider:

```python
expected = [
    "get_service_health",
    "get_recent_errors"
]

actual = [
    "get_service_health",
    "search_documentation",
    "get_recent_errors"
]
```

Should this trajectory pass an order-aware subsequence evaluation? Why?

**Q4.** What is the difference between a trace and a span?

**Q5.** Your agent creates a ticket successfully, but the worker crashes before saving the result. How would you prevent duplicate tickets when the workflow resumes?

**Q6 — System Design**

You have 50,000 agent executions per day.

Each execution may call an LLM several times and invoke multiple backend tools.

How would you design a system that records execution traces, evaluates a representative sample of results, monitors cost and latency, and alerts engineers when a new model version causes regressions?

Think about asynchronous processing, PostgreSQL, object storage, evaluation workers, observability, and deployment controls.

---

# Session 7 Recap

Today, we moved from building AI agents to measuring their reliability.

You learned that agent quality has several dimensions:

\[
\text{Agent Quality}
=
f(
\text{Correctness},
\text{Tool Behavior},
\text{Safety},
\text{Reliability},
\text{Latency},
\text{Cost}
)
\]

This is a conceptual relationship, not a formula that should be reduced to a single score without considering the importance of each dimension.

You built an evaluation harness, learned how to inspect trajectories, and explored tracing, failure injection, and production regression testing.

The most important lesson is:

**An AI system is not production-ready simply because it works in a demonstration. It needs measurable correctness, enforced safety boundaries, and evidence that it behaves reliably under failure conditions.**

---

## 🔜 Session 8: Fine-Tuning LLMs — When, Why & How

In our next lesson, we'll move into model customization.

We've already covered pretrained models, embeddings, RAG, and agents. Now we'll explore what happens when prompting and retrieval are insufficient.

We'll study the difference between pretraining, supervised fine-tuning, and preference optimization. We'll also introduce LoRA, QLoRA, training datasets, tokenization for fine-tuning, GPU memory requirements, and model evaluation.

Our practical project will prepare a small instruction-tuning dataset and demonstrate how parameter-efficient fine-tuning works using Python and PyTorch.

**Today's priority:** Complete the evaluation harness, connect it to your LangGraph agent, and understand the difference between observing an agent's behavior and proving that its behavior meets your requirements.