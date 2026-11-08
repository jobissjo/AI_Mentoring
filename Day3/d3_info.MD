# AI Engineer Mentorship — Session 3
## Production RAG Engineering: Chunking, Hybrid Search & Retrieval Evaluation

**Level:** Intermediate → Advanced  
**Duration:** 60 minutes  
**Focus:** Building reliable retrieval systems for production AI applications  
**Technology:** Python, Sentence Transformers, Scikit-learn, FastAPI concepts

In Session 1, we explored neural networks, gradients, and PyTorch training.

In Session 2, we learned tokenization, embeddings, cosine similarity, and built a basic semantic search engine.

**Today's goal is to transform that semantic search prototype into a more reliable retrieval system suitable for a production RAG application.**

We will focus on an important engineering question:

> Why can an AI assistant return an incorrect answer even when the correct information exists in its knowledge base?

Understanding this problem is essential for becoming a Senior AI Engineer.

---

# 1. Learning Objectives

By the end of this session, you should understand:

1. Why document chunking is necessary and how chunk size affects retrieval.
2. How a production RAG ingestion pipeline works.
3. Why semantic search alone is sometimes insufficient.
4. How hybrid search combines keyword and vector retrieval.
5. How Reciprocal Rank Fusion (RRF) combines search results.
6. How to measure retrieval quality using Recall@K.
7. How to design secure, observable retrieval systems.

### Today's Schedule

| Time | Topic |
|---|---|
| 0–10 min | Document ingestion and chunking |
| 10–20 min | Semantic search limitations and hybrid retrieval |
| 20–40 min | Build a hybrid search engine in Python |
| 40–50 min | Retrieval evaluation and debugging |
| 50–55 min | Production architecture and system design |
| 55–60 min | Practical assignment and interview quiz |

---

# Part 1: Why Chunking Matters

**Time: 0–10 minutes**

Imagine you're building an internal AI assistant for a company.

Your company has a 200-page technical document containing information about PostgreSQL, Django, Redis, and Docker.

A developer asks:

> Why does PostgreSQL return the error "too many clients already"?

Your system must find the relevant section.

## Approach A: Embed the entire document

```text
200-page document
       ↓
Embedding Model
       ↓
Single Vector
       ↓
Vector Database
```

There are two major problems.

First, embedding models have input-length limits. Long documents may be truncated.

Second, a single embedding may not preserve enough information about every individual topic.

A 200-page document might discuss hundreds of unrelated concepts.

Representing all of them with one vector makes precise retrieval difficult.

## Approach B: Split the document into chunks

```text
200-page document
       ↓
Text Extraction
       ↓
Chunking
       ↓
Chunk 1 → Embedding 1
Chunk 2 → Embedding 2
Chunk 3 → Embedding 3
...
Chunk N → Embedding N
       ↓
Vector Database
```

Now the system can retrieve the specific section related to PostgreSQL connections.

### What is chunking?

Chunking means dividing large documents into smaller pieces that can be independently embedded and retrieved.

For example:

```python
document = """
PostgreSQL supports multiple database connections.
Connection pools help applications reuse existing connections.

When the maximum number of connections is reached,
PostgreSQL may reject new connections.

Developers should inspect active sessions,
connection pool settings, and long-running transactions.
"""
```

We could divide this document into three chunks.

However, splitting at arbitrary boundaries may separate related information.

For example:

**Chunk 1:**

```text
When the maximum number of connections is reached,
```

**Chunk 2:**

```text
PostgreSQL may reject new connections.
```

Neither chunk contains the complete explanation.

This is why chunking strategy matters.

## Chunk overlap

One common solution is overlapping chunks.

```text
Chunk 1:
[A B C D E]

Chunk 2:
[D E F G H]

Chunk 3:
[G H I J K]
```

Some content is repeated between adjacent chunks.

This helps preserve context near chunk boundaries.

However, overlap also increases storage requirements and can create redundant retrieval results.

### Python implementation

Let's implement a simple word-based chunker.

```python
def chunk_text(
    text: str,
    chunk_size: int = 50,
    overlap: int = 10
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk = words[start:start + chunk_size]

        if not chunk:
            break

        chunks.append(" ".join(chunk))

        if start + chunk_size >= len(words):
            break

    return chunks
```

Usage:

```python
chunks = chunk_text(
    document,
    chunk_size=20,
    overlap=5
)

for index, chunk in enumerate(chunks):
    print(f"Chunk {index + 1}:")
    print(chunk)
    print()
```

This implementation is useful for learning, but production systems generally require more sophisticated strategies.

### Common chunking strategies

| Strategy | Description | Suitable for |
|---|---|---|
| Fixed-size | Split after a fixed number of tokens | Simple documents |
| Recursive | Split using paragraphs, sentences, then smaller units | General documents |
| Semantic | Split based on topic or semantic changes | Complex knowledge bases |
| Structure-aware | Respect headings, tables, and sections | Technical documentation |
| Parent-child | Retrieve small chunks but provide larger parent sections | Context-heavy documents |

**Senior Engineer Insight:** There is no universally optimal chunk size.

You should select chunking strategies based on document structure, embedding-model limits, retrieval quality, and measured performance.

---

# Part 2: Why Semantic Search Alone Is Not Enough

**Time: 10–20 minutes**

In our previous session, we used embedding vectors to find semantically similar documents.

That approach works well for many natural-language questions.

But consider this query:

```text
PostgreSQL error 53300
```

Error `53300` is a specific PostgreSQL SQLSTATE code associated with too many connections.

A semantic embedding model might retrieve a general article about PostgreSQL performance rather than the exact error documentation.

This happens because embedding models are optimized to represent useful relationships, not necessarily exact identifier matches.

Now consider two retrieval approaches.

### Semantic search

Query:

> Why are database connections getting rejected?

Potentially relevant document:

> PostgreSQL rejects new sessions when connection limits are exceeded.

Semantic search can recognize the relationship even when the wording differs.

### Keyword search

Query:

> PostgreSQL 53300

Relevant document:

> SQLSTATE 53300: too_many_connections.

Keyword retrieval is useful because the exact error code appears in the document.

## Hybrid search

Hybrid search combines both approaches.

```text
                User Query
                    |
          ┌─────────┴─────────┐
          |                   |
     Semantic Search     Keyword Search
          |                   |
     Vector Results      Lexical Results
          |                   |
          └─────────┬─────────┘
                    |
                Rank Fusion
                    |
              Final Results
```

This architecture is common in enterprise search and RAG systems.

### Why not simply add the similarity scores?

Suppose semantic search returns:

```text
Document A: 0.85
Document B: 0.78
```

Keyword search returns:

```text
Document A: 12.5
Document C: 9.8
```

These scores have different meanings and scales.

Adding them directly can produce misleading rankings.

One alternative is **Reciprocal Rank Fusion (RRF)**.

## Reciprocal Rank Fusion

RRF combines ranked lists using document positions rather than raw similarity scores.

The formula is:

\[
RRF(d)=\sum_{r\in R}\frac{1}{k+rank_r(d)}
\]

Where:

- \(d\) is a document.
- \(R\) is the collection of ranking systems.
- \(rank_r(d)\) is the document's position in ranking \(r\).
- \(k\) is a smoothing constant, often set to 60.

Example:

Semantic ranking:

```text
1. Document A
2. Document B
3. Document C
```

Keyword ranking:

```text
1. Document C
2. Document A
3. Document D
```

Document A receives:

\[
RRF(A)=\frac{1}{60+1}+\frac{1}{60+2}
\]

Approximately:

\[
RRF(A)=0.0325
\]

Document C receives:

\[
RRF(C)=\frac{1}{60+3}+\frac{1}{60+1}
\]

Approximately:

\[
RRF(C)=0.0323
\]

Document A ranks slightly higher.

**Important:** RRF scores are ranking signals, not probabilities of relevance.

---

# Part 3: Build a Hybrid Search Engine in Python

**Time: 20–40 minutes**

We're going to extend the semantic search project from Session 2.

Our new system will support:

- Document chunking
- Sentence embeddings
- Keyword retrieval
- Hybrid ranking
- Source metadata

For simplicity, we'll use TF-IDF for keyword retrieval.

Production search systems often use BM25 or other lexical ranking methods.

## Step 1: Install dependencies

```bash
pip install sentence-transformers scikit-learn numpy
```

## Step 2: Prepare sample documentation

Create `hybrid_search.py`.

```python
documents = [
    {
        "id": "postgres",
        "title": "PostgreSQL Connection Errors",
        "content": """
        PostgreSQL SQLSTATE 53300 indicates too_many_connections.
        This happens when the database reaches its configured
        connection limit. Applications may receive connection
        failures when too many sessions remain active.

        Developers should inspect pg_stat_activity, review
        connection pool configuration, and identify long-running
        database sessions. Increasing max_connections without
        checking memory requirements can cause additional problems.
        """
    },
    {
        "id": "redis",
        "title": "Redis Persistence Problems",
        "content": """
        Redis may return a MISCONF error when configured RDB
        snapshots cannot be saved successfully. When write
        protection is enabled, commands that modify data
        may be rejected.

        Administrators should inspect Redis logs, available
        disk space, filesystem permissions, and persistence
        configuration before changing server settings.
        """
    },
    {
        "id": "django",
        "title": "Django Authentication",
        "content": """
        Django REST Framework supports authentication using
        JWT access tokens and refresh tokens. Authentication
        verifies the identity of a requester.

        Authorization determines which resources an authenticated
        user is permitted to access. Production APIs should
        validate tokens and enforce object-level permissions.
        """
    },
    {
        "id": "celery",
        "title": "Celery Worker Troubleshooting",
        "content": """
        Celery processes asynchronous background jobs using
        workers and message brokers. Tasks may remain pending
        when workers are unavailable or cannot consume messages.

        Developers should inspect worker logs, queue routing,
        broker connectivity, task retries, and worker concurrency.
        """
    }
]
```

## Step 3: Chunk the documents

Reuse the `chunk_text()` function from Part 1.

```python
chunks = []

for document in documents:

    parts = chunk_text(
        document["content"],
        chunk_size=35,
        overlap=8
    )

    for index, part in enumerate(parts):

        chunks.append({
            "id": f"{document['id']}:{index}",
            "document_id": document["id"],
            "title": document["title"],
            "content": part
        })

print("Total chunks:", len(chunks))
```

Each chunk retains information about its source document.

This is important because retrieval results must eventually be traceable to their original source.

A production system should also retain metadata such as document version, ingestion timestamp, tenant, and access permissions.

## Step 4: Generate embeddings

```python
import numpy as np

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

chunk_texts = [
    chunk["content"]
    for chunk in chunks
]

embeddings = embedding_model.encode(
    chunk_texts,
    normalize_embeddings=True,
    convert_to_numpy=True
)

print("Embedding shape:", embeddings.shape)
```

If there are \(N\) chunks, the embedding matrix has shape:

\[
(N,384)
\]

We normalize the embeddings so cosine similarity can be computed using a dot product.

## Step 5: Build the keyword index

We'll use [Scikit-learn's TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html?utm_source=chatgpt.com).

TF-IDF stands for **Term Frequency–Inverse Document Frequency**.

It gives higher weight to words that are important within a document relative to the overall collection.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(
    chunk_texts
)

print("TF-IDF shape:", tfidf_matrix.shape)
```

Unlike dense embeddings, TF-IDF represents text using a sparse vocabulary-based matrix.

## Step 6: Implement hybrid retrieval

```python
def hybrid_search(
    query: str,
    top_k: int = 3
):

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # 1. Semantic retrieval
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    semantic_scores = (
        embeddings @ query_embedding
    )

    semantic_ranking = np.argsort(
        -semantic_scores
    )

    # 2. Keyword retrieval
    query_tfidf = vectorizer.transform(
        [query]
    )

    keyword_scores = (
        tfidf_matrix @ query_tfidf.T
    ).toarray().ravel()

    keyword_ranking = np.argsort(
        -keyword_scores
    )

    # 3. Reciprocal Rank Fusion
    rrf_scores = {}

    k = 60

    for rank, index in enumerate(
        semantic_ranking,
        start=1
    ):

        index = int(index)

        rrf_scores[index] = (
            rrf_scores.get(index, 0)
            + 1 / (k + rank)
        )

    lexical_rank = 0

    for index in keyword_ranking:

        index = int(index)

        if keyword_scores[index] <= 0:
            continue

        lexical_rank += 1

        rrf_scores[index] = (
            rrf_scores.get(index, 0)
            + 1 / (k + lexical_rank)
        )

    # 4. Sort by fused score
    ranked_indices = sorted(
        rrf_scores,
        key=lambda i: rrf_scores[i],
        reverse=True
    )

    # 5. Return results
    results = []

    for index in ranked_indices[:top_k]:

        results.append({
            **chunks[index],
            "score": round(
                rrf_scores[index],
                5
            )
        })

    return results
```

This implementation uses full rankings for a small in-memory dataset.

At production scale, the semantic and lexical systems would normally retrieve bounded candidate sets before fusion.

## Step 7: Test the search engine

```python
queries = [
    "PostgreSQL error 53300",
    "Why are database connections rejected?",
    "Redis MISCONF snapshot error",
    "Why are Celery jobs stuck?"
]

for query in queries:

    print(f"\nQUERY: {query}")

    results = hybrid_search(
        query,
        top_k=3
    )

    for result in results:

        print(
            result["document_id"],
            result["score"]
        )
```

Inspect the results.

You should expect PostgreSQL-related queries to favor PostgreSQL documentation, Redis queries to favor Redis documentation, and so on.

However, don't assume the rankings are correct without evaluating them.

That brings us to the next important topic.

---

# Part 4: How Do We Know Retrieval Is Working?

**Time: 40–50 minutes**

This is where the responsibilities of an AI Engineer become different from those of a traditional backend developer.

In conventional backend development, you might test:

```python
assert response.status_code == 200
```

But imagine your RAG retrieval endpoint returns:

```json
{
  "status": "success",
  "results": [
    {
      "document_id": "django",
      "content": "Django JWT authentication..."
    }
  ]
}
```

The request succeeded technically.

But the user's question was:

> Why is PostgreSQL rejecting connections?

Your API works, but your AI system has failed.

**A successful HTTP response does not mean the retrieved information is relevant.**

We need retrieval evaluation.

## Create a small evaluation dataset

```python
evaluation_queries = [
    {
        "query": "PostgreSQL error 53300",
        "relevant_documents": {"postgres"}
    },
    {
        "query": "Redis MISCONF snapshot failure",
        "relevant_documents": {"redis"}
    },
    {
        "query": "How does JWT authentication work?",
        "relevant_documents": {"django"}
    },
    {
        "query": "Why are Celery tasks pending?",
        "relevant_documents": {"celery"}
    }
]
```

This dataset contains queries and their expected relevant documents.

In real applications, relevance judgments should be based on representative user queries and reviewed documentation.

## Recall@K

Recall@K measures how many relevant documents appear in the top K retrieval results.

\[
Recall@K=
\frac{\text{Relevant documents retrieved in top K}}
{\text{Total relevant documents}}
\]

Example:

There are four relevant documents.

Your retrieval system returns two of them in its top five results.

\[
Recall@5=\frac{2}{4}=0.5
\]

Your Recall@5 is 50%.

### Python implementation

```python
def evaluate_recall_at_k(
    evaluation_queries,
    k=3
):

    scores = []

    for item in evaluation_queries:

        results = hybrid_search(
            item["query"],
            top_k=k
        )

        retrieved_ids = {
            result["document_id"]
            for result in results
        }

        relevant_ids = item[
            "relevant_documents"
        ]

        recall = (
            len(retrieved_ids & relevant_ids)
            / len(relevant_ids)
        )

        scores.append(recall)

        print(
            item["query"],
            f"Recall@{k}: {recall:.2f}"
        )

    return sum(scores) / len(scores)


average_recall = evaluate_recall_at_k(
    evaluation_queries,
    k=3
)

print("Average Recall:", average_recall)
```

This evaluates document-level recall, even though the system retrieves chunks.

That distinction is important: chunk-level and document-level evaluations measure different things.

Our four-query dataset is only a smoke test. It is far too small to establish production retrieval quality.

## What if Recall@K is poor?

Don't immediately switch to a larger embedding model.

Investigate the pipeline.

A useful debugging sequence is:

```text
Incorrect Retrieval
        ↓
Was the correct document ingested?
        ↓
Was the relevant section extracted?
        ↓
Was the section chunked correctly?
        ↓
Was the embedding generated correctly?
        ↓
Was the correct index searched?
        ↓
Did access filters exclude the document?
        ↓
Did fusion or reranking push it down?
```

This is a key Senior AI Engineer skill.

You need to identify **which stage is responsible for a failure**.

### Retrieval vs generation evaluation

When we later connect an LLM, we'll evaluate two separate systems.

| Component | Evaluation |
|---|---|
| Retrieval | Recall@K, Precision@K, MRR, nDCG |
| Generation | Answer correctness, faithfulness, citation accuracy |
| Operations | Latency, cost, error rate, availability |

A correct document can be retrieved while the LLM still generates an incorrect answer.

Therefore, evaluating retrieval alone is insufficient for a complete RAG application.

---

# Part 5: Production RAG System Design

**Time: 50–55 minutes**

Let's design the architecture you might build for a company using Django or FastAPI.

```text
                 DOCUMENT INGESTION

Google Drive / PDFs / Database / APIs
                    |
                    v
             Ingestion Worker
                    |
                    v
           Text Extraction
                    |
                    v
             Chunking
                    |
                    v
           Embedding Model
                    |
                    v
       PostgreSQL + pgvector
                    |
                    v
             Search Index


                   QUERY FLOW

                 Client
                    |
                    v
               FastAPI
                    |
                    v
          Authentication / ACL
                    |
                    v
             Query Processing
                    |
           ┌────────┴────────┐
           |                 |
           v                 v
       Vector Search     Keyword Search
           |                 |
           └────────┬────────┘
                    |
                    v
                 RRF
                    |
                    v
               Reranker
                    |
                    v
            Context Builder
                    |
                    v
                   LLM
                    |
                    v
              Response
```

A reranker is different from RRF.

RRF combines existing rankings. A learned reranker, such as a cross-encoder, directly scores query-document pairs and can improve the ordering of retrieved candidates.

### Engineering decisions

**1. Ingestion should be asynchronous.**

Large document ingestion can take time.

For your Python stack, Celery workers are a reasonable option.

**2. Reindex only changed documents.**

Store document versions or content hashes so unchanged content does not need to be reprocessed.

**3. Enforce authorization before returning retrieved content.**

Tenant and document permissions must be applied within the retrieval process.

**4. Treat retrieved documents as untrusted input.**

A document may contain instructions designed to manipulate an LLM. Retrieved text must not override application instructions.

**5. Monitor the retrieval pipeline.**

Useful operational metrics include:

- Embedding generation latency
- Vector search latency
- Retrieval Recall@K on evaluation datasets
- Index freshness
- Number of failed ingestion jobs
- Search error rates
- Retrieval and LLM costs

For a PostgreSQL-based implementation, study the [pgvector project](https://github.com/pgvector/pgvector?utm_source=chatgpt.com).

It is a natural next step given your existing backend experience.

---

# Part 6: Today's Practical Assignment

**Time: 55–60 minutes**

## Mini-project: Production-Style Hybrid Search API

Extend today's hybrid search engine using FastAPI.

Create this endpoint:

```http
POST /api/v1/search/hybrid
```

Request:

```json
{
  "query": "PostgreSQL error 53300",
  "top_k": 3
}
```

Response structure:

```json
{
  "query": "PostgreSQL error 53300",
  "results": [
    {
      "document_id": "postgres",
      "title": "PostgreSQL Connection Errors",
      "score": 0.0325,
      "content": "PostgreSQL SQLSTATE 53300 indicates..."
    }
  ],
  "retrieval_method": "hybrid"
}
```

The score is illustrative; return the score calculated by your implementation.

### Requirements

Your API should load the embedding model and build the indexes once during startup.

It should validate the query, restrict `top_k` to a sensible maximum, return source metadata, and record retrieval latency.

Add an evaluation function that measures Recall@3 using the sample evaluation dataset.

**Bonus:** Compare semantic-only, keyword-only, and hybrid retrieval on the same queries.

Record the results in a table:

| Query | Semantic | Keyword | Hybrid |
|---|---|---|---|
| PostgreSQL 53300 | ? | ? | ? |
| Database connections rejected | ? | ? | ? |
| Redis MISCONF | ? | ? | ? |
| Celery tasks pending | ? | ? | ? |

Use Recall@3 for the comparison rather than inventing similarity thresholds.

---

# Senior AI Engineer Interview Questions

Try answering these without referring to the explanations.

**Q1.** Why is embedding an entire 200-page PDF usually a poor retrieval strategy?

**Q2.** What is the difference between semantic search, keyword search, and hybrid search?

**Q3.** Why is Reciprocal Rank Fusion useful when combining retrieval systems with different scoring scales?

**Q4.** Your RAG application returns HTTP 200, but users report incorrect answers. How would you determine whether retrieval or generation is responsible?

**Q5.** How would you prevent one organization's private documents from appearing in another organization's retrieval results?

### Advanced question

You have 10 million document chunks stored in PostgreSQL with pgvector.

Search latency has increased from 80 ms to 1.5 seconds.

What would you investigate?

Think about indexing, approximate nearest-neighbor search, filtering, database load, query plans, and the number of candidates being retrieved.

---

# Session 3 Recap

Today, you learned that building a production RAG system involves much more than generating embeddings and calling an LLM.

The important pipeline is:

\[
\text{Documents}
\rightarrow
\text{Chunks}
\rightarrow
\text{Embeddings}
\rightarrow
\text{Hybrid Retrieval}
\rightarrow
\text{Reranking}
\rightarrow
\text{Context}
\]

You implemented chunking, semantic search, keyword retrieval, and Reciprocal Rank Fusion.

You also learned how to measure retrieval quality and diagnose failures.

**The key engineering principle is that retrieval quality must be measured, not assumed.**

## Next Session: RAG Generation, Prompt Injection & Answer Evaluation

We'll continue directly from today's hybrid search project.

In Session 4, we'll connect a real LLM to our retrieval pipeline and build a complete RAG API.

We'll cover context construction, grounding, citations, hallucination detection, prompt-injection risks, and answer evaluation.

We'll also introduce the distinction between deterministic RAG workflows and AI agents.

**Today's learning outcome:** You should be able to explain and implement a hybrid retrieval system, evaluate its retrieval quality, and identify the architectural changes required to operate it reliably in production.