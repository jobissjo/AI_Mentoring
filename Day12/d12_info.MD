# 🎓 AI Engineer Mentorship — Session 12
## Advanced RAG: Hybrid Search, BM25, Reciprocal Rank Fusion & Reranking

**Level:** Intermediate → Advanced AI Engineering  
**Duration:** 60 minutes  
**Today's project:** Build a hybrid retrieval engine that combines keyword search, vector similarity, and cross-encoder reranking.

In Session 11, we built a semantic retrieval API using FastAPI, PostgreSQL, and pgvector. We explored HNSW indexing, document ingestion, metadata filtering, and retrieval evaluation.

Today, we'll address an important limitation of vector search.

Imagine a developer asks your AI assistant:

> How do I resolve ORA-12541 in my Oracle database?

Your vector database returns documentation about general database connection errors, but misses the document specifically explaining `ORA-12541`.

Why?

Because **semantic similarity doesn't guarantee exact keyword matching**.

This becomes especially important when building AI assistants for developers, where users frequently search for error codes, function names, API endpoints, database columns, and technical identifiers.

Today, we'll build a retrieval system that handles both semantic meaning and exact terminology.

---

# 1. Learning Objectives

By the end of this session, you should understand:

1. Why vector search alone can be insufficient for production RAG.
2. How BM25 keyword ranking works.
3. The difference between sparse and dense retrieval.
4. How hybrid search combines multiple retrieval strategies.
5. How Reciprocal Rank Fusion (RRF) merges rankings.
6. How cross-encoder reranking improves candidate ordering.
7. How to implement hybrid retrieval using Python.
8. How to evaluate retrieval improvements objectively.
9. How to design hybrid retrieval for millions of documents.

## ⏱️ Today's 60-Minute Agenda

| Time | Topic |
|---|---|
| 0–10 min | Limitations of dense vector retrieval |
| 10–20 min | BM25 and lexical search |
| 20–30 min | Hybrid search and Reciprocal Rank Fusion |
| 30–45 min | Build a hybrid retrieval engine in Python |
| 45–50 min | Cross-encoder reranking |
| 50–55 min | Evaluation and production architecture |
| 55–60 min | Practical assignment, interview questions, and quiz |

---

# Part 1: Why Vector Search Alone Is Not Enough

**Time: 0–10 minutes**

Let's consider two queries.

### Query A

> Why is my database running out of connections?

A relevant document might say:

> Connection pool exhaustion occurs when all available database connections are occupied.

Although the words differ, the meaning is similar.

Dense vector search is well suited to this type of query.

### Query B

> What does ORA-12541 mean?

The correct document contains:

```text
ORA-12541: TNS:no listener
```

This query contains an exact technical identifier.

A dense embedding model may retrieve general Oracle networking documentation instead of the document containing that specific error code.

That's because embeddings are optimized to represent learned semantic relationships, not necessarily to preserve every exact identifier.

This motivates combining two retrieval methods.

| Dense retrieval | Sparse/lexical retrieval |
|---|---|
| Matches semantic meaning | Matches words and terms |
| Useful for paraphrases | Useful for exact identifiers |
| Uses embeddings | Uses lexical indexes |
| Often uses cosine similarity | Often uses BM25 |
| May miss exact codes | May miss semantic paraphrases |

Neither approach is universally better.

**Senior AI Engineer principle:** Choose retrieval methods based on the types of queries users actually submit.

For a developer-focused AI assistant, hybrid retrieval is often a strong starting point.

---

# Part 2: Understanding BM25

**Time: 10–20 minutes**

BM25 stands for **Best Matching 25**.

It is a widely used lexical ranking algorithm.

It estimates how relevant a document is to a query based on matching terms and corpus statistics.

Let's understand its underlying ideas.

## 2.1 Term Frequency

Suppose a user searches:

> PostgreSQL connection timeout

Consider two documents.

**Document A:**

> PostgreSQL connection timeout occurs when a connection cannot be established within the configured timeout.

**Document B:**

> PostgreSQL supports multiple indexing techniques.

Document A contains more of the relevant query terms.

BM25 considers how frequently matching terms appear.

However, repeating a keyword hundreds of times shouldn't make a document infinitely more relevant.

BM25 uses a saturating term-frequency function to limit the benefit of repetition.

## 2.2 Inverse Document Frequency

Now imagine searching a collection of 10,000 technical documents.

The word `error` appears in 8,000 documents.

The term `ORA-12541` appears in only 10.

Which term is more useful for distinguishing relevant documents?

Usually, `ORA-12541`.

This is the intuition behind **Inverse Document Frequency (IDF)**.

Rare terms generally receive more importance than terms appearing in nearly every document.

## 2.3 Document Length Normalization

Suppose two documents contain the same query terms.

Document A contains 100 words.

Document B contains 10,000 words.

The matching terms may be more concentrated in Document A.

BM25 includes document-length normalization so long documents don't automatically dominate simply because they contain more words.

## 2.4 BM25 Formula

A common form of Okapi BM25 is:

\[
BM25(D,Q)=
\sum_{q\in Q}
IDF(q)
\frac{f(q,D)(k_1+1)}
{f(q,D)+k_1(1-b+b\frac{|D|}{avgdl})}
\]

Where:

| Symbol | Meaning |
|---|---|
| \(D\) | Document |
| \(Q\) | Query |
| \(f(q,D)\) | Frequency of term \(q\) in document \(D\) |
| \(IDF(q)\) | Importance of the term across the collection |
| \(|D|\) | Document length |
| \(avgdl\) | Average document length |
| \(k_1\) | Controls term-frequency saturation |
| \(b\) | Controls document-length normalization |

Common starting values are:

\[
k_1=1.5,\qquad b=0.75
\]

These are not universally optimal.

They should be tuned or validated against your retrieval dataset.

### Interview insight

**Why doesn't BM25 understand semantic meaning?**

Because it relies primarily on lexical matches and term statistics.

For example, it may not automatically understand that:

> database connection exhaustion

and

> all connections in the pool are occupied

describe the same problem.

Dense embeddings can help bridge that vocabulary gap.

---

# Part 3: Hybrid Search

**Time: 20–25 minutes**

Hybrid search combines multiple retrieval approaches.

For today's project:

```text
                  User Query
                      |
              ┌───────┴───────┐
              |               |
              v               v
         BM25 Search      Vector Search
              |               |
              v               v
        Keyword Results   Semantic Results
              |               |
              └───────┬───────┘
                      |
                      v
             Reciprocal Rank Fusion
                      |
                      v
              Combined Candidates
                      |
                      v
                Reranking
                      |
                      v
               Top Documents
                      |
                      v
                     LLM
```

Let's examine why the fusion step is necessary.

Suppose BM25 returns:

| Rank | Document | BM25 score |
|---|---|---:|
| 1 | A | 12.5 |
| 2 | B | 9.8 |
| 3 | C | 7.2 |

Vector search returns:

| Rank | Document | Cosine similarity |
|---|---|---:|
| 1 | B | 0.91 |
| 2 | D | 0.87 |
| 3 | A | 0.82 |

We cannot simply add the raw scores.

Why?

Because BM25 and cosine similarity use different numerical scales.

A BM25 score of 12.5 is not directly comparable to a cosine similarity of 0.91.

One solution is score normalization.

Another is **Reciprocal Rank Fusion**.

---

# Part 4: Reciprocal Rank Fusion (RRF)

**Time: 25–30 minutes**

RRF combines ranked lists using the position of each document rather than its raw score.

The formula is:

\[
RRF(d)=\sum_{i=1}^{n}\frac{1}{k+rank_i(d)}
\]

Where:

- \(d\) is a document.
- \(n\) is the number of retrieval systems.
- \(rank_i(d)\) is the document's rank in system \(i\).
- \(k\) is a ranking constant.

A common value is:

\[
k=60
\]

Documents missing from a particular ranking contribute zero for that ranking.

RRF is useful because it avoids directly comparing scores from incompatible retrieval methods. [Elastic](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion?utm_source=chatgpt.com)

## Example

Suppose:

**BM25 results:**

```text
1. Document A
2. Document B
3. Document C
```

**Vector results:**

```text
1. Document B
2. Document D
3. Document A
```

For Document A:

\[
RRF(A)=\frac{1}{60+1}+\frac{1}{60+3}
\]

\[
RRF(A)\approx0.0323
\]

For Document B:

\[
RRF(B)=\frac{1}{60+2}+\frac{1}{60+1}
\]

\[
RRF(B)\approx0.0325
\]

Document B receives a slightly higher combined score.

This makes sense because it performs well in both retrieval systems.

### Important limitation

RRF is not guaranteed to outperform dense or sparse retrieval.

It depends on candidate quality, ranking depth, and query characteristics.

That's why we must evaluate it.

---

# Part 5: Build a Hybrid Retrieval Engine in Python

**Time: 30–45 minutes**

Let's implement the concepts.

We'll use three libraries:

- `rank_bm25` for lexical search.
- Sentence Transformers for dense embeddings and reranking.
- NumPy for vector similarity calculations.

The first version will operate on an in-memory document collection.

This lets us understand the ranking logic before integrating it with the pgvector database from Session 11.

## Step 1: Install dependencies

```bash
pip install numpy rank_bm25 sentence-transformers
```

Create:

```text
advanced-rag/
    retrieval_lab.py
```

## Step 2: Create a document collection

Add to `retrieval_lab.py`:

```python
documents = [
    {
        "id": "postgres_pool",
        "text": (
            "PostgreSQL connection pool exhaustion occurs "
            "when all available database connections are "
            "occupied. Inspect active connections and "
            "connection pool limits."
        )
    },
    {
        "id": "oracle_listener",
        "text": (
            "ORA-12541 TNS no listener means the Oracle "
            "database listener is unavailable or not "
            "accepting connections."
        )
    },
    {
        "id": "redis_persistence",
        "text": (
            "Redis MISCONF indicates that Redis cannot "
            "persist snapshots to disk. Check storage "
            "permissions and available disk space."
        )
    },
    {
        "id": "celery_worker",
        "text": (
            "Celery worker lost errors occur when a "
            "background worker process crashes or "
            "becomes unavailable."
        )
    },
    {
        "id": "jwt_auth",
        "text": (
            "JWT authentication fails when an access "
            "token is expired, invalid, or has an "
            "incorrect signature."
        )
    },
    {
        "id": "postgres_index",
        "text": (
            "PostgreSQL query performance can improve "
            "through appropriate indexes and execution "
            "plan analysis using EXPLAIN ANALYZE."
        )
    },
    {
        "id": "docker_oom",
        "text": (
            "Docker containers may terminate when "
            "their processes exceed available memory "
            "or configured memory limits."
        )
    },
    {
        "id": "redis_cache",
        "text": (
            "Redis is an in-memory data store commonly "
            "used for caching, sessions, and message "
            "broker workloads."
        )
    },
    {
        "id": "api_rate_limit",
        "text": (
            "HTTP 429 Too Many Requests indicates "
            "that a client has exceeded an API "
            "rate limit."
        )
    },
    {
        "id": "database_timeout",
        "text": (
            "Database connection timeouts can result "
            "from network failures, unavailable "
            "servers, or overloaded connection pools."
        )
    }
]
```

This is our miniature knowledge base.

In production, these records would come from the database and document-ingestion pipeline we developed previously.

---

## Step 3: Implement BM25 search

```python
import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[a-z0-9]+(?:-[a-z0-9]+)*",
        text.lower()
    )


tokenized_documents = [
    tokenize(doc["text"])
    for doc in documents
]

bm25 = BM25Okapi(tokenized_documents)


def bm25_search(
    query: str,
    top_k: int = 5
) -> list[str]:

    scores = bm25.get_scores(
        tokenize(query)
    )

    ranked_indices = sorted(
        range(len(documents)),
        key=lambda i: float(scores[i]),
        reverse=True
    )

    return [
        documents[i]["id"]
        for i in ranked_indices
        if scores[i] > 0
    ][:top_k]
```

The `rank_bm25` library implements Okapi BM25 and expects tokenized documents. It does not automatically handle text preprocessing. [GitHub](https://github.com/dorianbrown/rank_bm25?utm_source=chatgpt.com)

Try:

```python
print(
    bm25_search(
        "ORA-12541 database error"
    )
)
```

You should expect the Oracle listener document to rank prominently because it contains the exact error identifier.

### Why do we tokenize?

BM25 operates on terms.

Our tokenizer converts:

```text
"ORA-12541: TNS no listener"
```

into:

```python
[
    "ora-12541",
    "tns",
    "no",
    "listener"
]
```

For production search, tokenization becomes more sophisticated.

You may need language-specific analyzers, stemming, stop-word handling, and special treatment for identifiers.

For developer documentation, preserving error codes and function names is especially important.

---

## Step 4: Implement dense vector search

Now add semantic retrieval.

```python
import numpy as np

from sentence_transformers import (
    SentenceTransformer
)


embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

document_embeddings = embedding_model.encode(
    [doc["text"] for doc in documents],
    normalize_embeddings=True
)


def dense_search(
    query: str,
    top_k: int = 5
) -> list[str]:

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    similarities = (
        document_embeddings @ query_embedding
    )

    ranked_indices = np.argsort(
        -similarities
    )[:top_k]

    return [
        documents[i]["id"]
        for i in ranked_indices
    ]
```

Because both document and query embeddings are normalized, their dot product corresponds to cosine similarity, up to numerical precision.

Now test:

```python
query = (
    "Why does my application run out "
    "of database connections?"
)

print("BM25:", bm25_search(query))

print("Dense:", dense_search(query))
```

Compare the rankings.

BM25 focuses on matching terms.

Dense retrieval considers learned semantic relationships.

Depending on the query, the two methods may produce different results.

---

## Step 5: Implement Reciprocal Rank Fusion

Now we'll combine both rankings.

```python
from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60
) -> list[str]:

    scores = defaultdict(float)

    for ranked_list in ranked_lists:

        for rank, document_id in enumerate(
            ranked_list,
            start=1
        ):

            scores[document_id] += (
                1.0 / (k + rank)
            )

    return sorted(
        scores,
        key=lambda doc_id: scores[doc_id],
        reverse=True
    )
```

Now create a hybrid retrieval function.

```python
def hybrid_search(
    query: str,
    top_k: int = 5,
    candidate_k: int = 8
) -> list[str]:

    sparse_results = bm25_search(
        query,
        top_k=candidate_k
    )

    dense_results = dense_search(
        query,
        top_k=candidate_k
    )

    fused_results = reciprocal_rank_fusion(
        [
            sparse_results,
            dense_results
        ]
    )

    return fused_results[:top_k]
```

Test:

```python
query = "PostgreSQL connection pool exhausted"

print("BM25:")
print(bm25_search(query))

print("Dense:")
print(dense_search(query))

print("Hybrid:")
print(hybrid_search(query))
```

We now have three retrieval strategies.

**BM25:** Exact lexical matching.

**Dense:** Semantic similarity.

**Hybrid:** Rank fusion across both systems.

This is the foundation of our advanced retrieval engine.

---

# Part 6: Cross-Encoder Reranking

**Time: 45–50 minutes**

Hybrid search improves candidate retrieval, but the resulting order may still be imperfect.

Suppose our hybrid retriever returns:

```text
1. General database timeout guide
2. PostgreSQL connection pool guide
3. PostgreSQL indexing guide
4. Oracle listener guide
5. Redis cache guide
```

The second document may be the most relevant.

How can we improve its position?

One approach is reranking.

## 6.1 Bi-encoder vs cross-encoder

Our dense embedding model is a bi-encoder.

It encodes the query and documents separately.

```text
Query ────────> Encoder ────────> Query Vector

Document ─────> Encoder ────────> Document Vector

                        ↓
                 Similarity Score
```

This is efficient because document embeddings can be precomputed.

A cross-encoder processes the query and document together.

```text
       Query + Document
               |
               v
          Cross-Encoder
               |
               v
         Relevance Score
```

Cross-encoders can model more detailed interactions between query terms and document content.

However, they require a forward pass for each query-document pair.

That makes them more computationally expensive than comparing precomputed embeddings.

This is why cross-encoders are commonly used to rerank a limited candidate set rather than search the entire corpus. [Sentence Transformers](https://www.sbert.net/docs/package_reference/cross_encoder/model.html?utm_source=chatgpt.com)

## 6.2 Implement reranking

Add:

```python
from sentence_transformers import CrossEncoder


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

documents_by_id = {
    doc["id"]: doc
    for doc in documents
}


def rerank_documents(
    query: str,
    document_ids: list[str],
    top_k: int = 3
) -> list[str]:

    if not document_ids:
        return []

    pairs = [
        (
            query,
            documents_by_id[doc_id]["text"]
        )
        for doc_id in document_ids
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(document_ids, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    return [
        doc_id
        for doc_id, score in ranked[:top_k]
    ]
```

Now combine everything.

```python
def advanced_retrieval(
    query: str,
    top_k: int = 3
) -> list[str]:

    candidates = hybrid_search(
        query,
        top_k=8,
        candidate_k=8
    )

    return rerank_documents(
        query,
        candidates,
        top_k=top_k
    )
```

Run:

```python
query = (
    "Why are all my PostgreSQL "
    "connections being used?"
)

results = advanced_retrieval(query)

for document_id in results:

    print(
        document_id,
        documents_by_id[document_id]["text"]
    )
```

We now have a complete retrieval pipeline:

\[
\boxed{
BM25 + Dense
\rightarrow
RRF
\rightarrow
CrossEncoder
\rightarrow
TopK
}
\]

### Important detail

The selected MS MARCO cross-encoder typically produces relevance logits.

These scores should not automatically be interpreted as calibrated probabilities.

Their primary purpose here is to rank candidate documents.

---

# Part 7: Evaluating Retrieval Quality

**Time: 50–53 minutes**

In Session 11, we introduced Recall@K and Mean Reciprocal Rank.

Now we'll use those metrics to compare our retrieval methods.

## Step 1: Create evaluation cases

```python
evaluation_cases = [
    {
        "query": "ORA-12541 listener error",
        "relevant": {"oracle_listener"}
    },
    {
        "query": (
            "Why are all database "
            "connections occupied?"
        ),
        "relevant": {
            "postgres_pool",
            "database_timeout"
        }
    },
    {
        "query": "Redis MISCONF snapshot failure",
        "relevant": {"redis_persistence"}
    },
    {
        "query": "Celery background worker crashed",
        "relevant": {"celery_worker"}
    },
    {
        "query": "JWT token expired",
        "relevant": {"jwt_auth"}
    },
    {
        "query": "HTTP 429 rate limit",
        "relevant": {"api_rate_limit"}
    }
]
```

These are illustrative relevance labels.

A production evaluation dataset should be larger, independently labeled, and representative of actual user queries.

## Step 2: Implement evaluation metrics

```python
def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int
) -> float:

    if not relevant:
        return 0.0

    retrieved_set = set(retrieved[:k])

    return len(
        retrieved_set & relevant
    ) / len(relevant)


def reciprocal_rank(
    retrieved: list[str],
    relevant: set[str],
    k: int
) -> float:

    for rank, doc_id in enumerate(
        retrieved[:k],
        start=1
    ):

        if doc_id in relevant:
            return 1.0 / rank

    return 0.0
```

## Step 3: Compare retrieval methods

```python
retrievers = {
    "BM25": bm25_search,
    "Dense": dense_search,
    "Hybrid": hybrid_search,
    "Hybrid + Reranker": advanced_retrieval
}


for name, retriever in retrievers.items():

    recalls = []
    reciprocal_ranks = []

    for case in evaluation_cases:

        results = retriever(
            case["query"],
            top_k=3
        )

        recalls.append(
            recall_at_k(
                results,
                case["relevant"],
                k=3
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                results,
                case["relevant"],
                k=3
            )
        )

    print(
        name,
        "Recall@3:",
        round(float(np.mean(recalls)), 3),
        "MRR@3:",
        round(float(np.mean(reciprocal_ranks)), 3)
    )
```

This calculates macro-averaged Recall@3 and MRR@3 across our test queries.

Your output will contain actual measured values for each strategy.

Do not assume the reranker will always win.

It might improve some queries while making others worse.

### What should you compare?

| Metric | BM25 | Dense | Hybrid | Hybrid + Reranker |
|---|---|---|---|---|
| Recall@3 | Measure | Measure | Measure | Measure |
| MRR@3 | Measure | Measure | Measure | Measure |
| p95 latency | Measure | Measure | Measure | Measure |
| Cost per query | Measure | Measure | Measure | Measure |

A production system should evaluate both retrieval quality and operational performance.

For example, improving MRR by 1% may not justify tripling inference latency.

The decision depends on your application's requirements.

---

# Part 8: Production Hybrid RAG Architecture

**Time: 53–55 minutes**

Now let's connect today's work to the PostgreSQL and pgvector system from Session 11.

Our current implementation stores everything in memory.

That is useful for learning but unsuitable for millions of chunks.

A production architecture could look like:

```text
                     USER QUERY
                          |
                          v
                     FastAPI
                          |
                          v
                 Authentication
                          |
                          v
                 Query Processing
                          |
                ┌─────────┴─────────┐
                |                   |
                v                   v
          Lexical Search       Dense Search
                |                   |
                v                   v
          Search Index           pgvector
                |                   |
                └─────────┬─────────┘
                          |
                          v
                    RRF Fusion
                          |
                          v
                 Cross-Encoder Rerank
                          |
                          v
                   Top 5 Chunks
                          |
                          v
                    Context Builder
                          |
                          v
                      LLM Server
                          |
                          v
                 Answer with Sources
```

## 8.1 Lexical search with PostgreSQL

PostgreSQL provides built-in full-text search.

For example:

```sql
SELECT
    document_id,
    content,
    ts_rank_cd(
        to_tsvector('english', content),
        plainto_tsquery('english', 'connection timeout')
    ) AS rank
FROM document_chunks
WHERE
    to_tsvector('english', content)
    @@ plainto_tsquery(
        'english',
        'connection timeout'
    )
ORDER BY rank DESC
LIMIT 10;
```

**Important:** PostgreSQL's `ts_rank_cd` is not BM25.

It provides lexical relevance ranking using PostgreSQL's full-text search mechanisms.

If you specifically require BM25, use a search engine or extension that implements it.

For a production PostgreSQL solution, you could combine its full-text search with pgvector, then apply RRF in Python.

For larger deployments, a dedicated lexical search engine may also be appropriate.

## 8.2 Why candidate generation matters

Suppose you have 10 million chunks.

You cannot efficiently run a cross-encoder against all 10 million for every query.

Instead:

```text
10,000,000 Chunks
        |
        v
BM25 retrieves 50
Dense retrieves 50
        |
        v
RRF combines candidates
        |
        v
Rerank up to 100 unique candidates
        |
        v
Select Top 5
        |
        v
LLM
```

The numbers are illustrative.

You should tune retrieval depth and reranking depth using quality and latency measurements.

## 8.3 Security requirements

For multi-tenant RAG, authorization must be applied before documents are exposed to downstream stages.

Do not retrieve documents from every organization, rerank them, and then filter unauthorized results afterward.

Instead, apply tenant and document-level access constraints during candidate retrieval.

Also remember that retrieved content is untrusted.

A document containing instructions to reveal credentials or bypass authorization must not gain control over the application's tools.

---

# Part 9: Today's Practical Assignment

**Time: 55–58 minutes**

## Mini-Project: Upgrade Your pgvector RAG System

Your assignment is to extend the retrieval API from Session 11.

### Task 1: Add lexical retrieval

Implement a PostgreSQL full-text search query.

Return the top 20 lexical candidates.

### Task 2: Add dense retrieval

Use your existing pgvector HNSW index.

Return the top 20 semantic candidates.

### Task 3: Add RRF

Combine the two result lists using Reciprocal Rank Fusion.

Deduplicate candidates by a stable chunk identifier.

### Task 4: Add reranking

Use a cross-encoder to rerank the top 20 fused candidates.

Return the best five.

### Task 5: Evaluate all strategies

Create at least 20 test queries.

Include exact error codes, semantic paraphrases, technical identifiers, and ambiguous questions.

Compare Recall@5, MRR@5, and p95 latency.

### Bonus challenge: Query rewriting

Consider:

> Why does it keep crashing?

This query is ambiguous.

A previous conversation might establish that the user is discussing a Celery worker.

You could rewrite the query to:

> Why does the Celery worker process repeatedly crash?

This may improve retrieval.

However, query rewriting can introduce incorrect assumptions.

Evaluate rewritten queries against the original queries rather than assuming rewriting always improves performance.

---

# Part 10: Senior AI Engineer Interview Preparation

### Q1. Why might vector search fail to retrieve a document containing an exact error code?

Embedding models prioritize learned semantic relationships and may not preserve exact identifiers reliably.

Lexical search can help retrieve documents containing the specific code.

### Q2. What is BM25?

BM25 is a lexical ranking algorithm that uses term frequency, inverse document frequency, and document-length normalization to estimate document relevance.

### Q3. Why shouldn't we directly add BM25 scores and cosine similarity scores?

Because they use different scales and distributions.

Combining them directly without calibration can make one scoring system dominate.

### Q4. What is Reciprocal Rank Fusion?

RRF combines multiple ranked lists using document positions rather than raw relevance scores.

### Q5. What is the difference between a bi-encoder and a cross-encoder?

A bi-encoder embeds inputs separately, allowing efficient similarity search.

A cross-encoder jointly processes a query and candidate document to estimate relevance.

Cross-encoders are generally more computationally expensive per candidate.

### Q6. Does reranking improve retrieval recall?

It can improve ranking quality among retrieved candidates.

However, **a reranker cannot recover a relevant document that was never retrieved**.

If the initial candidate-generation stage misses the correct document, reranking alone cannot fix that failure.

This is an important distinction between candidate recall and ranking quality.

### Q7. How would you reduce hybrid RAG latency?

Potential approaches include reducing candidate counts, caching embeddings, optimizing indexes, parallelizing independent retrieval operations, using smaller rerankers, and tuning concurrency.

Each optimization should be evaluated for its impact on retrieval quality.

---

# 🧠 Today's Knowledge Quiz

Try answering these without looking back.

**Q1.** What is the difference between sparse retrieval and dense retrieval?

**Q2.** Why does BM25 use inverse document frequency?

**Q3.** Suppose BM25 ranks Document A first, while dense retrieval ranks it third. How does RRF combine these rankings?

**Q4.** Why is a cross-encoder usually applied after initial retrieval rather than to every document in the database?

**Q5.** Your hybrid retrieval system has Recall@20 of 95%, but MRR@5 is poor. What does this suggest about the retrieval system, and which component might help?

**Q6.** Your reranker produces excellent results for general questions but performs poorly on exact error codes. What would you investigate?

**Q7 — Advanced System Design**

You are designing a production RAG system containing 20 million chunks.

Requirements:

- 500 concurrent users.
- Multi-tenant document isolation.
- Exact error-code retrieval.
- Semantic search.
- p95 retrieval latency below 500 ms.
- Daily document updates.
- Source citations in generated answers.

How would you design the lexical index, vector index, fusion layer, reranking service, caching strategy, and evaluation pipeline?

Explain how you would enforce tenant isolation and what trade-offs you would make if reranking pushes latency above the target.

---

# Session 12 Recap

Today, we improved our production RAG architecture by combining complementary retrieval methods.

We learned that dense embeddings are useful for semantic meaning, while BM25 helps identify exact terms and technical identifiers.

We implemented Reciprocal Rank Fusion to combine rankings and introduced cross-encoder reranking to improve the order of retrieved candidates.

Our advanced retrieval pipeline is:

\[
\boxed{
\text{Query}
\rightarrow
\begin{cases}
\text{BM25}\\
\text{Dense Retrieval}
\end{cases}
\rightarrow
\text{RRF}
\rightarrow
\text{Reranking}
\rightarrow
\text{Context}
\rightarrow
\text{LLM}
}
\]

**The most important lesson:** Retrieval quality depends on both finding the right candidates and ranking them correctly. Hybrid search can improve candidate coverage, while reranking can improve candidate ordering. Both must be evaluated against real queries.

---

## 🔜 Session 13: Production RAG Evaluation, Hallucination Detection & LLM-as-a-Judge

In our next session, we'll focus on measuring the quality of complete RAG answers rather than retrieval alone.

We'll study:

- Faithfulness and groundedness.
- Answer relevance and correctness.
- Hallucination detection.
- Context precision and recall.
- LLM-as-a-judge evaluation.
- Human evaluation and judge calibration.
- Automated RAG regression testing.
- Production evaluation dashboards.

**Practical project:** Build an evaluation pipeline that tests retrieved evidence, generated answers, and source citations, then tracks regressions across model and retrieval changes.

This will prepare us for later sessions on production AI observability, security, distributed system design, and cloud deployment.

**Today's priority:** Implement BM25, dense retrieval, RRF, and cross-encoder reranking. Then compare their performance using the same evaluation dataset.