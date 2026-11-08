# AI Engineer Mentorship — Session 2
## Tokenization, Embeddings & Building Your First Semantic Search Engine

**Level:** Intermediate → Advanced AI Engineering  
**Duration:** 60 minutes  
**Prerequisite:** Neural networks, forward propagation, gradients, and PyTorch training loops.

In our previous session, we built a neural network with PyTorch and learned how models update their weights through backpropagation.

Today, we'll connect those foundations to **Large Language Models (LLMs)** and build something directly relevant to production AI engineering: a semantic search system.

By the end of this lesson, you'll understand how text becomes numbers, how embedding models represent meaning, and how to retrieve relevant information without relying on exact keyword matches.

---

# 1. Today's Learning Objectives

After completing this session, you should be able to:

1. Explain tokenization and why LLMs process tokens instead of raw text.
2. Understand embedding vectors and their relationship to neural networks.
3. Calculate cosine similarity between vectors.
4. Build a semantic search engine using Python.
5. Understand how semantic search becomes the retrieval component of RAG.
6. Identify production challenges involving indexing, latency, data security, and evaluation.

### 60-Minute Agenda

| Time | Topic |
|---|---|
| 0–10 min | Tokenization: converting text into tokens |
| 10–20 min | Embeddings: representing meaning mathematically |
| 20–30 min | Vector similarity and semantic search |
| 30–45 min | Hands-on Python semantic search project |
| 45–52 min | Production architecture and engineering trade-offs |
| 52–60 min | Assignment, interview preparation, and quiz |

---

# 2. Tokenization: How LLMs Read Text
**Time: 0–10 minutes**

Consider this sentence:

> My PostgreSQL database is responding slowly.

As a backend developer, you see a Python string.

```python
text = "My PostgreSQL database is responding slowly."
```

But a neural network cannot directly perform matrix multiplication on strings.

It needs numerical inputs.

The first major step is **tokenization**.

## What is a token?

A token is a unit of text recognized by a tokenizer.

Depending on the tokenizer, a token might represent an entire word, part of a word, a character, or even a byte sequence.

For example, a tokenizer might divide a sentence into pieces resembling:

```text
"My"
"Post"
"gre"
"SQL"
"database"
"is"
"responding"
"slowly"
"."
```

This is an illustration, not the guaranteed output of a particular tokenizer.

Different tokenizers produce different token sequences.

Each token is mapped to a numerical ID.

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embedding Layer
 ↓
Numerical Vectors
 ↓
Transformer
```

An important distinction:

**Token IDs are identifiers, not representations of semantic meaning.**

If token ID `1024` represents one word and token ID `1025` represents another, that does not mean those words are semantically close.

Their meanings are learned through embedding representations.

## Python example

Install the required library:

```bash
pip install transformers
```

Then run:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

text = "My PostgreSQL database is responding slowly."

tokens = tokenizer.tokenize(text)

token_ids = tokenizer.encode(text)

print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Number of tokens:", len(token_ids))
```

Notice that `encode()` may include special tokens that `tokenize()` does not display.

### Interview insight

**Question: Why can't we just split text using `text.split()`?**

Whitespace splitting cannot reliably handle subwords, punctuation, languages without whitespace-separated words, and previously unseen vocabulary.

Modern tokenizers use learned or algorithmic segmentation strategies that allow models to represent a much broader range of text.

---

# 3. Embeddings: How Models Represent Meaning
**Time: 10–20 minutes**

Now comes one of the most important concepts in AI engineering.

Suppose we have these sentences:

**Sentence A:** "The database connection pool is exhausted."

**Sentence B:** "All PostgreSQL connections are currently occupied."

**Sentence C:** "The customer purchased a new smartphone."

Sentence A and Sentence B express related meanings despite using different words.

Traditional keyword search may struggle to recognize that relationship.

Embedding models are designed to map text into numerical vectors that capture useful semantic relationships.

## What is an embedding?

An embedding is a numerical vector.

For example:

```python
embedding = [
    0.12,
    -0.45,
    0.78,
    0.21
]
```

This example has four dimensions.

Real embedding models often produce hundreds or thousands of dimensions.

The model we'll use today produces **384-dimensional embeddings**.

Mathematically:

\[
E(text) \rightarrow \mathbb{R}^{384}
\]

Here, \(E\) represents the embedding model.

For each input sentence, it produces a vector containing 384 numerical values.

## Where do these values come from?

Remember our previous neural-network lesson?

We learned that neural networks contain trainable weights.

During training, those weights are adjusted to minimize a loss function.

Embedding models also use learned parameters.

A simplified sentence-embedding pipeline looks like this:

```text
Input Sentence
      ↓
Tokenizer
      ↓
Token IDs
      ↓
Token Embeddings
      ↓
Transformer Layers
      ↓
Contextual Token Representations
      ↓
Pooling
      ↓
Sentence Embedding
```

**Pooling** combines information from multiple token representations into one fixed-length vector.

Different embedding models use different pooling or representation strategies.

### Important distinction

A token embedding represents an individual token within a model.

A sentence embedding represents an entire sentence or text passage.

For semantic search, we generally want an embedding that represents the meaning of a complete passage.

## Why does this matter?

Imagine building an AI assistant for your company's backend documentation.

A developer asks:

> Why is my API timing out?

Your documentation might contain:

> Requests exceed the configured gateway timeout when downstream services respond slowly.

These sentences have different wording but related meaning.

Semantic search can help retrieve that documentation.

This capability becomes essential when we start building RAG systems.

---

# 4. Cosine Similarity: Comparing Embeddings
**Time: 20–30 minutes**

Once we have embeddings, how do we compare them?

One common approach is **cosine similarity**.

For two vectors \(A\) and \(B\):

\[
\cos(\theta)=\frac{A\cdot B}{\|A\|\|B\|}
\]

Here:

- \(A\cdot B\) is the dot product.
- \(\|A\|\) is the magnitude of vector A.
- \(\|B\|\) is the magnitude of vector B.

Cosine similarity measures how closely the vectors point in the same direction.

Its mathematical range is -1 to 1.

For many text embedding models, semantically related passages tend to have higher similarity scores.

However, **a similarity score is not a calibrated probability that a document is correct or relevant**.

## Python example

```python
import numpy as np

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)

    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    return dot_product / (
        magnitude_a * magnitude_b
    )

vector_a = np.array([1.0, 2.0, 3.0])
vector_b = np.array([2.0, 4.0, 6.0])
vector_c = np.array([3.0, -2.0, 0.0])

print(cosine_similarity(vector_a, vector_b))
print(cosine_similarity(vector_a, vector_c))
```

Output:

```text
1.0
-0.0741...
```

Vector B points in exactly the same direction as vector A.

Vector C points in a different direction.

### Optimization insight

If vectors are already normalized to unit length:

\[
\|A\|=\|B\|=1
\]

Then:

\[
\cos(\theta)=A\cdot B
\]

This means cosine similarity can be calculated using a simple dot product.

That is useful when building fast retrieval systems.

---

# 5. Hands-On Project: Build a Semantic Search Engine
**Time: 30–45 minutes**

Let's build a working project.

### Project: Backend Documentation Search

Imagine your company maintains technical documentation for:

- PostgreSQL connection issues
- Redis caching
- Django authentication
- Celery background jobs
- API performance
- Docker deployment

We want developers to search those documents using natural language.

Instead of searching only for exact keywords, we'll compare embedding vectors.

## Step 1: Install dependencies

```bash
pip install sentence-transformers numpy
```

We'll use the pretrained embedding model:

`sentence-transformers/all-MiniLM-L6-v2`

The first execution downloads the model. Subsequent runs can use the local cache.

## Step 2: Create sample documents

```python
documents = [
    {
        "id": "doc_1",
        "title": "PostgreSQL Connections",
        "content": (
            "Database requests may fail when the "
            "PostgreSQL connection pool is exhausted. "
            "Check active connections and pool limits."
        )
    },
    {
        "id": "doc_2",
        "title": "Redis Cache",
        "content": (
            "Redis caching reduces database queries "
            "and improves API response times."
        )
    },
    {
        "id": "doc_3",
        "title": "Django Authentication",
        "content": (
            "Django REST Framework supports JWT "
            "authentication for securing API endpoints."
        )
    },
    {
        "id": "doc_4",
        "title": "Celery Workers",
        "content": (
            "Celery executes background tasks "
            "asynchronously using workers and "
            "a message broker."
        )
    },
    {
        "id": "doc_5",
        "title": "API Performance",
        "content": (
            "Slow API responses can result from "
            "inefficient database queries, "
            "network latency, or overloaded services."
        )
    }
]
```

These documents form our initial knowledge base.

## Step 3: Load the embedding model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
```

This is an inference operation.

We're using a pretrained model rather than training one from scratch.

## Step 4: Generate document embeddings

```python
texts = [
    doc["content"]
    for doc in documents
]

document_embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    convert_to_numpy=True
)

print(document_embeddings.shape)
```

Expected output:

```text
(5, 384)
```

Think about this shape.

We have five documents.

Each document has a 384-dimensional embedding.

Our embedding matrix is:

\[
D \in \mathbb{R}^{5\times384}
\]

This connects directly to the tensor-shape concepts you studied in PyTorch.

## Step 5: Embed a user query

```python
query = "Why are all my database connections busy?"

query_embedding = model.encode(
    query,
    normalize_embeddings=True,
    convert_to_numpy=True
)

print(query_embedding.shape)
```

Expected output:

```text
(384,)
```

Now both our documents and query are represented in the same vector space.

## Step 6: Calculate similarity

```python
import numpy as np

similarities = (
    document_embeddings @ query_embedding
)

print(similarities)
```

Because we normalized the vectors, matrix multiplication computes cosine similarities.

Mathematically:

\[
(5\times384)\cdot(384)\rightarrow(5)
\]

We get one similarity score per document.

## Step 7: Return the most relevant documents

```python
top_k = 3

top_indices = np.argsort(
    similarities
)[::-1][:top_k]

for index in top_indices:

    doc = documents[index]

    print(
        f"Title: {doc['title']}"
    )

    print(
        f"Score: {similarities[index]:.4f}"
    )

    print(
        f"Content: {doc['content']}"
    )

    print("-" * 50)
```

For this query, the PostgreSQL connection document is a strong candidate for retrieval.

The actual ranking depends on the embedding model's learned representations.

### What have we built?

Our search engine performs:

```text
Developer Question
        ↓
Embedding Model
        ↓
Query Vector
        ↓
Compare with Document Vectors
        ↓
Rank Documents
        ↓
Return Top Results
```

**Congratulations—you now have the retrieval foundation of a RAG application.**

But we haven't built a complete RAG system yet.

RAG adds a generation step where an LLM uses retrieved information to produce an answer.

---

# 6. From Semantic Search to Production RAG
**Time: 45–52 minutes**

Let's examine the difference between our current implementation and a production architecture.

## Current architecture

```text
Documents
    ↓
Embedding Model
    ↓
NumPy Array
    ↓
Similarity Search
    ↓
Results
```

This is suitable for a small prototype.

However, what happens when your company has one million documents?

Our current approach compares the query against every document embedding.

For \(N\) documents with embedding dimension \(d\), the similarity calculation requires approximately:

\[
O(Nd)
\]

For small datasets, exact search can be perfectly reasonable.

At larger scales, we may need a specialized vector index.

## Production architecture

```text
                  INGESTION PIPELINE

Documents / PDFs / Database Records
                  ↓
             Data Cleaning
                  ↓
             Text Chunking
                  ↓
             Embedding Model
                  ↓
              Vector Index
                  ↓
            Metadata Storage


                    QUERY PATH

               User Question
                     ↓
              Authentication
                     ↓
               Query Embedding
                     ↓
          Permission-Filtered Search
                     ↓
               Top-K Documents
                     ↓
                 Reranking
                     ↓
              Context Assembly
                     ↓
                     LLM
                     ↓
             Grounded Response
```

Let's understand three important production concepts.

### A. Chunking

Imagine a PDF containing 100 pages.

Embedding the entire document as one vector may lose important details or exceed the embedding model's input limit.

Instead, divide it into smaller passages.

```text
Document
   ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
   ↓
Individual Embeddings
```

A chunk should contain enough context to be meaningful while remaining focused enough for accurate retrieval.

Chunk size and overlap are tuning parameters, not universal constants.

### B. Vector databases

Vector databases or vector indexes help retrieve nearby embeddings efficiently.

Common approaches include approximate nearest-neighbor indexing, such as HNSW.

Because you already work with PostgreSQL, [pgvector](https://github.com/pgvector/pgvector?utm_source=chatgpt.com) is a useful starting point.

It supports vector storage and similarity search inside PostgreSQL.

A simple conceptual table might contain:

```sql
CREATE TABLE document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384)
);
```

This assumes the pgvector extension has been installed and enabled.

In production, you would also need document metadata, tenant identifiers, access controls, and appropriate indexes.

### C. Security

Suppose your application serves multiple organizations.

A user from Organization A must never retrieve private documents belonging to Organization B.

Vector similarity alone does not enforce permissions.

Your retrieval layer must apply authorization constraints.

Also, retrieved documents are untrusted data. They can contain malicious instructions intended to manipulate the LLM.

This is why production RAG requires both access control and prompt-injection defenses.

---

# 7. Today's Practical Assignment
**Time: 52–57 minutes**

## Mini-project: Semantic Search API

Your task is to convert today's semantic search implementation into a FastAPI application.

### Required endpoint

```http
POST /api/v1/search
```

Request:

```json
{
  "query": "Why is PostgreSQL refusing new connections?",
  "top_k": 3
}
```

Expected response structure:

```json
{
  "query": "Why is PostgreSQL refusing new connections?",
  "results": [
    {
      "document_id": "doc_1",
      "title": "PostgreSQL Connections",
      "score": 0.0,
      "content": "Database requests may fail when the PostgreSQL connection pool is exhausted."
    }
  ]
}
```

The score shown is only a placeholder for the response schema. Your implementation should return the actual computed similarity.

### Engineering requirements

Your application should load the embedding model once during startup rather than loading it for every HTTP request.

It should also generate document embeddings once, validate incoming requests, enforce a reasonable `top_k` limit, and return ranked results.

**Bonus challenge:** Add a `/health` endpoint and measure search latency using `time.perf_counter()`.

This exercise connects your existing FastAPI/backend experience with your new AI knowledge.

---

# 8. Senior AI Engineer Interview Insights
**Time: 57–60 minutes**

### Question 1: What is the difference between tokenization and embedding?

**Answer:** Tokenization divides text into recognized units and maps them to IDs. Embedding converts tokens or larger text units into dense numerical representations that can encode learned relationships.

### Question 2: Why do we normalize embeddings?

**Answer:** Normalization makes vector magnitude equal to one, allowing cosine similarity to be calculated using a dot product. It can also make similarity calculations more consistent when vector direction is the intended comparison.

### Question 3: What is the difference between semantic search and RAG?

**Answer:** Semantic search retrieves relevant information. RAG combines retrieval with a generative model that uses retrieved context to produce an answer.

### Question 4: Why might semantic search fail even when embeddings are high quality?

**Answer:** Relevant information may be missing, poorly chunked, incorrectly indexed, outdated, or excluded by filters. Queries involving exact identifiers, error codes, or numbers may also benefit from keyword or hybrid retrieval.

### Question 5: How would you evaluate retrieval quality?

**Answer:** Create a test dataset containing queries and known relevant documents.

Useful metrics include:

| Metric | Meaning |
|---|---|
| Recall@K | Fraction of relevant documents retrieved in the top K |
| Precision@K | Fraction of top-K results that are relevant |
| MRR | How highly the first relevant result is ranked |
| Retrieval latency | Time required to return results |

You should evaluate retrieval quality separately from the quality of LLM-generated answers.

---

# Session Recap

Today, we connected neural-network fundamentals to practical AI engineering.

The most important concept is this:

\[
\text{Text}\rightarrow\text{Tokens}\rightarrow\text{Embeddings}\rightarrow\text{Similarity Search}
\]

You learned how pretrained embedding models represent text, how cosine similarity compares vectors, and how to build a semantic search engine in Python.

You also saw how this small project evolves into a production retrieval architecture involving chunking, vector indexing, access control, and evaluation.

For additional study, use the official [Sentence Transformers documentation](https://sbert.net/?utm_source=chatgpt.com).

## Knowledge Check

Before the next session, make sure you can answer these five questions without referring to the explanations above.

**Q1.** Why can't a transformer process raw Python strings directly?

**Q2.** If you have 10,000 documents and each embedding contains 384 dimensions, what is the shape of your embedding matrix?

**Q3.** Why does normalizing embeddings allow us to use a dot product for cosine similarity?

**Q4.** Why might an exact error-code search work better with keyword retrieval than pure semantic search?

**Q5.** What additional components are required to turn our semantic search engine into a secure production RAG application?

---

## Next Session — RAG Engineering: Chunking, Retrieval & Context Construction

We'll build directly on today's project rather than repeat embedding fundamentals.

Our next session will cover document ingestion, chunking strategies, retrieval quality, hybrid search, reranking, and assembling reliable context for an LLM.

We'll also examine one of the most common production problems: **why an LLM can generate an incorrect answer even when the correct information exists in your knowledge base.**

**Your priority today:** Implement the FastAPI semantic search endpoint, verify the embedding shapes, and understand every operation in the similarity-search calculation.