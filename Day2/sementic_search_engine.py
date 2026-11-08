from sentence_transformers import SentenceTransformer

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


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


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



query = "Why are all my database connections busy?"

query_embedding = model.encode(
    query,
    normalize_embeddings=True,
    convert_to_numpy=True
)

print(query_embedding.shape)


import numpy as np

similarities = (
    document_embeddings @ query_embedding
)

print(similarities)


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
