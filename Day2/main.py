from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

text = "My PostgreSQL database is responding slowly."

tokens = tokenizer.tokenize(text)

token_ids = tokenizer.encode(text)

# print("Tokens:", tokens)
# print("Token IDs:", token_ids)
# print("Number of tokens:", len(token_ids))


# Cosine similarity function



def cosine_similarity(a, b):
    dot_product = np.dot(a, b)

    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    return dot_product / (
        magnitude_a * magnitude_b
    )

vector_a = np.array([1.0, 2.0, 3.0])
vector_b = np.array([2.0, 4.0, 6.0])
vector_c = np.array([3.0, 6.0, 9.0])

print(cosine_similarity(vector_a, vector_b))
print(cosine_similarity(vector_a, vector_c))
