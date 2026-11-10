# --8<-- [start:lst-1.5]
# src/testing/semantic_assert.py
import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(text_a: str, text_b: str) -> float:
    embeddings = _model.encode([text_a, text_b], convert_to_numpy=True)
    vec_a, vec_b = embeddings[0], embeddings[1]
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

def assert_semantic_match(
    actual: str,
    expected: str,
    threshold: float = 0.85,
) -> None:
    similarity = semantic_similarity(actual, expected)
    if similarity < threshold:
        raise AssertionError(
            f"Semantic similarity too low: {similarity:.3f} < {threshold:.3f}")
# --8<-- [end:lst-1.5]
