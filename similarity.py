from typing import Dict
def dot_product(vec_a: Dict[int, float], vec_b: Dict[int, float]) -> float:
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    total = 0.0
    for idx, value_a in vec_a.items():
        if idx in vec_b:
            total += value_a * vec_b[idx]
    return total
def cosine_similarity_sparse(
    vec_a: Dict[int, float],
    vec_b: Dict[int, float],
) -> float:
    return dot_product(vec_a, vec_b)
