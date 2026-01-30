# clustering.py

from typing import List, Dict, Tuple

from config import SIMILARITY_THRESHOLD, MIN_DOCS_PER_TOPIC, TOPK_KEYWORDS
from similarity import cosine_similarity_sparse


Vector = Dict[int, float]


class Topic:
    def __init__(self, initial_vector: Vector, initial_doc_index: int):
        self.centroid: Vector = dict(initial_vector)
        self.doc_indices: List[int] = [initial_doc_index]

    def add_document(self, doc_vector: Vector, doc_index: int) -> None:
        self.doc_indices.append(doc_index)
        n = len(self.doc_indices)

        new_centroid: Vector = {}
        all_indices = set(self.centroid.keys()) | set(doc_vector.keys())
        for idx in all_indices:
            old_value = self.centroid.get(idx, 0.0)
            new_value = doc_vector.get(idx, 0.0)
            updated_avg = (old_value * (n - 1) + new_value) / n
            new_centroid[idx] = updated_avg

        self.centroid = new_centroid


def cluster_documents(
    tfidf_vectors: List[Vector],
) -> List[Topic]:
    topics: List[Topic] = []

    for doc_index, vec in enumerate(tfidf_vectors):
        if not topics:
            topics.append(Topic(vec, doc_index))
            continue

        best_sim = 0.0
        best_topic_idx = -1
        for t_idx, topic in enumerate(topics):
            sim = cosine_similarity_sparse(vec, topic.centroid)
            if sim > best_sim:
                best_sim = sim
                best_topic_idx = t_idx

        if best_sim >= SIMILARITY_THRESHOLD:
            topics[best_topic_idx].add_document(vec, doc_index)
        else:
            topics.append(Topic(vec, doc_index))

    filtered_topics = [
        topic for topic in topics
        if len(topic.doc_indices) >= MIN_DOCS_PER_TOPIC
    ]

    return filtered_topics


def get_topic_top_terms(
    topic: Topic,
    vocab_inv: Dict[int, str],
    top_k: int = TOPK_KEYWORDS,
) -> List[Tuple[str, float]]:
    items = list(topic.centroid.items())
    items.sort(key=lambda x: x[1], reverse=True)
    top_items = items[:top_k]

    result: List[Tuple[str, float]] = []
    for idx, weight in top_items:
        term = vocab_inv.get(idx, f"<UNK_{idx}>")
        result.append((term, weight))
    return result
