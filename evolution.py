# evolution.py

from typing import Dict, List, Tuple
from collections import defaultdict
from config import TOPK_KEYWORDS, TOPIC_MATCH_THRESHOLD, SPLIT_MERGE_SECOND_BEST_MARGIN
from similarity import cosine_similarity_sparse
def build_vocab_inv(vocab: Dict[str, int]) -> Dict[int, str]:
    return {idx: term for term, idx in vocab.items()}


def topic_to_keyword_vector(
    topic_centroid: Dict[int, float],
    top_k: int,
) -> Dict[int, float]:
    items = list(topic_centroid.items())
    items.sort(key=lambda x: x[1], reverse=True)
    top_items = items[:top_k]
    return dict(top_items)


def match_topics_between_years(
    topics_prev: List,
    topics_next: List,
    top_k: int = TOPK_KEYWORDS,
    threshold: float = TOPIC_MATCH_THRESHOLD,
) -> Dict[int, List[Tuple[int, float]]]:
    matches: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    prev_kw_vectors = [
        topic_to_keyword_vector(t.centroid, top_k) for t in topics_prev
    ]
    next_kw_vectors = [
        topic_to_keyword_vector(t.centroid, top_k) for t in topics_next
    ]

    for i, vec_prev in enumerate(prev_kw_vectors):
        for j, vec_next in enumerate(next_kw_vectors):
            sim = cosine_similarity_sparse(vec_prev, vec_next)
            if sim >= threshold:
                matches[i].append((j, sim))

        matches[i].sort(key=lambda x: x[1], reverse=True)

    return matches


def classify_topic_transitions(
    matches_prev_to_next: Dict[int, List[Tuple[int, float]]],
    num_prev: int,
    num_next: int,
    second_best_margin: float = SPLIT_MERGE_SECOND_BEST_MARGIN,
) -> Dict[str, List[Tuple[int, ...]]]:
    continued = []
    split = []
    disappeared = []

    from collections import defaultdict
    next_to_prev_candidates: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for prev_idx in range(num_prev):
        candidates = matches_prev_to_next.get(prev_idx, [])
        if not candidates:
            disappeared.append((prev_idx,))
            continue

        best_next_idx, best_sim = candidates[0]
        continued.append((prev_idx, best_next_idx))

        if len(candidates) > 1:
            second_next_idx, second_sim = candidates[1]
            if best_sim - second_sim <= second_best_margin:
                involved = [prev_idx] + [idx for idx, _ in candidates]
                split.append(tuple(involved))

        for next_idx, sim in candidates:
            next_to_prev_candidates[next_idx].append((prev_idx, sim))

    merged = []
    for next_idx, prev_candidates in next_to_prev_candidates.items():
        if len(prev_candidates) > 1:
            prev_ids = [p_idx for p_idx, _ in prev_candidates]
            merged.append(tuple(prev_ids + [next_idx]))

    has_parent = set()
    for prev_idx, next_idx in continued:
        has_parent.add(next_idx)
    born = []
    for next_idx in range(num_next):
        if next_idx not in has_parent:
            born.append((next_idx,))

    return {
        "continued": continued,
        "split": split,
        "merged": merged,
        "disappeared": disappeared,
        "born": born,
    }
