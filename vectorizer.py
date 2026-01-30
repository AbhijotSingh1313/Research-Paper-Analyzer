import math
from collections import Counter
from typing import List, Dict, Tuple

from config import USE_LOG_TF, SMOOTH_IDF, NORMALIZE_TFIDF
from preprocessing import preprocess_text


def build_vocabulary(docs_tokens: List[List[str]]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    index = 0
    for tokens in docs_tokens:
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = index
                index += 1
    return vocab


def compute_document_frequencies(
    docs_tokens: List[List[str]],
    vocab: Dict[str, int],
) -> List[int]:

    vocab_size = len(vocab)
    df = [0] * vocab_size

    for tokens in docs_tokens:
        unique_tokens = set(tokens)
        for tok in unique_tokens:
            idx = vocab[tok]
            df[idx] += 1

    return df


def compute_idf(num_docs: int, df: List[int]) -> List[float]:
    idf_values: List[float] = []
    for doc_freq in df:
        if doc_freq == 0:
            idf_values.append(0.0)
            continue

        if SMOOTH_IDF:
            value = math.log((1.0 + num_docs) / (1.0 + doc_freq)) + 1.0
        else:
            value = math.log(num_docs / float(doc_freq))

        idf_values.append(value)

    return idf_values


def compute_tf_vector(
    tokens: List[str],
    vocab: Dict[str, int],
) -> Dict[int, float]:
    counts = Counter(tokens)
    tf_vector: Dict[int, float] = {}

    for tok, count in counts.items():
        if tok not in vocab:
            continue
        idx = vocab[tok]
        if USE_LOG_TF:
            tf_value = math.log(1.0 + count)
        else:
            tf_value = float(count)
        tf_vector[idx] = tf_value

    return tf_vector


def compute_tfidf_vector(
    tf_vector: Dict[int, float],
    idf: List[float],
    normalize: bool = NORMALIZE_TFIDF,
) -> Dict[int, float]:
    tfidf: Dict[int, float] = {}

    for idx, tf_value in tf_vector.items():
        tfidf_value = tf_value * idf[idx]
        tfidf[idx] = tfidf_value

    if not normalize:
        return tfidf
    norm_sq = 0.0
    for value in tfidf.values():
        norm_sq += value * value

    if norm_sq == 0.0:
        return tfidf

    norm = math.sqrt(norm_sq)
    for idx in list(tfidf.keys()):
        tfidf[idx] = tfidf[idx] / norm

    return tfidf
def build_tfidf_for_year(
    docs: List[str],
) -> Tuple[List[Dict[int, float]], Dict[str, int], List[float]]:
    docs_tokens = [preprocess_text(text) for text in docs]
    vocab = build_vocabulary(docs_tokens)
    num_docs = len(docs_tokens)
    df = compute_document_frequencies(docs_tokens, vocab)
    idf = compute_idf(num_docs, df)

    tfidf_vectors: List[Dict[int, float]] = []
    for tokens in docs_tokens:
        tf = compute_tf_vector(tokens, vocab)
        tfidf = compute_tfidf_vector(tf, idf)
        tfidf_vectors.append(tfidf)

    return tfidf_vectors, vocab, idf
