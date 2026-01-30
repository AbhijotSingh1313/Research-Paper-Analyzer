
from typing import Dict, List

from config import DATA_PATH, TOPK_KEYWORDS
from data_loader import load_papers_by_year
from vectorizer import build_tfidf_for_year
from clustering import cluster_documents, get_topic_top_terms
from evolution import build_vocab_inv, match_topics_between_years, classify_topic_transitions


def run_for_all_years() -> None:
    papers_by_year = load_papers_by_year(DATA_PATH)
    years = sorted(papers_by_year.keys())
    year_topics = {}
    year_vocab = {}
    year_idf = {}

    for year in years:
        papers = papers_by_year[year]
        abstracts = [p["abstract"] for p in papers]

        print(f"\n=== Processing year {year} with {len(abstracts)} abstracts ===")
        tfidf_vectors, vocab, idf = build_tfidf_for_year(abstracts)
        year_vocab[year] = vocab
        year_idf[year] = idf
        topics = cluster_documents(tfidf_vectors)
        year_topics[year] = topics

        print(f"Year {year}: found {len(topics)} topics")
        vocab_inv = build_vocab_inv(vocab)
        for t_idx, topic in enumerate(topics):
            top_terms = get_topic_top_terms(topic, vocab_inv, TOPK_KEYWORDS)
            keywords_only = [term for term, _ in top_terms]
            print(f"  Topic {t_idx} (docs={len(topic.doc_indices)}): {', '.join(keywords_only)}")
    for i in range(len(years) - 1):
        y_prev = years[i]
        y_next = years[i + 1]
        print(f"\n=== Topic evolution from {y_prev} to {y_next} ===")
        topics_prev = year_topics[y_prev]
        topics_next = year_topics[y_next]

        matches = match_topics_between_years(
            topics_prev,
            topics_next,
            top_k=TOPK_KEYWORDS,
            threshold=0.3,
        )

        transitions = classify_topic_transitions(
            matches_prev_to_next=matches,
            num_prev=len(topics_prev),
            num_next=len(topics_next),
            second_best_margin=0.15,
        )

        print("Continued topics:", transitions["continued"])
        print("Splits:", transitions["split"])
        print("Merges:", transitions["merged"])
        print("Disappeared:", transitions["disappeared"])
        print("Born:", transitions["born"])


if __name__ == "__main__":
    run_for_all_years()
