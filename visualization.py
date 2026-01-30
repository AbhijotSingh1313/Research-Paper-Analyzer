
from typing import Dict, List
import matplotlib.pyplot as plt

from config import TOPK_KEYWORDS


def plot_topic_frequency_over_time(
    topic_counts_by_year: Dict[int, int],
    topic_label: str,
) -> None:

    years = sorted(topic_counts_by_year.keys())
    counts = [topic_counts_by_year[y] for y in years]

    plt.figure(figsize=(6, 4))
    plt.plot(years, counts, marker="o")
    plt.xlabel("Year")
    plt.ylabel("Number of documents")
    plt.title(f"Topic frequency over time: {topic_label}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_topic_keywords_table(
    year: int,
    topic_index: int,
    keywords: List[str],
) -> None:
    print(f"Year {year} - Topic {topic_index}")
    print("-" * 40)
    for k in keywords[:TOPK_KEYWORDS]:
        print(f"- {k}")
    print()
