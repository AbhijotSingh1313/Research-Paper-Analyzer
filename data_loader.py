import csv
from collections import defaultdict
from typing import List, Dict, Any


def load_papers_by_year(csv_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load papers from a CSV file and group them by year.

    Expected columns: id, title, abstract, year

    Returns a dict: {year: [paper_dict, ...]}
    """
    papers_by_year: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                year = int(row["year"])
            except (KeyError, ValueError):

                continue

            paper = {
                "id": row.get("id", "").strip(),
                "title": row.get("title", "").strip(),
                "abstract": row.get("abstract", "").strip(),
                "year": year,
            }
            papers_by_year[year].append(paper)

    return dict(papers_by_year)
