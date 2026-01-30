import re
from typing import List

from config import MIN_TOKEN_LENGTH, LOWERCASE
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to",
    "is", "are", "was", "were", "be", "being", "been",
    "this", "that", "these", "those",
    "it", "its", "as", "by", "with", "from",
    "we", "our", "they", "their", "you", "your",
    "at", "into", "about", "than", "then", "but",
}
TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")
def tokenize(text: str) -> List[str]:
    if LOWERCASE:
        text = text.lower()

    tokens = TOKEN_PATTERN.findall(text)
    return tokens
def clean_tokens(tokens: List[str]) -> List[str]:
    cleaned = []
    for tok in tokens:
        if len(tok) < MIN_TOKEN_LENGTH:
            continue
        if tok in STOPWORDS:
            continue
        cleaned.append(tok)
    return cleaned
def preprocess_text(text: str) -> List[str]:
    tokens = tokenize(text)
    tokens = clean_tokens(tokens)
    return tokens
