from __future__ import annotations

import math
import re
from typing import Dict, List

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "from",
    "breaking",
    "update",
    "live",
    "news",
}


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alnum, filter short tokens and stopwords."""
    if not text:
        return []
    lowered = text.lower()
    parts = re.split(r"[^0-9a-z]+", lowered)
    tokens = [p for p in parts if len(p) >= 3]
    return [t for t in tokens if t not in STOPWORDS]


def build_tfidf_vectors(texts: List[str]) -> List[Dict[str, float]]:
    """Build simple TF-IDF vectors for a list of texts."""
    tokenized: List[List[str]] = [tokenize(t) for t in texts]
    N = len(tokenized)
    if N == 0:
        return []

    # Document frequency
    df: Dict[str, int] = {}
    for tokens in tokenized:
        seen = set(tokens)
        for tok in seen:
            df[tok] = df.get(tok, 0) + 1

    # IDF
    idf: Dict[str, float] = {}
    for tok, freq in df.items():
        idf[tok] = math.log((N + 1) / (freq + 1)) + 1.0

    # TF-IDF per document
    vectors: List[Dict[str, float]] = []
    for tokens in tokenized:
        if not tokens:
            vectors.append({})
            continue
        counts: Dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        total = float(len(tokens))
        vec: Dict[str, float] = {}
        for tok, cnt in counts.items():
            tf = cnt / total
            vec[tok] = tf * idf.get(tok, 0.0)
        vectors.append(vec)
    return vectors


def cosine_sparse(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Cosine similarity for sparse TF-IDF vectors."""
    if not v1 or not v2:
        return 0.0
    # Intersection for dot product
    if len(v1) < len(v2):
        a, b = v1, v2
    else:
        a, b = v2, v1
    dot = 0.0
    for tok, w in a.items():
        if tok in b:
            dot += w * b[tok]
    if dot <= 0.0:
        return 0.0
    norm1 = math.sqrt(sum(w * w for w in v1.values()))
    norm2 = math.sqrt(sum(w * w for w in v2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


__all__ = ["tokenize", "build_tfidf_vectors", "cosine_sparse"]
