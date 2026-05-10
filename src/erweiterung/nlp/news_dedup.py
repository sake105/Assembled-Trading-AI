"""News-Deduplication via SimHash + Title-Similarity.

Theorie
-------
News-Aggregator-Feeds enthalten viele Duplikate (gleiche Story, mehrere Quellen).
Für Sentiment-Aggregation sind Duplikate problematisch — sie erzeugen
artificial sentiment-magnitude.

Methoden
--------
1. **MinHash + LSH**: schnelle Near-Duplicate-Detection (Broder 1997).
2. **SimHash (Charikar 2002)**: 64-bit Fingerprint, Hamming-Distance < 3 = Dup.
3. **Title-Token-Jaccard**: einfacher Baseline.

Implementation
--------------
Wir bieten SimHash + Token-Jaccard. MinHash ist optional (datasketch lib).
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower()) if isinstance(text, str) else []


def simhash(text: str, n_bits: int = 64) -> int:
    """Charikar SimHash."""
    tokens = _tokenize(text)
    if not tokens:
        return 0
    v = [0] * n_bits
    for tok in tokens:
        h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16) & (
            (1 << n_bits) - 1
        )  # noqa: S324
        for i in range(n_bits):
            bit = (h >> i) & 1
            if bit:
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = 0
    for i, val in enumerate(v):
        if val > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def jaccard_similarity(text_a: str, text_b: str) -> float:
    a = set(_tokenize(text_a))
    b = set(_tokenize(text_b))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(
    df: pd.DataFrame,
    text_col: str = "headline",
    by_date: str = "date",
    simhash_bits_threshold: int = 3,
    jaccard_threshold: float = 0.8,
) -> pd.DataFrame:
    """Entferne Near-Duplicates innerhalb derselben Date-Group.

    Args:
        df: DataFrame mit text + date.
        text_col, by_date: column names.
        simhash_bits_threshold: Hamming-Distance ≤ threshold = duplicate.
        jaccard_threshold: Token-Jaccard ≥ threshold = duplicate.

    Returns:
        DataFrame mit unique-Hits.
    """
    if df.empty:
        return df
    out_rows = []
    for d, g in df.groupby(by_date):
        seen_hashes: list[tuple[int, str]] = []
        for _, r in g.iterrows():
            text = r.get(text_col, "")
            h = simhash(text)
            is_dup = False
            for h_seen, t_seen in seen_hashes:
                if hamming_distance(h, h_seen) <= simhash_bits_threshold:
                    is_dup = True
                    break
                if jaccard_similarity(text, t_seen) >= jaccard_threshold:
                    is_dup = True
                    break
            if not is_dup:
                seen_hashes.append((h, text))
                out_rows.append(r)
    return pd.DataFrame(out_rows)


__all__ = [
    "simhash",
    "hamming_distance",
    "jaccard_similarity",
    "deduplicate",
]
