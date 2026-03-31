from __future__ import annotations

import hashlib
import re
from typing import List


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alnum, keep tokens with length >= 3."""
    if not text:
        return []
    lowered = text.lower()
    parts = re.split(r"[^0-9a-z]+", lowered)
    return [p for p in parts if len(p) >= 3]


def simhash64(text: str) -> int:
    """Compute 64-bit SimHash fingerprint for the given text."""
    tokens = _tokenize(text)
    if not tokens:
        return 0

    vector = [0] * 64
    for token in tokens:
        h_bytes = hashlib.md5(token.encode("utf-8")).digest()[:8]
        h = int.from_bytes(h_bytes, byteorder="big", signed=False)
        for bit in range(64):
            if h & (1 << bit):
                vector[bit] += 1
            else:
                vector[bit] -= 1

    fp = 0
    for bit in range(64):
        if vector[bit] > 0:
            fp |= 1 << bit
    return fp


def hamming_distance(a: int, b: int) -> int:
    """Compute Hamming distance between two 64-bit integers."""
    return (a ^ b).bit_count()


__all__ = ["simhash64", "hamming_distance"]
