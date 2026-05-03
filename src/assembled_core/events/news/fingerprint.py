from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

_BIT_POSITIONS = np.arange(64, dtype=np.uint64)


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

    vector = np.zeros(64, dtype=np.int64)
    for token in tokens:
        h_bytes = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()[:8]
        h = np.uint64(int.from_bytes(h_bytes, byteorder="big", signed=False))
        set_mask = (h >> _BIT_POSITIONS) & np.uint64(1)
        vector += np.where(set_mask, np.int64(1), np.int64(-1))

    fp = 0
    for b in np.where(vector > 0)[0]:
        fp |= 1 << int(b)
    return fp


def hamming_distance(a: int, b: int) -> int:
    """Compute Hamming distance between two 64-bit integers."""
    return (a ^ b).bit_count()


__all__ = ["simhash64", "hamming_distance"]
