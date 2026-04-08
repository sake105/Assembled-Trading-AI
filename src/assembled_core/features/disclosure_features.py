"""Disclosure Text Complexity Features (Plan 3.10).

Features from SEC filings:
- Fog Index on Risk Factors section
- Sentiment change between filings
- Document length change YoY
"""

from __future__ import annotations

import re



def compute_fog_index(text: str) -> float:
    """Gunning Fog Index: higher = more complex = worse future performance.

    ``FOG = 0.4 × (avg_sentence_length + pct_complex_words)``
    Complex word: 3+ syllables.
    """
    if not text or len(text) < 50:
        return 0.0

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    if not sentences:
        return 0.0

    words = text.split()
    n_words = len(words)
    n_sentences = len(sentences)

    if n_words == 0 or n_sentences == 0:
        return 0.0

    # Simple syllable count heuristic
    def count_syllables(word: str) -> int:
        word = word.lower().strip(".,;:!?")
        vowels = "aeiou"
        count = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(1, count)

    complex_words = sum(1 for w in words if count_syllables(w) >= 3)
    avg_sentence_len = n_words / n_sentences
    pct_complex = (complex_words / n_words) * 100

    return round(0.4 * (avg_sentence_len + pct_complex), 2)


def compute_filing_length_change(current_length: int, prior_length: int) -> float:
    """YoY change in filing length. Increase often = more risks to report."""
    if prior_length <= 0:
        return 0.0
    return (current_length - prior_length) / prior_length


__all__ = ["compute_fog_index", "compute_filing_length_change"]
