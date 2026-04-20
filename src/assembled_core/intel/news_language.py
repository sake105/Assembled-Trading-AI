"""Lightweight language detection for news headlines.

Uses character- and word-level heuristics to detect the dominant language
of a headline. No external dependency.

Supported: en, de, fr, es, it, ru, zh, ja, ar, uk, pt, nl, sv, pl, tr,
he (Hebrew), el (Greek), vi (Vietnamese).
Anything unrecognised → "en" (fallback).

Design: rule-based, deterministic, O(n) in headline length. Intentionally
simple — for deep NLP, use a dedicated library (langdetect, fasttext).
"""

from __future__ import annotations

import re
from collections import Counter

# Unicode blocks that clearly identify a language family
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")          # Chinese / Japanese Kanji
_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")     # Japanese hiragana
_KATAKANA_RE = re.compile(r"[\u30a0-\u30ff]")     # Japanese katakana
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")       # Korean
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")       # Arabic
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")     # Russian / Ukrainian / Bulgarian
_HEBREW_RE = re.compile(r"[\u0590-\u05ff]")       # Hebrew
_GREEK_RE = re.compile(r"[\u0370-\u03ff]")        # Greek

# Small stopword lists per latin-script language (high signal, low overlap)
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset({"the", "and", "of", "to", "in", "for", "on", "is", "with", "a", "after", "as", "that"}),
    "de": frozenset({"der", "die", "das", "und", "ist", "mit", "für", "von", "im", "nach", "auf", "nicht", "auch", "eine"}),
    "fr": frozenset({"le", "la", "les", "et", "de", "du", "pour", "avec", "dans", "un", "une", "sur", "ne"}),
    "es": frozenset({"el", "la", "los", "las", "y", "de", "en", "con", "para", "por", "un", "una", "que", "no"}),
    "it": frozenset({"il", "la", "lo", "i", "gli", "le", "e", "di", "per", "con", "in", "un", "una", "non"}),
    "pt": frozenset({"o", "a", "os", "as", "e", "de", "em", "para", "com", "um", "uma", "não"}),
    "nl": frozenset({"de", "het", "en", "van", "voor", "met", "op", "is", "niet", "een"}),
    "sv": frozenset({"och", "att", "det", "som", "på", "inte", "med", "för", "av"}),
    "pl": frozenset({"i", "w", "na", "nie", "z", "jest", "się", "do", "że"}),
    "tr": frozenset({"ve", "bir", "bu", "için", "ile", "de", "da", "olan"}),
    # H8: Vietnamese uses Latin script with diacritics; stopwords disambiguate.
    "vi": frozenset({"và", "của", "là", "trong", "với", "không", "được", "đã", "có", "tại", "theo"}),
}

# Ukrainian vs Russian discriminator (Cyrillic charset alone is ambiguous)
_UK_CHARS = frozenset("їєіґ")
_RU_CHARS = frozenset("ыъэё")


def detect_language(text: str) -> str:
    """Detect the dominant language of `text`.

    Returns an ISO-639-1 code. Unknown → "en".
    """
    if not text:
        return "en"

    if _HIRAGANA_RE.search(text) or _KATAKANA_RE.search(text):
        return "ja"
    if _HANGUL_RE.search(text):
        return "ko"
    if _CJK_RE.search(text):
        # Could be Japanese kanji-only, but more likely Chinese if no kana present
        return "zh"
    if _ARABIC_RE.search(text):
        return "ar"
    if _HEBREW_RE.search(text):
        return "he"
    if _GREEK_RE.search(text):
        return "el"
    if _CYRILLIC_RE.search(text):
        low = text.lower()
        uk_hits = sum(1 for c in low if c in _UK_CHARS)
        ru_hits = sum(1 for c in low if c in _RU_CHARS)
        if uk_hits > ru_hits:
            return "uk"
        return "ru"

    # Latin-script: stopword voting. Include Vietnamese diacritics so
    # vi headlines tokenise correctly.
    tokens = re.findall(
        r"[a-zA-ZäöüÄÖÜßàâçéèêëîïôùûüÿñáíóúÿœæãõâêôơưăđáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]+",
        text.lower(),
    )
    if not tokens:
        return "en"

    scores: Counter[str] = Counter()
    for tok in tokens:
        for lang, stops in _STOPWORDS.items():
            if tok in stops:
                scores[lang] += 1

    if not scores:
        return "en"
    # H8: previously had a dead `best_count < 1` branch here — Counter only
    # tracks non-zero entries, so the check could never fire. Removed.
    best, _ = scores.most_common(1)[0]
    return best


def is_english(text: str) -> bool:
    """Convenience: True if headline is (likely) English."""
    return detect_language(text) == "en"


__all__ = ["detect_language", "is_english"]
