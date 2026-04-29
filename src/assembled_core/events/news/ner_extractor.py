"""Named Entity Recognition for news headlines and text.

From 11_FREE_MODELLE.md §11.15.
Fast path: spaCy en_core_web_lg + cashtag regex.
Slow path: GLiNER zero-shot for low-confidence entities.

Company-to-ticker mapping hierarchy:
1. Local alias cache (fastest)
2. OpenFIGI v3 (free, MIT) — V2 sunset 01.07.2026
3. Wikidata SPARQL (P414=ISIN, P249=Ticker)
4. yfinance fallback

Install: pip install spacy gliner && python -m spacy download en_core_web_lg
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Cashtag pattern: $AAPL, $BRK.B, etc.
_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5}(?:\.[A-B])?)\b")

# Minimal alias cache — top 50 companies
_COMPANY_TO_TICKER: dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "berkshire": "BRK-B",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "exxon": "XOM",
    "johnson": "JNJ",
    "visa": "V",
    "walmart": "WMT",
    "mastercard": "MA",
    "unitedhealth": "UNH",
    "procter": "PG",
    "chevron": "CVX",
    "home depot": "HD",
    "abbvie": "ABBV",
    "eli lilly": "LLY",
    "broadcom": "AVGO",
    "costco": "COST",
    "merck": "MRK",
    "pepsico": "PEP",
    "coca-cola": "KO",
    "amd": "AMD",
    "intel": "INTC",
    "netflix": "NFLX",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "qualcomm": "QCOM",
    "boeing": "BA",
    "caterpillar": "CAT",
    "3m": "MMM",
    "ibm": "IBM",
    "ge": "GE",
    "ford": "F",
    "general motors": "GM",
    "palantir": "PLTR",
    "uber": "UBER",
    "airbnb": "ABNB",
    "coinbase": "COIN",
    "snowflake": "SNOW",
    "crowdstrike": "CRWD",
    "datadog": "DDOG",
    "shopify": "SHOP",
}


@dataclass
class ExtractedEntity:
    text: str
    label: str
    ticker: str | None
    confidence: float
    source: str


def _try_spacy():
    try:
        import spacy
        return spacy
    except ImportError:
        logger.warning("spaCy not installed — pip install spacy && python -m spacy download en_core_web_lg")
        return None


def _try_gliner():
    try:
        from gliner import GLiNER
        return GLiNER
    except ImportError:
        logger.debug("GLiNER not installed — pip install gliner")
        return None


def extract_cashtags(text: str) -> list[str]:
    """Extract cashtag tickers ($AAPL etc.) from text."""
    return _CASHTAG_RE.findall(text)


def company_to_ticker(company_name: str) -> str | None:
    """Map company name to ticker via local cache."""
    key = company_name.lower().strip()
    # Direct match
    if key in _COMPANY_TO_TICKER:
        return _COMPANY_TO_TICKER[key]
    # Partial match
    for alias, ticker in _COMPANY_TO_TICKER.items():
        if alias in key or key in alias:
            return ticker
    return None


def add_alias(company_name: str, ticker: str) -> None:
    """Add a company→ticker mapping to the local cache."""
    _COMPANY_TO_TICKER[company_name.lower().strip()] = ticker


_spacy_nlp: Any = None


def _load_spacy_model():
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp

    spacy = _try_spacy()
    if spacy is None:
        return None

    try:
        _spacy_nlp = spacy.load("en_core_web_lg")
        return _spacy_nlp
    except OSError:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
            logger.warning("en_core_web_lg not found, using en_core_web_sm")
            return _spacy_nlp
        except OSError:
            logger.warning("No spaCy model found. Run: python -m spacy download en_core_web_lg")
            return None


def extract_entities_spacy(text: str, min_confidence: float = 0.7) -> list[ExtractedEntity]:
    """Extract ORG/PERSON/GPE entities using spaCy.

    Args:
        text: News headline or article text.
        min_confidence: Minimum entity confidence (spaCy doesn't provide per-entity scores;
                        this filters by label type relevance).

    Returns:
        List of ExtractedEntity with ticker mapping if available.
    """
    nlp = _load_spacy_model()
    entities: list[ExtractedEntity] = []

    # Always extract cashtags first
    for ticker in extract_cashtags(text):
        entities.append(ExtractedEntity(
            text=f"${ticker}",
            label="CASHTAG",
            ticker=ticker,
            confidence=1.0,
            source="cashtag_regex",
        ))

    if nlp is None:
        return entities

    try:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ not in ("ORG", "PERSON", "GPE", "PRODUCT"):
                continue

            ticker = company_to_ticker(ent.text)
            entities.append(ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                ticker=ticker,
                confidence=0.8,
                source="spacy",
            ))
    except Exception as exc:
        logger.debug("spaCy NER failed: %s", exc)

    return entities


_gliner_model: Any = None


def _load_gliner_model():
    global _gliner_model
    if _gliner_model is not None:
        return _gliner_model

    GLiNER = _try_gliner()
    if GLiNER is None:
        return None

    try:
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        return _gliner_model
    except Exception as exc:
        logger.debug("GLiNER model load failed: %s", exc)
        return None


def extract_entities_gliner(
    text: str,
    labels: list[str] | None = None,
    threshold: float = 0.4,
) -> list[ExtractedEntity]:
    """Zero-shot NER using GLiNER for low-confidence spaCy entities.

    Args:
        text: Text to extract from.
        labels: Entity types to detect (default: financial company types).
        threshold: Minimum GLiNER score threshold.

    Returns:
        List of ExtractedEntity. Empty if GLiNER unavailable.
    """
    model = _load_gliner_model()
    if model is None:
        return []

    if labels is None:
        labels = ["company", "organization", "financial institution", "stock ticker"]

    try:
        predictions = model.predict_entities(text, labels, threshold=threshold)
        results = []
        for pred in predictions:
            ticker = company_to_ticker(pred["text"])
            results.append(ExtractedEntity(
                text=pred["text"],
                label=pred["label"],
                ticker=ticker,
                confidence=float(pred.get("score", 0.5)),
                source="gliner",
            ))
        return results
    except Exception as exc:
        logger.debug("GLiNER extraction failed: %s", exc)
        return []


def extract_entities(
    text: str,
    use_gliner_fallback: bool = False,
    gliner_threshold: float = 0.4,
) -> list[ExtractedEntity]:
    """Full NER pipeline: cashtag regex → spaCy → optional GLiNER fallback.

    Args:
        text: News headline or article text.
        use_gliner_fallback: Run GLiNER on entities without ticker mapping.
        gliner_threshold: GLiNER minimum score.

    Returns:
        Deduplicated list of extracted entities with ticker mappings.
    """
    entities = extract_entities_spacy(text)

    if use_gliner_fallback:
        # Only run GLiNER for entities without a ticker match
        unmapped = [e for e in entities if e.ticker is None and e.source == "spacy"]
        if unmapped or not entities:
            gliner_entities = extract_entities_gliner(text, threshold=gliner_threshold)
            # Avoid duplicating already-found entities
            existing_texts = {e.text.lower() for e in entities}
            for ge in gliner_entities:
                if ge.text.lower() not in existing_texts:
                    entities.append(ge)

    # Deduplicate by ticker
    seen_tickers: set[str] = set()
    unique: list[ExtractedEntity] = []
    for e in entities:
        key = e.ticker or e.text
        if key not in seen_tickers:
            seen_tickers.add(key)
            unique.append(e)

    return unique


def tickers_from_text(text: str) -> list[str]:
    """Convenience: return just the ticker symbols mentioned in text."""
    entities = extract_entities(text)
    return [e.ticker for e in entities if e.ticker is not None]


__all__ = [
    "ExtractedEntity",
    "extract_cashtags",
    "company_to_ticker",
    "add_alias",
    "extract_entities_spacy",
    "extract_entities_gliner",
    "extract_entities",
    "tickers_from_text",
]
