"""Filing → Trading-Signal-Mapping.

Pipeline
--------
1. Receive klassifiziertes Filing (von ``material_event_classifier``).
2. Map to (symbol, direction, size, conviction).
3. Apply Conviction-Discount via:
   - Confidence aus material_score.
   - Recency (Filing-Age in Stunden).
   - Pre-market vs After-Hours-Filing-Discount.
4. Return Trade-Recommendation.

Achtung
-------
Diese Module liefert **Signal-Vorschläge**, keine Orders. Production-Use
sollte zusätzliche Gates haben: Risk-Limits, Universe-Filter, Slippage.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from erweiterung.live_pipeline.material_event_classifier import EventClassification


@dataclass
class FilingSignal:
    symbol: str
    direction: int  # +1 buy / -1 sell
    conviction: float  # 0..1
    target_weight_pct: float  # 0..max_per_filing
    rationale: str
    accession: str


def map_classification_to_signal(
    classification: EventClassification,
    symbol: str,
    filing_age_hours: float = 0.0,
    market_session: str = "intraday",
    max_per_filing_weight: float = 0.05,
) -> FilingSignal | None:
    """Map classified filing to trading-signal.

    Args:
        classification: from material_event_classifier.
        symbol: ticker.
        filing_age_hours: hours since filing was published.
        market_session: 'pre_market' | 'intraday' | 'after_hours'.
        max_per_filing_weight: cap target-weight.

    Returns:
        FilingSignal or None if not actionable.
    """
    if classification.material_score < 0.4 or classification.expected_direction == 0:
        return None

    # Conviction = material-score × recency-decay × session-discount
    age_decay = max(0.0, 1.0 - filing_age_hours / 48.0)  # 48h = 0 conviction
    session_discount = {
        "pre_market": 0.7,  # has potential for next-open gap; lower conviction
        "intraday": 1.0,
        "after_hours": 0.6,  # similar; price discovery happens next morning
    }.get(market_session, 1.0)

    conviction = classification.material_score * age_decay * session_discount
    if conviction < 0.2:
        return None

    target_weight = (
        max_per_filing_weight * conviction * classification.expected_direction
    )

    return FilingSignal(
        symbol=symbol,
        direction=int(classification.expected_direction),
        conviction=float(conviction),
        target_weight_pct=float(target_weight),
        rationale=classification.explanation,
        accession=classification.accession,
    )


def aggregate_filings_to_portfolio(
    signals: list[FilingSignal], existing_weights: pd.Series | None = None
) -> pd.Series:
    """Aggregiere mehrere Filing-Signals zu Portfolio-Gewichten.

    Wenn mehrere Signals für denselben symbol vorliegen: nimm conviction-gewichteten
    Mittelwert der target_weights.

    Args:
        signals: Liste FilingSignal.
        existing_weights: vorhandene Portfolio-Gewichte (Tilts werden additiv).

    Returns:
        Series of new portfolio-weights per symbol.
    """
    if not signals:
        return (
            existing_weights.copy()
            if existing_weights is not None
            else pd.Series(dtype=float)
        )

    # Aggregate per symbol via conviction-weighted mean
    by_symbol: dict[str, list[FilingSignal]] = {}
    for s in signals:
        by_symbol.setdefault(s.symbol, []).append(s)

    out = (
        existing_weights.copy()
        if existing_weights is not None
        else pd.Series(dtype=float)
    )
    for sym, sigs in by_symbol.items():
        weights = [s.conviction for s in sigs]
        total_w = sum(weights)
        if total_w <= 0:
            continue
        agg_target = (
            sum(s.target_weight_pct * w for s, w in zip(sigs, weights)) / total_w
        )
        out[sym] = float(out.get(sym, 0.0) + agg_target)
    return out


def signals_to_dataframe(signals: list[FilingSignal]) -> pd.DataFrame:
    if not signals:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "symbol": s.symbol,
                "direction": s.direction,
                "conviction": s.conviction,
                "target_weight_pct": s.target_weight_pct,
                "rationale": s.rationale,
                "accession": s.accession,
            }
            for s in signals
        ]
    )


__all__ = [
    "FilingSignal",
    "map_classification_to_signal",
    "aggregate_filings_to_portfolio",
    "signals_to_dataframe",
]
