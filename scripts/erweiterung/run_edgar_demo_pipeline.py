#!/usr/bin/env python
"""EDGAR-Live-Pipeline Smoke-Demo mit synthetischen 8-K-Filings.

Pipeline-Steps
--------------
1. Parse 8-K-Text → Item-Codes (regex)
2. Classify → (category, direction, material_score)
3. Map → FilingSignal (conviction-decay)
4. Aggregate → Portfolio-Weights

Echte EDGAR-Live-Daten würden via SEC-EDGAR-API geholt werden — siehe
``src/erweiterung/live_pipeline/`` für die Module. Dieser Demo zeigt
nur die Pipeline-Mechanik mit handgeschriebenen Filing-Beispielen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.live_pipeline.filing_signal_mapper import (  # noqa: E402
    aggregate_filings_to_portfolio,
    map_classification_to_signal,
    signals_to_dataframe,
)
from erweiterung.live_pipeline.material_event_classifier import (  # noqa: E402
    classify_filing_text,
    extract_items,
    filter_high_material_events,
)


# Stilisierte 8-K-Filings (realitätsnah, aber konstruiert)
SAMPLE_FILINGS = [
    {
        "accession": "0001-23-001",
        "symbol": "ACME",
        "filing_age_hours": 2,
        "market_session": "intraday",
        "text": """
        Item 2.02. Results of Operations and Financial Condition.
        On May 1, 2026, ACME Corp reported quarterly revenue of $5.2B, up 23%
        year-over-year, exceeding consensus estimates of $4.8B. Adjusted EPS
        came in at $1.42 vs estimate $1.18. Guidance raised for fiscal 2026.
        Item 9.01. Financial Statements and Exhibits.
        """,
    },
    {
        "accession": "0002-23-002",
        "symbol": "BADCO",
        "filing_age_hours": 5,
        "market_session": "after_hours",
        "text": """
        Item 3.01. Notice of Delisting or Failure to Satisfy a Continued
        Listing Rule. The Company received notice from NASDAQ on May 5,
        2026 that it no longer complies with the minimum bid price requirement.
        Item 8.01. Other Events.
        """,
    },
    {
        "accession": "0003-23-003",
        "symbol": "MERGECO",
        "filing_age_hours": 12,
        "market_session": "pre_market",
        "text": """
        Item 1.01. Entry into a Material Definitive Agreement. On May 2,
        2026, the Company entered into a definitive Agreement and Plan of Merger
        with TargetCo, pursuant to which the Company will acquire all outstanding
        shares of TargetCo for $50 per share in cash, totaling approximately
        $8.2 billion.
        Item 5.01. Changes in Control of Registrant.
        """,
    },
    {
        "accession": "0004-23-004",
        "symbol": "IMPAIR",
        "filing_age_hours": 1,
        "market_session": "intraday",
        "text": """
        Item 2.06. Material Impairments. On May 6, 2026, the Company concluded
        that a non-cash goodwill impairment charge of approximately $2.1B is
        required to be recorded in Q2 2026, related to the legacy business unit.
        """,
    },
    {
        "accession": "0005-23-005",
        "symbol": "ROUTINE",
        "filing_age_hours": 8,
        "market_session": "intraday",
        "text": """
        Item 9.01. Financial Statements and Exhibits.
        Item 7.01. Regulation FD Disclosure. Routine investor relations update.
        """,
    },
]


def main():
    print("=" * 100)
    print("EDGAR LIVE-PIPELINE SMOKE-DEMO")
    print("=" * 100)
    print(f"Processing {len(SAMPLE_FILINGS)} sample 8-K filings ...")

    classifications = []
    signals = []
    for f in SAMPLE_FILINGS:
        # Step 1+2: Classify
        items = extract_items(f["text"])
        cls = classify_filing_text(f["text"], accession=f["accession"])
        classifications.append(cls)
        print(f"\n  {f['symbol']:<10} [{f['accession']}]")
        print(f"    Items: {items}")
        print(f"    Categories: {cls.categories}")
        print(
            f"    Direction: {cls.expected_direction:+d}, "
            f"Material-Score: {cls.material_score:.2f}"
        )
        print(f"    Explanation: {cls.explanation}")

        # Step 3: Map to Signal
        sig = map_classification_to_signal(
            cls,
            symbol=f["symbol"],
            filing_age_hours=f["filing_age_hours"],
            market_session=f["market_session"],
        )
        if sig is None:
            print("    -> NO ACTIONABLE SIGNAL (below threshold)")
        else:
            print(
                f"    -> Signal: direction={sig.direction:+d}, "
                f"conviction={sig.conviction:.2f}, "
                f"target_weight={sig.target_weight_pct:+.2%}"
            )
            signals.append(sig)

    # Step 4: Aggregate to Portfolio
    print("\n" + "=" * 100)
    print("PORTFOLIO-AGGREGATION")
    print("=" * 100)
    print(f"{len(signals)} actionable signals -> portfolio weights:")
    weights = aggregate_filings_to_portfolio(signals)
    for sym, w in weights.sort_values(ascending=False).items():
        print(f"  {sym:<10} weight = {w:+.4f}")

    # Filter high-material events
    filings_df = pd.DataFrame(
        [{"accession": f["accession"], "symbol": f["symbol"]} for f in SAMPLE_FILINGS]
    )
    high_material = filter_high_material_events(
        filings_df, classifications, score_threshold=0.7
    )
    print(f"\nHigh-material events (score >= 0.7): {len(high_material)}")
    if not high_material.empty:
        print(high_material.to_string(index=False))

    # Save signals
    if signals:
        df = signals_to_dataframe(signals)
        df.to_csv("output/erweiterung_edgar_demo_signals.csv", index=False)
        print("\nSaved -> output/erweiterung_edgar_demo_signals.csv")

    print("\n[OK] EDGAR Pipeline-Mechanik validiert end-to-end.")
    print("Echter Live-Run benötigt: SEC-EDGAR-API-Polling + News-Stream-Integration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
