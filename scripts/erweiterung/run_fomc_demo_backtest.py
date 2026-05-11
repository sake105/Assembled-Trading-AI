#!/usr/bin/env python
"""FOMC-Macro-Signal Demo-Backtest auf handgeschriebenen historischen Statements.

Daten-Caveat
------------
Echte FOMC-Statement-Texte sind nicht im Repo. Wir nutzen hier
**stilisierte / handgeschriebene** Statements für 8 wichtige historische
FOMC-Meetings (2008 GFC, 2020 COVID, 2022 Hike-Cycle, 2024 Pause).

Das ist eine **Pipeline-Demo**, kein finaler Backtest. Mit echten
Multi-Jahr-FOMC-Texten (vom FED-Archiv) wäre der selbe Code statistisch
aussagekräftig.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.backtest.calmar_bootstrap import calmar_diff_bootstrap  # noqa: E402
from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.strategies.fomc_macro_signal import (  # noqa: E402
    FOMCMacroConfig,
    apply_fomc_allocation_override,
    build_fomc_signal_series,
)


# Stilisierte FOMC-Statements (handgeschrieben, basierend auf öffentlich bekannten Themen)
HISTORICAL_STATEMENTS = [
    # 2008-12 — emergency rate cut (deep dove)
    (
        "2008-12-16",
        "The Federal Open Market Committee decided to lower its target for the federal "
        "funds rate to a range of 0 to 1/4 percent. The Committee will employ all "
        "available tools to promote the resumption of sustainable economic growth. "
        "Aggressive accommodation will support employment and price stability. "
        "Easing financial conditions, the Committee will continue stimulus measures.",
    ),
    # 2015-12 — first hike post-GFC (hawk surprise)
    (
        "2015-12-16",
        "The Committee voted to raise the target range for the federal funds rate. "
        "Economic activity has been expanding at a moderate pace. Tightening labor "
        "market conditions. The Committee expects further gradual increases. "
        "Removing accommodation appropriate to support continued progress.",
    ),
    # 2018-12 — controversial hike (mild hawk)
    (
        "2018-12-19",
        "The Committee raised the target range. Labor market continues to strengthen. "
        "Inflation near 2 percent. Gradual increases in the target range will be appropriate. "
        "Restrictive policy is needed to balance employment and price stability.",
    ),
    # 2020-03 — emergency COVID cut (deep dove)
    (
        "2020-03-15",
        "The Committee voted unanimously to lower the target range to 0 to 1/4 percent. "
        "Easing financial conditions, supporting accommodation, employing all available tools. "
        "Aggressive stimulus, accommodative policy will continue. Ample liquidity. "
        "Supporting the flow of credit to households and businesses.",
    ),
    # 2022-03 — Inflation-Hike begin (hawk surprise)
    (
        "2022-03-16",
        "The Committee decided to raise the target range. Inflation remains elevated, "
        "reflecting supply demand imbalances. Tightening policy is appropriate to bring "
        "inflation back to 2 percent. Robust labor market. Further increases will be "
        "appropriate. Restrictive policy stance will continue.",
    ),
    # 2022-09 — Aggressive hike (deeper hawk)
    (
        "2022-09-21",
        "Committee raised target range by 75 basis points. Highly elevated inflation. "
        "Tightening continues. Strong labor markets. Restrictive policy stance is "
        "needed to bring inflation down. Hiking will continue. Inflation pressures broad-based.",
    ),
    # 2024-01 — Pause / signal end of hikes (neutral / mild dove)
    (
        "2024-01-31",
        "Committee maintains target range. Economic activity expanded solidly. "
        "Inflation has eased over the past year. Maintaining current policy stance. "
        "Risks to achieving employment and inflation goals are moving better balanced.",
    ),
    # 2024-09 — first cut (dove)
    (
        "2024-09-18",
        "Committee voted to lower the target range. Inflation continues to slow. "
        "Easing policy is appropriate to support employment. Lower rates will help. "
        "Confidence that inflation is moving sustainably toward 2 percent.",
    ),
]


def main():
    # Lade Master-Returns
    p = Path("output/erweiterung_master_allocation_equity.csv")
    if not p.exists():
        print(f"ERROR: {p} not found.")
        return 1
    df = pd.read_csv(p)
    first = df.columns[0]
    df = df.rename(columns={first: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    master_ret = df["Master_70_30"].pct_change().dropna()

    statements = [s for _, s in HISTORICAL_STATEMENTS]
    dates = [pd.Timestamp(d, tz="UTC") for d, _ in HISTORICAL_STATEMENTS]
    # Filter: nur statements im Master-Range
    in_range = [
        (d, s)
        for d, s in zip(dates, statements)
        if master_ret.index.min() <= d <= master_ret.index.max()
    ]
    print(
        f"Statements: {len(HISTORICAL_STATEMENTS)} total, "
        f"{len(in_range)} im Master-Range ({master_ret.index.min().date()} -> {master_ret.index.max().date()})"
    )
    if not in_range:
        print("WARNING: Keine FOMC-Statements im Master-Range. Backtest übersprungen.")
        return 0

    in_dates = [d for d, _ in in_range]
    in_statements = [s for _, s in in_range]

    # Build signal
    cfg = FOMCMacroConfig(
        hawkish_delta_threshold=0.10,
        dovish_delta_threshold=-0.10,
        exposure_hawkish=0.50,
        exposure_dovish=1.20,
        decay_days=14,
    )
    fomc_signal = build_fomc_signal_series(
        in_statements, in_dates, master_ret.index, cfg
    )
    print(
        f"FOMC-Signal: {len(fomc_signal)} days, "
        f"non-neutral: {(fomc_signal != 1.0).sum()} days"
    )

    # Apply
    hedged = apply_fomc_allocation_override(master_ret, fomc_signal)

    # Compare
    m_raw = all_metrics(master_ret.dropna())
    m_h = all_metrics(hedged.dropna())

    print("\n" + "=" * 100)
    print("FOMC-MACRO-SIGNAL DEMO-BACKTEST")
    print("=" * 100)
    print(
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)
    for name, m in [("Pure Master_70_30", m_raw), ("FOMC-Override Master", m_h)]:
        print(
            f"  {name:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    # Calmar-Bootstrap
    out = calmar_diff_bootstrap(
        hedged.dropna(),
        master_ret.dropna(),
        n_bootstrap=2000,
        avg_block_size=20,
        seed=42,
    )
    p_gt = 1.0 - out["p_value_one_sided_greater"] if "error" not in out else None
    if p_gt is not None:
        ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(
            f"\nCalmar-Bootstrap FOMC vs Pure: obs_diff={out['observed_diff']:+.3f}, "
            f"95% CI {ci}, p(>0)={p_gt:.3f}"
        )

    # Save
    pd.DataFrame(
        {
            "raw_return": master_ret,
            "fomc_signal": fomc_signal,
            "hedged_return": hedged,
            "hedged_equity": (1 + hedged.fillna(0)).cumprod(),
        }
    ).to_csv("output/erweiterung_fomc_demo_equity.csv")
    Path("output/erweiterung_fomc_demo_summary.json").write_text(
        json.dumps(
            {
                "caveat": "DEMO — handgeschriebene Statements, kein echter FOMC-Archive-Backtest",
                "n_statements_used": len(in_range),
                "metrics_master": {
                    k: (
                        float(v)
                        if isinstance(v, (int, float, np.floating, np.integer))
                        else v
                    )
                    for k, v in m_raw.items()
                    if not isinstance(v, (pd.Series, pd.DataFrame))
                },
                "metrics_fomc_override": {
                    k: (
                        float(v)
                        if isinstance(v, (int, float, np.floating, np.integer))
                        else v
                    )
                    for k, v in m_h.items()
                    if not isinstance(v, (pd.Series, pd.DataFrame))
                },
            },
            indent=2,
            default=str,
        )
    )
    print("\nSaved -> output/erweiterung_fomc_demo_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
