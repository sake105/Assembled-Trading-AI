#!/usr/bin/env python
"""End-to-End Research-Pipeline-Demo auf lokalen Daten.

Kombiniert alle Stufen: Daten → Faktoren → Strategien → Validation → Stress-Test → Report.

Output:
- output/erweiterung_research/metrics.json
- output/erweiterung_research/equity_curves.csv
- output/erweiterung_research/report.html
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.factors.fama_french import momentum_12_1  # noqa: E402
from erweiterung.factors.low_vol import low_vol_signal  # noqa: E402
from erweiterung.pipelines.research_pipeline import (  # noqa: E402
    ResearchPipelineConfig,
    run_research_pipeline,
)
from erweiterung.strategies.templates import StrategyConfig  # noqa: E402
from erweiterung.strategies.templates import (  # noqa: E402
    low_vol_strategy as run_low_vol,
)
from erweiterung.strategies.templates import (
    trend_following as run_trend,  # noqa: E402
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def signal_builder(panel: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
    """Add momentum + low-vol signals to panel."""
    out = panel.copy()
    mom = momentum_12_1(out[["date", "symbol", "close"]])
    out = out.set_index(["date", "symbol"])
    out["momentum_12_1"] = mom.reindex(out.index)
    out = out.reset_index()
    lv = low_vol_signal(out[["date", "symbol", "return"]], window=60)
    out = out.set_index(["date", "symbol"])
    out["rolling_vol_60"] = -lv.reindex(out.index)  # negate so high=low-vol
    out = out.reset_index()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2026-04-01")
    p.add_argument("--parquet", default="data/sample/watchlist_2007_2026.parquet")
    p.add_argument("--out-dir", default="output/erweiterung_research")
    args = p.parse_args()

    logger.info("Loading panel: %s ...", args.parquet)
    df = pd.read_parquet(args.parquet)
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[
        (df["date"] >= pd.Timestamp(args.start, tz="UTC"))
        & (df["date"] <= pd.Timestamp(args.end, tz="UTC"))
    ]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["return"] = df.groupby("symbol")["close"].pct_change()
    logger.info("Panel: %d rows, %d symbols", len(df), df["symbol"].nunique())

    market = df.groupby("date")["return"].mean().sort_index()

    cfg = ResearchPipelineConfig(
        output_dir=args.out_dir, transaction_cost_bps=5.0, n_bootstrap=1000, seed=42
    )

    strategies = {
        "trend_following_LongOnly": lambda p: run_trend(
            p, StrategyConfig(quantile=0.2, long_only=True, transaction_cost_bps=5.0)
        ),
        "low_vol_LongOnly": lambda p: run_low_vol(
            p, StrategyConfig(quantile=0.2, long_only=True, transaction_cost_bps=5.0)
        ),
        "trend_following_LongShort": lambda p: run_trend(
            p, StrategyConfig(quantile=0.2, long_only=False, transaction_cost_bps=5.0)
        ),
    }

    res = run_research_pipeline(
        panel=df,
        market_returns=market,
        signal_builder=signal_builder,
        strategy_runners=strategies,
        config=cfg,
    )

    print("\n" + "=" * 80)
    print("RESEARCH PIPELINE SUMMARY")
    print("=" * 80)
    for name, m in res.metrics.items():
        if "sharpe" in m:
            print(
                f"  {name:<32} Sharpe={m.get('sharpe', 0):+.3f}  "
                f"AnnRet={m.get('annualized_return', 0):+.2%}  "
                f"MDD={m.get('max_drawdown', 0):+.2%}  "
                f"Calmar={m.get('calmar', 0):+.3f}"
            )
    print()
    print(
        f"Reality-Check best={res.reality_check.get('best_strategy')}, p={res.reality_check.get('p_value'):.3f}"
    )
    print(
        f"Hansen-SPA   best={res.spa.get('best_strategy')}, p={res.spa.get('p_value'):.3f}"
    )
    if "stress_score" in res.stress:
        ss = res.stress["stress_score"]
        print(
            f"Stress: worst-DD={ss.get('worst_drawdown', 0):+.2%}  worst-crisis={ss.get('worst_crisis', 'n/a')}"
        )
    if "monte_carlo" in res.stress:
        mc = res.stress["monte_carlo"]
        print(
            f"Monte-Carlo (h=126): expected={mc.get('terminal_mean', 0):+.2%}  "
            f"VaR95-loss={mc.get('terminal_var_loss', 0):+.2%}  "
            f"prob-loss={mc.get('prob_of_loss', 0):.2%}"
        )
    print(f"\nOutputs: {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
