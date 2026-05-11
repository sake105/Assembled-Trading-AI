#!/usr/bin/env python
"""Tail-Risk-Hedging-Backtest: VIX-Spike-Trigger auf Master-Allocator.

Test
----
1. Lade Master-Pipeline-Equity (oder Long-History-Master-Returns)
2. Lade VIX aus output/macro.parquet
3. Apply Tail-Hedge (z-score und absolute-Trigger getrennt)
4. Vergleich: ungehedget vs gehedget
5. Calmar-Bootstrap

Hypothese: MDD reduziert, AnnRet leicht gemindert. Sharpe könnte
unverändert bleiben oder verbessern.
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
from erweiterung.strategies.tail_risk_hedge import (  # noqa: E402
    TailHedgeConfig,
    apply_tail_hedge,
)


def main():
    # Lade Master-Returns
    p_master = Path("output/erweiterung_master_allocation_equity.csv")
    if not p_master.exists():
        print(f"ERROR: {p_master} not found.")
        return 1
    df = pd.read_csv(p_master)
    first = df.columns[0]
    df = df.rename(columns={first: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    master_ret = df["Master_70_30"].pct_change().dropna()
    print(
        f"Master returns: {len(master_ret)} days, "
        f"{master_ret.index.min()} -> {master_ret.index.max()}"
    )

    # Lade VIX
    macro = pd.read_parquet("output/macro.parquet")
    macro["timestamp"] = pd.to_datetime(macro["timestamp"], utc=True)
    macro = macro.set_index("timestamp").sort_index()
    vix = macro["vix_close"].dropna()
    print(
        f"VIX: {len(vix)} days, "
        f"{vix.index.min()} -> {vix.index.max()}, "
        f"mean={vix.mean():.1f}, max={vix.max():.1f}"
    )

    # Configs to test
    configs = {
        "VIX-Z-1.5_StressExp50": TailHedgeConfig(
            use_zscore=True,
            spike_zscore_threshold=1.5,
            exposure_during_stress=0.50,
            re_engage_zscore=0.5,
        ),
        "VIX-Z-2.0_StressExp50": TailHedgeConfig(
            use_zscore=True,
            spike_zscore_threshold=2.0,
            exposure_during_stress=0.50,
            re_engage_zscore=0.5,
        ),
        "VIX-Abs-30_StressExp50": TailHedgeConfig(
            use_zscore=False,
            vix_absolute_threshold=30.0,
            re_engage_absolute=22.0,
            exposure_during_stress=0.50,
        ),
        "VIX-Abs-30_StressExp0": TailHedgeConfig(
            use_zscore=False,
            vix_absolute_threshold=30.0,
            re_engage_absolute=22.0,
            exposure_during_stress=0.0,
        ),
    }

    print("\n" + "=" * 100)
    print("TAIL-HEDGE BACKTEST RESULTS")
    print("=" * 100)
    print(
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8} {'StressDays':>11}"
    )
    print("-" * 100)
    m_master = all_metrics(master_ret.dropna())
    print(
        f"  {'Pure Master_70_30':<30} "
        f"{m_master.get('annualized_return', 0):>+8.2%} "
        f"{m_master.get('sharpe', 0):>+7.3f} "
        f"{m_master.get('sortino', 0):>+8.3f} "
        f"{m_master.get('calmar', 0):>+7.3f} "
        f"{m_master.get('max_drawdown', 0):>+7.2%} "
        f"{'-':>11}"
    )
    metrics_dump = {"Pure Master_70_30": m_master}

    hedged_curves = {}
    for label, cfg in configs.items():
        out = apply_tail_hedge(master_ret, vix, cfg)
        if out.empty:
            continue
        ret = out["hedged_return"]
        m = all_metrics(ret.dropna())
        stress_share = (out["trigger"].shift(1) == "stress").mean()
        print(
            f"  {label:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%} "
            f"{stress_share:>10.1%}"
        )
        metrics_dump[label] = m
        hedged_curves[label] = ret

    # Calmar-Bootstrap vs Pure Master
    print("\n" + "=" * 100)
    print("CALMAR-BOOTSTRAP vs Pure Master_70_30")
    print("=" * 100)
    print(
        f"{'Hedge Variant':<32} {'obs_diff':>9} {'mean_diff':>10} {'95% CI':>22} {'p(>0)':>7}"
    )
    print("-" * 100)
    for label, ret in hedged_curves.items():
        out = calmar_diff_bootstrap(
            ret.dropna(),
            master_ret.dropna(),
            n_bootstrap=2000,
            avg_block_size=20,
            seed=42,
        )
        if "error" in out:
            continue
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(
            f"  {label:<30} "
            f"{out['observed_diff']:>+8.3f} "
            f"{out['mean_diff']:>+9.3f} "
            f"{ci:>22} "
            f"{p_gt:>6.3f}"
        )

    # Save canonical (z-1.5 / exp 0.50)
    canonical_label = "VIX-Z-1.5_StressExp50"
    if canonical_label in hedged_curves:
        canonical = hedged_curves[canonical_label]
        out_df = pd.DataFrame(
            {
                "raw_return": master_ret,
                "hedged_return": canonical,
                "hedged_equity": (1 + canonical.fillna(0)).cumprod(),
            }
        )
        out_df.to_csv("output/erweiterung_tail_hedge_equity.csv")
        Path("output/erweiterung_tail_hedge_summary.json").write_text(
            json.dumps(
                {
                    "canonical": canonical_label,
                    "metrics": {
                        name: {
                            k: (
                                float(v)
                                if isinstance(v, (int, float, np.floating, np.integer))
                                else v
                            )
                            for k, v in m.items()
                            if not isinstance(v, (pd.Series, pd.DataFrame))
                        }
                        for name, m in metrics_dump.items()
                    },
                },
                indent=2,
                default=str,
            )
        )
    print("\nSaved -> output/erweiterung_tail_hedge_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
