#!/usr/bin/env python
"""Multi-Faktor-Vol-Targeting-Backtest auf 19-Jahres-Equity.

Pipeline
--------
1. Lade Faktor-Returns (Mom-12/1, ResMom, LowVol) aus Long-History-Backtest.
2. Vol-Targete jeden Faktor einzeln auf target_vol_annual=0.12 (Default).
3. Kombiniere via Equal-Weight / Inverse-Vol / HRP.
4. Vergleiche mit Single-Faktor-Vol-Target und Pure Long-Only.
5. Statistik via Calmar-Bootstrap (robuster als Sharpe-Bootstrap).
6. Walk-Forward OOS-Test.
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
from erweiterung.strategies.multi_factor_vol_target import (  # noqa: E402
    MultiFactorVolTargetConfig,
    combine_factors,
)


def _convert(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return o


def _walk(o):
    if isinstance(o, dict):
        return {str(k): _walk(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_walk(v) for v in o]
    return _convert(o)


def main():
    eq_csv = Path("output/erweiterung_long_history_equity.csv")
    if not eq_csv.exists():
        print(f"ERROR: {eq_csv} not found.")
        return 1

    df = pd.read_csv(eq_csv)
    if df.columns[0] in ("date", "Date", "Unnamed: 0"):
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()

    factor_returns = {
        "mom_12_1": df["momentum_12_1_LongOnly"].pct_change().dropna(),
        "res_mom": df["residual_momentum_LongOnly"].pct_change().dropna(),
        "low_vol": df["low_vol_LongOnly"].pct_change().dropna(),
    }
    bench_ret = df["benchmark_equal_weight"].pct_change().dropna()
    pure_mom = factor_returns["mom_12_1"]

    # === Verschiedene Kombinations-Schemata ===
    print("\n" + "=" * 100)
    print("MULTI-FACTOR VOL-TARGETING (target_vol=0.12)")
    print("=" * 100)
    print(
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)

    candidates: dict[str, pd.Series] = {}
    metrics_dump: dict[str, dict] = {}

    # Reference
    for name, ret in [
        ("Pure Equal-Weight", bench_ret),
        ("Pure Mom-12/1 LO", pure_mom),
        ("Pure ResMom LO", factor_returns["res_mom"]),
        ("Pure LowVol LO", factor_returns["low_vol"]),
    ]:
        m = all_metrics(ret.dropna())
        metrics_dump[name] = m
        candidates[name] = ret
        print(
            f"  {name:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    # Multi-Faktor-Vol-Target-Varianten
    for combiner in ("equal_weight", "inverse_vol", "hrp"):
        cfg = MultiFactorVolTargetConfig(target_vol_annual=0.12, combiner=combiner)
        out = combine_factors(factor_returns, cfg)
        ret = out["combined"].dropna()
        m = all_metrics(ret)
        label = f"MultiFac-VolTarget-{combiner}"
        candidates[label] = ret
        metrics_dump[label] = m
        print(
            f"  {label:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    # Single-Factor-Vol-Target zum Vergleich (existing)
    single_vol_csv = Path("output/erweiterung_vol_target_oos_equity.csv")
    if single_vol_csv.exists():
        sv = pd.read_csv(single_vol_csv, index_col=0)
        sv.index = pd.to_datetime(sv.index, utc=True)
        # Wir nehmen den In-Sample-Vol-Target (Re-Compute) vom Pure-Mom:
        from erweiterung.strategies.volatility_targeting import (
            VolTargetConfig,
            apply_vol_targeting,
        )

        sf = apply_vol_targeting(pure_mom, VolTargetConfig(target_vol_annual=0.12))
        single_vt_mom = sf["scaled_return"].dropna()
        candidates["Single-VolTarget Mom"] = single_vt_mom
        m = all_metrics(single_vt_mom)
        metrics_dump["Single-VolTarget Mom"] = m
        print(
            f"  {'Single-VolTarget Mom':<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    # === Calmar-Bootstrap-Tests vs Pure-Mom ===
    print("\n" + "=" * 100)
    print("CALMAR-BOOTSTRAP vs Pure Mom-12/1 LO (stationary bootstrap, avg_block=20)")
    print("=" * 100)
    print(
        f"{'Challenger':<32} {'obs_diff':>9} {'mean_diff':>10} {'95% CI':>22} {'p(>0)':>7}"
    )
    print("-" * 100)
    baseline = pure_mom
    for name, ret in candidates.items():
        if name == "Pure Mom-12/1 LO":
            continue
        out = calmar_diff_bootstrap(
            ret, baseline, n_bootstrap=2000, avg_block_size=20, seed=42
        )
        if "error" in out:
            continue
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci_str = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(
            f"  {name:<30} "
            f"{out['observed_diff']:>+8.3f} "
            f"{out['mean_diff']:>+9.3f} "
            f"{ci_str:>22} "
            f"{p_gt:>6.3f}"
        )

    # === Calmar-Bootstrap vs Equal-Weight (passiv) ===
    print("\n" + "=" * 100)
    print("CALMAR-BOOTSTRAP vs Pure Equal-Weight")
    print("=" * 100)
    print(
        f"{'Challenger':<32} {'obs_diff':>9} {'mean_diff':>10} {'95% CI':>22} {'p(>0)':>7}"
    )
    print("-" * 100)
    for name, ret in candidates.items():
        if name == "Pure Equal-Weight":
            continue
        out = calmar_diff_bootstrap(
            ret, bench_ret, n_bootstrap=2000, avg_block_size=20, seed=42
        )
        if "error" in out:
            continue
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        ci_str = f"[{out['ci_low_2.5']:+.2f}, {out['ci_high_97.5']:+.2f}]"
        print(
            f"  {name:<30} "
            f"{out['observed_diff']:>+8.3f} "
            f"{out['mean_diff']:>+9.3f} "
            f"{ci_str:>22} "
            f"{p_gt:>6.3f}"
        )

    # Save equity-curves
    eq_df = pd.DataFrame(
        {k: (1 + v.fillna(0)).cumprod() for k, v in candidates.items()}
    )
    eq_df.to_csv("output/erweiterung_multi_factor_vol_target_equity.csv")
    Path("output/erweiterung_multi_factor_vol_target_summary.json").write_text(
        json.dumps(
            _walk(
                {
                    "metrics": {
                        name: {
                            k: _convert(v)
                            for k, v in m.items()
                            if not isinstance(v, (pd.Series, pd.DataFrame))
                        }
                        for name, m in metrics_dump.items()
                    },
                }
            ),
            indent=2,
            default=str,
        )
    )
    print("\nSaved -> output/erweiterung_multi_factor_vol_target_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
