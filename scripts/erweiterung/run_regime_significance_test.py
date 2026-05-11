#!/usr/bin/env python
"""Statistischer Signifikanztest: schlägt das Ensemble die einzelnen Detectors?

Hansen-SPA-Test + White's-Reality-Check der Regime-Switching-Varianten
gegen Pure-Mom-12/1-LongOnly als Benchmark.

Plus: Deflated-Sharpe-Ratio für jede Strategie und Bootstrapped-Confidence-
Intervall für Sharpe-Differenzen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.backtest.deflated_sharpe import deflated_sharpe_ratio  # noqa: E402
from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.backtest.white_reality_check import (  # noqa: E402
    hansen_spa_test,
    whites_reality_check,
)


def _load_equity(path: str, col: str = "equity") -> pd.Series | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date").sort_index()
    else:
        first = df.columns[0]
        df["date"] = pd.to_datetime(df[first], utc=True)
        df = df.set_index("date").sort_index()
    if col in df.columns:
        return df[col]
    return None


def _bootstrap_sharpe_diff(
    a: pd.Series, b: pd.Series, n_bootstrap: int = 2000, seed: int = 42
) -> dict:
    """Block-Bootstrap Sharpe-Difference + 95-%-CI."""
    rng = np.random.default_rng(seed)
    aligned = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if len(aligned) < 50:
        return {"error": "insufficient overlap"}

    block_size = 20
    n_blocks = max(1, len(aligned) // block_size)
    diffs = []
    for _ in range(n_bootstrap):
        starts = rng.integers(0, len(aligned) - block_size, n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
        idx = idx[idx < len(aligned)]
        sample = aligned.iloc[idx]
        if sample["a"].std() == 0 or sample["b"].std() == 0:
            continue
        sh_a = sample["a"].mean() / sample["a"].std() * np.sqrt(252)
        sh_b = sample["b"].mean() / sample["b"].std() * np.sqrt(252)
        diffs.append(sh_a - sh_b)
    if not diffs:
        return {"error": "no bootstrap samples"}
    diffs = np.array(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low_2.5": float(np.percentile(diffs, 2.5)),
        "ci_high_97.5": float(np.percentile(diffs, 97.5)),
        "p_value_one_sided": float((diffs <= 0).mean()),
        "n_bootstrap": int(len(diffs)),
    }


def main():
    # Lade alle Equity-Curves
    expanded = _load_equity(
        "output/erweiterung_expanded_universe_equity.csv",
        col="momentum_12_1_LongOnly",
    )
    expanded_bench = _load_equity(
        "output/erweiterung_expanded_universe_equity.csv",
        col="benchmark_equal_weight",
    )
    candidates = {
        "pure_mom_12_1": expanded,
        "pure_equal_weight": expanded_bench,
        "drawdown_only": _load_equity(
            "output/erweiterung_regime_conditional_equity.csv"
        ),
        "multi_signal": _load_equity(
            "output/erweiterung_multi_signal_regime_equity.csv"
        ),
        "macro_only": _load_equity("output/erweiterung_macro_regime_equity.csv"),
        "ensemble_weighted_thr0.50": _load_equity(
            "output/erweiterung_ensemble_regime_equity.csv"
        ),
    }
    candidates = {k: v for k, v in candidates.items() if v is not None}

    # Convert to returns
    returns = {}
    for k, ser in candidates.items():
        r = ser.pct_change().dropna()
        # Drop zero-return prefix (initial $1.0 day)
        first_nonzero = r.ne(0).idxmax() if r.ne(0).any() else r.index[0]
        returns[k] = r.loc[first_nonzero:]
    common = pd.concat(returns, axis=1).dropna()
    print(
        f"Common period: {common.index.min()} -> {common.index.max()} ({len(common)} days)"
    )

    # Per-strategy metrics + Deflated Sharpe
    print("\n" + "=" * 100)
    print("PER-STRATEGY METRICS + DEFLATED SHARPE")
    print("=" * 100)
    print(
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'PSR':>7} {'DSR-z':>7} {'DSR-p':>7} {'MDD':>8}"
    )
    print("-" * 100)
    n_trials = len(common.columns)
    for k in common.columns:
        r = common[k]
        m = all_metrics(r)
        dsr = deflated_sharpe_ratio(r, n_trials=n_trials)
        psr = (
            m.get("sharpe", 0) / (m.get("sharpe", 0) + 1)
            if m.get("sharpe", 0) > 0
            else 0
        )
        print(
            f"  {k:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{psr:>+6.3f} "
            f"{dsr.get('dsr_z', float('nan')):>+6.2f} "
            f"{dsr.get('dsr_p', float('nan')):>6.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    # Reality-Check + Hansen-SPA gegen pure_mom_12_1
    if "pure_mom_12_1" in common.columns:
        baseline = common["pure_mom_12_1"]
        challengers = common.drop(columns=["pure_mom_12_1"])
        excess = challengers.subtract(baseline, axis=0)
        wrc = whites_reality_check(excess, n_bootstrap=2000, seed=42)
        spa = hansen_spa_test(excess, n_bootstrap=2000, seed=42)
        print("\n" + "=" * 100)
        print("HYPOTHESIS TEST vs Pure Mom-12/1 LO")
        print("=" * 100)
        print(
            f"White's Reality Check: best={wrc.get('best_strategy')}  p={wrc.get('p_value'):.4f}"
        )
        print(
            f"Hansen-SPA:            best={spa.get('best_strategy')}  p={spa.get('p_value'):.4f}"
        )
        print("  (p < 0.05 -> at least one challenger beats Mom-12/1 LO after MTC)")

    # Bootstrap-CI für Sharpe-Differenzen vs pure_mom_12_1
    print("\n" + "=" * 100)
    print("BLOCK-BOOTSTRAP SHARPE-DIFF vs Pure Mom-12/1 LO (2000x, block=20d)")
    print("=" * 100)
    print(f"{'Challenger':<32} {'mean_diff':>10} {'95% CI':>22} {'p(diff>0)':>11}")
    print("-" * 100)
    if "pure_mom_12_1" in common.columns:
        baseline = common["pure_mom_12_1"]
        for k in common.columns:
            if k == "pure_mom_12_1":
                continue
            bd = _bootstrap_sharpe_diff(common[k], baseline)
            if "error" in bd:
                continue
            ci_str = f"[{bd['ci_low_2.5']:+.2f}, {bd['ci_high_97.5']:+.2f}]"
            # p_value_one_sided is P(a-b <= 0), so 1-p is P(a-b > 0)
            p_gt = 1 - bd["p_value_one_sided"]
            print(f"  {k:<30} {bd['mean_diff']:>+9.3f} {ci_str:>22} {p_gt:>10.3f}")

    # MDD- und Calmar-Ratio
    print("\n" + "=" * 100)
    print("CALMAR RATIO (AnnRet / |MDD|)")
    print("=" * 100)
    print(f"{'Strategy':<32} {'AnnRet':>10} {'MDD':>10} {'Calmar':>10}")
    print("-" * 100)
    for k in common.columns:
        m = all_metrics(common[k])
        ar = m.get("annualized_return", 0)
        mdd = m.get("max_drawdown", 0)
        calmar = ar / abs(mdd) if mdd != 0 else float("nan")
        print(f"  {k:<30} {ar:>+9.2%} {mdd:>+9.2%} {calmar:>+9.3f}")

    summary = {
        "n_days_common": int(len(common)),
        "n_strategies": int(len(common.columns)),
        "reality_check": dict(wrc) if "pure_mom_12_1" in common.columns else None,
        "hansen_spa": dict(spa) if "pure_mom_12_1" in common.columns else None,
    }
    out_path = Path("output/erweiterung_regime_significance.json")
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
