"""Hold-Out-Leakage Permutation-Test (audit §8.7).

Tests whether a strategy's hold-out (test-set) performance is statistically
distinguishable from random — the canonical defence against look-ahead /
leakage from training to test that survives standard backtesting checks.

Two complementary tests run on the same equity curve:

1. **Trade-permutation test (returns-shuffle):** the strategy's returns are
   permuted (without replacement) 1000× to estimate the null distribution
   of Sharpe assuming the returns are an i.i.d. sample. If the observed
   Sharpe falls inside the bulk of the permutation distribution, the
   apparent edge is consistent with random ordering of an arbitrary set
   of returns. p-value = fraction of permutations with Sharpe ≥ observed.

2. **Train/Test split + test-set permutation:** the equity curve is split
   chronologically into train (default 70%) and test (default 30%). The
   train-set Sharpe and test-set Sharpe are reported. The test-set returns
   are then permuted 1000× to estimate p-value on the test slice alone.
   A real out-of-sample edge requires:
     a) test_sharpe > 0 (the strategy works on hold-out)
     b) test-set permutation p-value < 0.05 (the hold-out result is unlikely
        under the null)

Note: this is NOT the same as the C3-030 equity-curve audit (which uses
shuffle_trades for general bootstrap CIs). Hold-out leakage specifically
tests whether the *out-of-sample* slice carries the edge, which is the
audit §8.7 ask.

Usage::

    python scripts/forensic/hold_out_leakage_test.py
    python scripts/forensic/hold_out_leakage_test.py \\
        --input output/equity_curve_baseline.csv \\
        --train-frac 0.7 --n-permutations 2000

Output: JSON + Markdown under ``output/qa/hold_out_leakage_<run_id>.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _annualised_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio. NaN for zero-variance returns."""
    if len(returns) < 2:
        return float("nan")
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    if std <= 0:
        return float("nan")
    return mean / std * float(np.sqrt(periods_per_year))


def permutation_test_sharpe(
    returns: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """One-sided permutation test on Sharpe.

    For each permutation, reshuffle `returns` (without replacement) and
    compute the Sharpe of the permuted sequence. Note: a permutation
    preserves the sample mean and std exactly, so it preserves the
    Sharpe of the SET of returns. The test is therefore meaningful
    only for path-dependent metrics OR when the input is sub-sampled
    (e.g. test-set only — permuting moves day-1 returns to day-N etc.).

    Returns:
        Dict with observed_sharpe, permutation_distribution (mean/std/p10/p50/p90),
        and p_value = fraction with sharpe >= observed.
    """
    if len(returns) < 2:
        return {"error": "insufficient data", "n_obs": int(len(returns))}
    rng = np.random.default_rng(seed)
    observed = _annualised_sharpe(returns, periods_per_year)
    if not np.isfinite(observed):
        return {"error": "observed sharpe undefined (zero variance)"}
    # Note: pure permutation of an i.i.d. return series preserves Sharpe.
    # This test is informative for path-dependent statistics, NOT for
    # Sharpe-on-i.i.d.-data. We compute the permuted Sharpe distribution
    # for completeness and to expose the degeneracy honestly in the report.
    perm_sharpes = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(returns)
        perm_sharpes[i] = _annualised_sharpe(perm, periods_per_year)
    p_value = float((perm_sharpes >= observed).mean())
    return {
        "observed_sharpe": observed,
        "n_permutations": int(n_permutations),
        "perm_mean": float(perm_sharpes.mean()),
        "perm_std": float(perm_sharpes.std(ddof=1)),
        "perm_p10": float(np.percentile(perm_sharpes, 10)),
        "perm_p50": float(np.percentile(perm_sharpes, 50)),
        "perm_p90": float(np.percentile(perm_sharpes, 90)),
        "p_value_sharpe_ge_observed": p_value,
        "degeneracy_note": (
            "Permuting an i.i.d. returns sample preserves the sample Sharpe "
            "exactly — the p-value is therefore informative only when the "
            "metric is path-dependent (e.g. MDD) or when returns are "
            "autocorrelated. For Sharpe-on-test-set, the practical signal "
            "comes from train_vs_test split below."
        ),
    }


def permutation_test_mdd(
    returns: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """One-sided permutation test on MaxDrawdown (path-dependent — meaningful).

    Unlike Sharpe-on-iid, MDD IS path-dependent: reshuffling a return series
    changes which clusters of negative returns chain together. A real edge
    that comes from clustering risk-on/risk-off regimes will show a
    distinct MDD vs random orderings.

    p_value = fraction of permutations with MDD <= observed (less drawdown
    than observed under null is unlikely if observed already minimal).
    """
    if len(returns) < 2:
        return {"error": "insufficient data"}
    rng = np.random.default_rng(seed)
    observed_equity = np.cumprod(1.0 + returns)
    rm = np.maximum.accumulate(observed_equity)
    observed_mdd = float((observed_equity / rm - 1.0).min())
    perm_mdds = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(returns)
        eq = np.cumprod(1.0 + perm)
        rm_p = np.maximum.accumulate(eq)
        perm_mdds[i] = float((eq / rm_p - 1.0).min())
    # P(perm_mdd >= observed_mdd) — i.e. permuted has SHALLOWER drawdown
    # than observed. Low p ⇒ observed MDD is *better* than random ordering.
    p_value = float((perm_mdds >= observed_mdd).mean())
    return {
        "observed_mdd": observed_mdd,
        "n_permutations": int(n_permutations),
        "perm_mean": float(perm_mdds.mean()),
        "perm_std": float(perm_mdds.std(ddof=1)),
        "perm_p10": float(np.percentile(perm_mdds, 10)),
        "perm_p50": float(np.percentile(perm_mdds, 50)),
        "perm_p90": float(np.percentile(perm_mdds, 90)),
        "p_value_mdd_ge_observed": p_value,
    }


def train_test_split_audit(
    returns: np.ndarray,
    train_frac: float = 0.7,
    n_permutations: int = 1000,
    seed: int = 42,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Split chronologically into train/test, compute split-Sharpes, permute
    test-set MDD for p-value.

    The Sharpe drop from train → test is the qualitative leakage indicator:
    a strategy that overfits the train set typically shows test_sharpe
    << train_sharpe. The MDD permutation p-value quantifies whether the
    test-set drawdown is path-dependent vs. random ordering.
    """
    if not (0.1 < train_frac < 0.9):
        raise ValueError(f"train_frac must be in (0.1, 0.9), got {train_frac}")
    n = len(returns)
    if n < 20:
        return {"error": "insufficient data (n<20)", "n_obs": int(n)}
    cut = int(n * train_frac)
    train = returns[:cut]
    test = returns[cut:]
    train_sharpe = _annualised_sharpe(train, periods_per_year)
    test_sharpe = _annualised_sharpe(test, periods_per_year)
    # Path-dependent MDD permutation on test-set
    test_mdd_perm = permutation_test_mdd(test, n_permutations, seed)
    return {
        "train_frac": float(train_frac),
        "n_train": int(cut),
        "n_test": int(n - cut),
        "train_sharpe": train_sharpe,
        "test_sharpe": test_sharpe,
        "sharpe_decay_train_to_test": (
            float(train_sharpe - test_sharpe)
            if np.isfinite(train_sharpe) and np.isfinite(test_sharpe)
            else float("nan")
        ),
        "test_mdd_permutation": test_mdd_perm,
    }


def run_hold_out_leakage_test(
    equity_curve_path: Path,
    train_frac: float = 0.7,
    n_permutations: int = 1000,
    periods_per_year: int = 252,
    seed: int = 42,
) -> dict[str, Any]:
    """Full pipeline: load equity, run both tests, aggregate verdict."""
    if not equity_curve_path.exists():
        raise FileNotFoundError(f"equity curve not found: {equity_curve_path}")
    df = pd.read_csv(equity_curve_path)
    if "equity" not in df.columns:
        raise ValueError(f"missing 'equity' column in {equity_curve_path}")
    equity = df["equity"].to_numpy(dtype=float)
    if "daily_return" in df.columns:
        returns = df["daily_return"].dropna().to_numpy(dtype=float)
    else:
        returns = pd.Series(equity).pct_change().dropna().to_numpy(dtype=float)
    full_sharpe_perm = permutation_test_sharpe(
        returns, n_permutations, seed, periods_per_year
    )
    full_mdd_perm = permutation_test_mdd(returns, n_permutations, seed)
    split = train_test_split_audit(
        returns, train_frac, n_permutations, seed, periods_per_year
    )
    # Verdict: out-of-sample edge requires test_sharpe > 0 AND
    # test_mdd_perm.p_value_mdd_ge_observed < 0.05 (MDD is path-dependent
    # and the observed MDD is in the favourable tail).
    test_sharpe = split.get("test_sharpe")
    test_mdd_p = split.get("test_mdd_permutation", {}).get("p_value_mdd_ge_observed")
    if not np.isfinite(test_sharpe if test_sharpe is not None else float("nan")):
        verdict = "undefined"
    elif test_sharpe is not None and test_sharpe > 0:
        if test_mdd_p is not None and test_mdd_p < 0.05:
            verdict = "hold_out_edge_significant"
        elif test_mdd_p is not None and test_mdd_p < 0.20:
            verdict = "hold_out_edge_weak"
        else:
            verdict = "hold_out_edge_indistinguishable_from_random"
    else:
        verdict = "hold_out_negative_sharpe"
    return {
        "input_path": str(equity_curve_path),
        "n_periods": int(len(returns)),
        "params": {
            "train_frac": float(train_frac),
            "n_permutations": int(n_permutations),
            "periods_per_year": int(periods_per_year),
            "seed": int(seed),
        },
        "full_series": {
            "sharpe_permutation": full_sharpe_perm,
            "mdd_permutation": full_mdd_perm,
        },
        "train_test_split": split,
        "verdict": verdict,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hold-Out-Leakage Permutation Test (§8.7)",
        "",
        f"**Input:** `{report['input_path']}`",
        f"**Periods:** {report['n_periods']}",
        f"**Train fraction:** {report['params']['train_frac']:.0%}",
        f"**Permutations:** {report['params']['n_permutations']}",
        "",
        f"## Verdict: `{report['verdict']}`",
        "",
        "## Full-Series Tests",
    ]
    fs = report["full_series"]["sharpe_permutation"]
    fm = report["full_series"]["mdd_permutation"]
    if "error" not in fs:
        lines.append(
            f"- **Observed Sharpe:** {fs['observed_sharpe']:.4f} | "
            f"perm mean {fs['perm_mean']:.4f} | "
            f"p(perm ≥ observed) = {fs['p_value_sharpe_ge_observed']:.4f}"
        )
        lines.append(f"  - Note: {fs['degeneracy_note']}")
    if "error" not in fm:
        lines.append(
            f"- **Observed MDD:** {fm['observed_mdd']:.4f} | "
            f"perm median {fm['perm_p50']:.4f} | "
            f"p(perm ≥ observed) = {fm['p_value_mdd_ge_observed']:.4f}"
        )
    lines.append("")
    lines.append("## Train/Test Split Audit")
    s = report["train_test_split"]
    lines.append(
        f"- **Train (first {s['n_train']} periods):** Sharpe {s['train_sharpe']:.4f}"
    )
    lines.append(
        f"- **Test (last {s['n_test']} periods):** Sharpe {s['test_sharpe']:.4f}"
    )
    lines.append(
        f"- **Sharpe decay train → test:** {s['sharpe_decay_train_to_test']:.4f}"
    )
    mp = s.get("test_mdd_permutation", {})
    if "p_value_mdd_ge_observed" in mp:
        lines.append(
            f"- **Test-set MDD permutation:** observed {mp['observed_mdd']:.4f}, "
            f"perm median {mp['perm_p50']:.4f}, "
            f"p = {mp['p_value_mdd_ge_observed']:.4f}"
        )
    lines.append("")
    lines.append("## Verdict Semantics")
    lines.append("- `hold_out_edge_significant`: test_sharpe > 0 AND p < 0.05")
    lines.append("- `hold_out_edge_weak`: test_sharpe > 0 AND p in [0.05, 0.20)")
    lines.append(
        "- `hold_out_edge_indistinguishable_from_random`: test_sharpe > 0 AND p ≥ 0.20"
    )
    lines.append("- `hold_out_negative_sharpe`: test_sharpe ≤ 0")
    lines.append("- `undefined`: test_sharpe NaN (insufficient data / zero variance)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/equity_curve_baseline.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/qa"),
    )
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = run_hold_out_leakage_test(
        equity_curve_path=args.input,
        train_frac=args.train_frac,
        n_permutations=args.n_permutations,
        periods_per_year=args.periods_per_year,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    run_id = args.input.stem
    json_path = args.out / f"hold_out_leakage_{run_id}.json"
    md_path = args.out / f"hold_out_leakage_{run_id}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info("[hold_out_leakage] JSON: %s", json_path)
    logger.info("[hold_out_leakage] Markdown: %s", md_path)
    logger.info("[hold_out_leakage] verdict=%s", report["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
