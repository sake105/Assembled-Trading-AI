#!/usr/bin/env python
"""Meta-Labeling auf Master-Allocator-Returns — Lopez de Prado §3.6.

Pipeline
--------
1. Lade Master-Allocator Equity (master_return = 70% SA-VolTarget + 30% XA-Hybrid).
2. Triple-Barrier-Labels (±2.5%, horizon=21d).
3. Features: trailing-Vol/Sharpe, Drawdown, VIX, Yield-Curve-Spread.
4. Walk-Forward Meta-Klassifikator (Logistic + RandomForest).
5. Meta-Gate: trade nur wenn predict=1, sonst Cash.
6. Vergleich: Pure-Master vs Meta-gated-Master.
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
from erweiterung.ml.meta_labeling_master import (  # noqa: E402
    MetaLabelingConfig,
    apply_meta_gate,
    build_features,
    triple_barrier_simple,
    walk_forward_meta_predictions,
)


def main():
    eq_csv = Path("output/erweiterung_master_allocation_equity.csv")
    if not eq_csv.exists():
        print(f"ERROR: {eq_csv} not found.")
        return 1

    df = pd.read_csv(eq_csv)
    first = df.columns[0]
    df = df.rename(columns={first: "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()

    master_ret = df["Master_70_30"].pct_change().dropna()
    pure_mom_ret = df["Pure_Mom_12_1_LO"].pct_change().dropna()
    bench_ret = df["60_40_Classic"].pct_change().dropna()

    # Macro panel
    macro_path = Path("output/macro.parquet")
    macro = None
    if macro_path.exists():
        macro = pd.read_parquet(macro_path)
        macro["timestamp"] = pd.to_datetime(macro["timestamp"], utc=True)
        macro = macro.set_index("timestamp").sort_index()

    print("Computing triple-barrier labels ...")
    cfg = MetaLabelingConfig(
        take_profit_pct=0.025,
        stop_loss_pct=0.025,
        horizon_days=21,
        train_window=378,  # 18 Monate
        test_window=63,  # 3 Monate
        meta_threshold=0.50,
    )
    labels = triple_barrier_simple(master_ret, cfg)
    n_pos = (labels == 1.0).sum()
    n_neg = (labels == -1.0).sum()
    n_zero = (labels == 0.0).sum()
    print(f"Labels: +1={n_pos}, -1={n_neg}, 0={n_zero}, NaN={labels.isna().sum()}")
    print(f"Base-rate (P(label=+1)): {n_pos / max(labels.notna().sum(), 1):.1%}")

    feat = build_features(master_ret, macro_panel=macro, config=cfg)
    print(f"Features: {feat.shape[1]} cols, {len(feat)} rows")

    # Walk-Forward für 2 Modelle
    print("\nWalk-Forward Meta-Klassifikator (Logistic Regression) ...")
    preds_lr = walk_forward_meta_predictions(feat, labels, cfg, model_type="logistic")
    print(f"OOS Predictions: {len(preds_lr)} samples")

    print("Walk-Forward Meta-Klassifikator (Random Forest) ...")
    preds_rf = walk_forward_meta_predictions(feat, labels, cfg, model_type="rf")
    print(f"OOS Predictions: {len(preds_rf)} samples")

    # Classification accuracy
    for name, preds in [("Logistic", preds_lr), ("RandomForest", preds_rf)]:
        if preds.empty:
            continue
        acc = (preds["predicted"] == preds["actual_label"]).mean()
        base = preds["actual_label"].mean()
        print(f"\n{name}: accuracy = {acc:.3f} (base-rate = {base:.3f})")
        print(f"  Trade-Anteil (predicted=1): {preds['predicted'].mean():.1%}")

    # Apply Meta-Gate
    gated_lr = apply_meta_gate(master_ret, preds_lr)
    gated_rf = apply_meta_gate(master_ret, preds_rf)

    # Pure Master auf OOS-Period
    oos_idx = preds_lr.index
    master_oos = master_ret.loc[oos_idx[0] : oos_idx[-1]]
    bench_oos = bench_ret.loc[oos_idx[0] : oos_idx[-1]]
    pure_mom_oos = pure_mom_ret.loc[oos_idx[0] : oos_idx[-1]]

    print("\n" + "=" * 100)
    print("META-LABELING META-GATED MASTER vs PURE MASTER (OOS)")
    print("=" * 100)
    print(
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8}"
    )
    print("-" * 100)
    candidates = {
        "60_40_Classic (OOS)": bench_oos,
        "Pure_Mom_12_1_LO (OOS)": pure_mom_oos,
        "Master_70_30 (OOS, pure)": master_oos,
        "Master_70_30 + Logistic Meta-Gate": gated_lr,
        "Master_70_30 + RandomForest Meta-Gate": gated_rf,
    }
    metrics_dump = {}
    for name, ret in candidates.items():
        m = all_metrics(ret.dropna())
        metrics_dump[name] = m
        print(
            f"  {name:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%}"
        )

    # Calmar-Bootstrap vs Pure-Master
    print("\n" + "=" * 100)
    print("CALMAR-BOOTSTRAP vs Pure Master_70_30 (OOS)")
    print("=" * 100)
    print(f"{'Challenger':<36} {'obs_diff':>9} {'mean_diff':>10} {'p(>0)':>7}")
    print("-" * 100)
    for name, ret in candidates.items():
        if name == "Master_70_30 (OOS, pure)":
            continue
        out = calmar_diff_bootstrap(
            ret.dropna(),
            master_oos.dropna(),
            n_bootstrap=2000,
            avg_block_size=20,
            seed=42,
        )
        if "error" in out:
            continue
        p_gt = 1.0 - out["p_value_one_sided_greater"]
        print(
            f"  {name:<34} {out['observed_diff']:>+8.3f} {out['mean_diff']:>+9.3f} {p_gt:>6.3f}"
        )

    # Save
    Path("output/erweiterung_meta_labeling_summary.json").write_text(
        json.dumps(
            {
                "label_distribution": {
                    "pos": int(n_pos),
                    "neg": int(n_neg),
                    "zero": int(n_zero),
                },
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
    preds_lr.to_csv("output/erweiterung_meta_labeling_predictions_lr.csv")
    preds_rf.to_csv("output/erweiterung_meta_labeling_predictions_rf.csv")
    print("\nSaved -> output/erweiterung_meta_labeling_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
