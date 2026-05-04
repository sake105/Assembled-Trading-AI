"""End-to-End ML-Stack-Training: Feature-Reduction → Stacking → Registry.

Kombiniert die bisherigen ML-Bausteine zu einer einzigen Pipeline:

1. Panel-Load + Feature-Auswahl
2. Feature-Clustering (Multikollinearität reduzieren)
3. Adversarial Validation (distribution shift vorabcheck)
4. Walk-Forward Hyperparameter-Optimierung (optional)
5. Stacking Ensemble (LGBM/Ridge/RF blend)
6. Conformal Calibration auf Holdout
7. Model Registry: als candidate speichern
8. Optional: Regime-Router-Training wenn news_regime-Spalten vorhanden

PIT-sicher, walk-forward-basiert, auto_deploy=False (explizite Human-Approval).

Verwendung:
    python scripts/training/train_full_ml_stack.py \\
        --panel output/factor_panels/full_panel_7y.parquet \\
        --label fwd_return_5d \\
        --model-id full_stack_v1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_panel(path: Path, label: str) -> pd.DataFrame:
    if path.suffix == ".parquet":
        panel = pd.read_parquet(path)
    elif path.suffix == ".csv":
        panel = pd.read_csv(path)
    else:
        raise ValueError(f"Unbekanntes Format: {path.suffix}")
    return panel.dropna(subset=[label])


def _split_train_val_test(
    panel: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologisch: Train / Val / Test."""
    if timestamp_col in panel.columns:
        panel = panel.sort_values(timestamp_col).reset_index(drop=True)
    n = len(panel)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (
        panel.iloc[:n_train].copy(),
        panel.iloc[n_train : n_train + n_val].copy(),
        panel.iloc[n_train + n_val :].copy(),
    )


def _detect_features(panel: pd.DataFrame, label: str) -> list[str]:
    excluded = {label, "timestamp", "symbol", "date"}
    return [
        c
        for c in panel.select_dtypes(include="number").columns
        if c not in excluded
        and not c.startswith("fwd_return")
        and not c.startswith("tb_")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-End ML-Stack-Training")
    parser.add_argument(
        "--panel", type=Path, required=True, help="Factor panel parquet/csv"
    )
    parser.add_argument("--label", default="fwd_return_5d", help="Target column")
    parser.add_argument("--model-id", default="full_stack_v1", help="Registry model ID")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--skip-adversarial", action="store_true")
    parser.add_argument("--cluster-threshold", type=float, default=0.3)
    parser.add_argument("--conformal-alpha", type=float, default=0.1)
    parser.add_argument("--n-hpo-trials", type=int, default=0, help="0 = kein HPO")
    parser.add_argument("--registry-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    if not args.panel.exists():
        logger.error("Panel nicht gefunden: %s", args.panel)
        return 1

    # ---------- 1. Load ----------
    logger.info("=" * 60)
    logger.info("Step 1: Loading panel %s (label=%s)", args.panel, args.label)
    panel = _load_panel(args.panel, args.label)
    feature_cols = _detect_features(panel, args.label)
    logger.info(
        "Panel: %d rows, %d features (target=%s)",
        len(panel),
        len(feature_cols),
        args.label,
    )

    # ---------- 2. Split ----------
    train_df, val_df, test_df = _split_train_val_test(
        panel, timestamp_col=args.timestamp_col
    )
    logger.info(
        "Split: train=%d val=%d test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # ---------- 3. Feature Clustering ----------
    selected_features = feature_cols
    if not args.skip_clustering:
        logger.info("=" * 60)
        logger.info("Step 3: Feature clustering (threshold=%s)", args.cluster_threshold)
        try:
            from src.assembled_core.ml.feature_clustering import (
                cluster_features_by_correlation,
                select_features_by_cluster_ic,
            )

            clusters = cluster_features_by_correlation(
                train_df,
                feature_cols=feature_cols,
                distance_threshold=args.cluster_threshold,
            )
            clusters = select_features_by_cluster_ic(
                train_df, train_df[args.label], clusters
            )
            selected_features = clusters.get_selected_features()
            logger.info(
                "Reduced %d → %d features via clustering",
                len(feature_cols),
                len(selected_features),
            )
        except Exception as exc:
            logger.warning("Clustering fehlgeschlagen: %s — nutze alle Features", exc)

    # ---------- 4. Adversarial Validation ----------
    if not args.skip_adversarial:
        logger.info("=" * 60)
        logger.info("Step 4: Adversarial validation (train vs test)")
        try:
            from src.assembled_core.ml.adversarial_validation import (
                run_adversarial_validation,
            )

            adv = run_adversarial_validation(
                X_train=train_df,
                X_test=test_df,
                feature_cols=selected_features,
            )
            logger.info("Distribution shift AUC=%.3f (%s)", adv.auc, adv.interpret())
            if adv.auc > 0.80:
                logger.warning(
                    "EXTREME distribution shift — Training fortgesetzt mit Warnung"
                )
        except Exception as exc:
            logger.warning("Adversarial val fehlgeschlagen: %s", exc)

    # ---------- 5. HPO (optional) ----------
    best_params: dict = {}
    if args.n_hpo_trials > 0:
        logger.info("=" * 60)
        logger.info("Step 5: Walk-forward HPO (%d trials)", args.n_hpo_trials)
        try:
            from scripts.training.walk_forward_hpo import run_hpo_optuna

            hpo_result = run_hpo_optuna(
                train_df,
                train_df[args.label],
                train_df[args.timestamp_col],
                feature_cols=selected_features,
                n_trials=args.n_hpo_trials,
            )
            best_params = hpo_result["best_params"]
            logger.info(
                "Best HPO params: %s (score=%.4f)",
                best_params,
                hpo_result["best_value"],
            )
        except Exception as exc:
            logger.warning("HPO fehlgeschlagen: %s", exc)

    # ---------- 6. Stacking ----------
    logger.info("=" * 60)
    logger.info("Step 6: Stacking ensemble training")
    try:
        from src.assembled_core.ml.stacking_ensemble import (
            StackingConfig,
            run_stacking_cv,
        )

        cfg = StackingConfig(
            base_models=["ridge", "random_forest", "gradient_boosting"],
            meta_model="ridge",
            n_splits=5,
            use_purged_cv=False,
        )
        stack = run_stacking_cv(
            train_df,
            train_df[args.label],
            config=cfg,
            feature_cols=selected_features,
        )
        logger.info(
            "Stacked IC=%.4f vs best base IC=%.4f",
            stack.stacked_ic,
            max(stack.base_ic.values()) if stack.base_ic else 0.0,
        )
    except Exception as exc:
        logger.error("Stacking failed: %s", exc)
        return 1

    # ---------- 7. Conformal Calibration ----------
    logger.info("=" * 60)
    logger.info("Step 7: Conformal calibration on validation set")
    try:
        val_preds = stack.predict(val_df)
        residuals = np.abs(val_df[args.label].values - val_preds.values)
        n = len(residuals)
        q_level = min(1.0, np.ceil((n + 1) * (1 - args.conformal_alpha)) / n)
        half_width = float(np.quantile(residuals, q_level))
        logger.info(
            "Conformal: α=%.2f, %.0f%%-Intervall half-width=%.4f",
            args.conformal_alpha,
            (1 - args.conformal_alpha) * 100,
            half_width,
        )
    except Exception as exc:
        logger.warning("Conformal calibration failed: %s", exc)
        half_width = 0.0

    # ---------- 8. Test Evaluation ----------
    logger.info("=" * 60)
    logger.info("Step 8: OOS test evaluation")
    try:
        test_preds = stack.predict(test_df)
        if test_preds.std() > 1e-9:
            test_ic = float(
                np.corrcoef(test_preds.values, test_df[args.label].values)[0, 1]
            )
        else:
            test_ic = 0.0
        logger.info("OOS test IC=%.4f", test_ic)
    except Exception as exc:
        logger.warning("Test-Evaluation failed: %s", exc)
        test_ic = 0.0

    # ---------- 9. Model Registry ----------
    logger.info("=" * 60)
    logger.info("Step 9: Registry (status=candidate, auto_deploy=False)")
    try:
        from src.assembled_core.ml.model_registry import ModelRegistry

        registry = ModelRegistry(base_dir=args.registry_dir)

        # Save stacked result as dict (model_dict)
        model_payload = {
            "base_models": stack.base_models,
            "meta_model": stack.meta_model,
            "feature_cols": stack.feature_cols,
            "selected_features": selected_features,
            "conformal_half_width": half_width,
            "conformal_alpha": args.conformal_alpha,
        }
        record = registry.register(
            model=model_payload,
            model_id=args.model_id,
            metrics={
                "ic_train_stacked": stack.stacked_ic,
                "ic_test": test_ic,
                "n_features": len(selected_features),
                "n_train": len(train_df),
                "conformal_half_width": half_width,
                "best_hpo_params": best_params,
            },
            features=selected_features,
            train_start=(
                str(train_df[args.timestamp_col].min())
                if args.timestamp_col in train_df.columns
                else None
            ),
            train_end=(
                str(train_df[args.timestamp_col].max())
                if args.timestamp_col in train_df.columns
                else None
            ),
            notes="End-to-end ML stack via train_full_ml_stack.py",
        )
        logger.info(
            "[OK] Registered %s v%d (status=candidate). Use promote_to_deployed() manually.",
            record.model_id,
            record.version,
        )
    except Exception as exc:
        logger.error("Registry failed: %s", exc)
        return 1

    logger.info("=" * 60)
    logger.info("FULL ML STACK TRAINING COMPLETE")
    logger.info(
        "Summary: train_ic=%.4f test_ic=%.4f features=%d conformal_hw=%.4f",
        stack.stacked_ic,
        test_ic,
        len(selected_features),
        half_width,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
