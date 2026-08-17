# src/assembled_core/api/routers/diagnostics.py
"""Backend diagnostics endpoints — module wiring + feature drift (V2 mockup)."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from src.assembled_core.api.models import (
    FeatureDriftItem,
    FeatureDriftResponse,
    ModuleStatus,
    ModulesResponse,
)
from src.assembled_core.logging_utils import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Known module wiring status (ground truth from plan + repo audit 2026-05-05)
_MODULE_REGISTRY: list[dict] = [
    # wired
    {
        "module": "conviction_engine",
        "status": "wired",
        "callers": 4,
        "path": "src/assembled_core/intel/conviction_engine.py",
    },
    {
        "module": "market_stress",
        "status": "wired",
        "callers": 6,
        "path": "src/assembled_core/risk/market_stress.py",
    },
    {
        "module": "conformal_position",
        "status": "wired",
        "callers": 6,
        "path": "src/assembled_core/ml/conformal_position.py",
    },
    {
        "module": "signal_diagnostics",
        "status": "wired",
        "callers": 3,
        "path": "src/assembled_core/signals/signal_diagnostics.py",
    },
    {
        "module": "circuit_breaker",
        "status": "wired",
        "callers": 1,
        "path": "src/assembled_core/risk/circuit_breaker.py",
    },
    {
        "module": "behavioral_features",
        "status": "wired",
        "callers": 1,
        "path": "src/assembled_core/features/behavioral_features.py",
    },
    {
        "module": "altdata_loader",
        "status": "wired",
        "callers": 2,
        "path": "src/assembled_core/data/altdata_loader.py",
    },
    # canary
    {
        "module": "walk_forward",
        "status": "canary",
        "callers": 0,
        "path": "src/assembled_core/qa/walk_forward.py",
    },
    {
        "module": "cpcv_validation",
        "status": "canary",
        "callers": 0,
        "path": "src/assembled_core/qa/cpcv.py",
    },
    {
        "module": "leakage_analyzer",
        "status": "canary",
        "callers": 0,
        "path": "src/assembled_core/qa/leakage_analyzer.py",
    },
    {
        "module": "profit_targets",
        "status": "canary",
        "callers": 0,
        "path": "src/assembled_core/execution/profit_targets.py",
    },
    {
        "module": "edgar_source",
        "status": "archived",
        "callers": 0,
        "path": "archive/orphaned_code_2026-08-17/sources/edgar_source.py",
    },
    # canary (newly promoted from phantom — files exist, not yet in hot path)
    {
        "module": "disclosures_confirm",
        "status": "canary",
        "callers": 1,
        "path": "src/assembled_core/risk/disclosures_confirm.py",
    },
    {
        "module": "quantile_models",
        "status": "canary",
        "callers": 1,
        "path": "src/assembled_core/ml/quantile_models.py",
    },
    {
        "module": "retraining_scheduler",
        "status": "canary",
        "callers": 1,
        "path": "src/assembled_core/ml/retraining_scheduler.py",
    },
    {
        "module": "lime_explainer",
        "status": "canary",
        "callers": 1,
        "path": "src/assembled_core/ml/lime_explainer.py",
    },
    {
        "module": "covariance",
        "status": "canary",
        "callers": 4,
        "path": "src/assembled_core/portfolio/covariance.py",
    },
    # orphan
    {
        "module": "bootstrap_metrics",
        "status": "orphan",
        "callers": 0,
        "path": "src/assembled_core/qa/bootstrap_metrics.py",
    },
    {
        "module": "monte_carlo_paths",
        "status": "orphan",
        "callers": 0,
        "path": "src/assembled_core/qa/monte_carlo_paths.py",
    },
]


@router.get("/diagnostics/modules", response_model=ModulesResponse)
def get_modules_status() -> ModulesResponse:
    """Return module wiring status for all catalogued backend modules."""
    try:
        from pathlib import Path

        root = Path(__file__).resolve().parents[4]

        modules: list[ModuleStatus] = []
        counts = {"wired": 0, "canary": 0, "orphan": 0, "phantom": 0}

        for entry in _MODULE_REGISTRY:
            status = entry["status"]
            path = entry.get("path")
            # Verify file exists for non-phantom
            if path and status != "phantom":
                if not (root / path).exists():
                    status = "phantom"
            counts[status] = counts.get(status, 0) + 1
            modules.append(
                ModuleStatus(
                    module=entry["module"],
                    status=status,
                    callers=entry.get("callers", 0),
                    path=path,
                )
            )

        return ModulesResponse(
            registry_static=True,
            registry_as_of="2026-05-05",
            registry_caveat=(
                "static hand-maintained registry (audit 2026-08-16): status/"
                "callers reflect the 2026-05-05 snapshot, NOT a live scan; "
                "moved modules show as 'phantom' even when they exist elsewhere"
            ),
            total=len(modules),
            wired=counts.get("wired", 0),
            canary=counts.get("canary", 0),
            orphan=counts.get("orphan", 0),
            phantom=counts.get("phantom", 0),
            modules=modules,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.error("diagnostics/modules error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/diagnostics/feature-drift", response_model=FeatureDriftResponse)
def get_feature_drift() -> FeatureDriftResponse:
    """Compute PSI-based feature drift for panel features (base=60d, current=20d)."""
    try:
        from pathlib import Path
        import pandas as pd
        from src.assembled_core.qa.drift_detection import compute_psi
        from src.assembled_core.config import OUTPUT_DIR

        # Load panel parquet — find latest (check data/ and data/sample/)
        repo_root = Path(__file__).resolve().parents[4]
        panel_files = sorted((repo_root / "data").rglob("watchlist*.parquet"))
        if not panel_files:
            panel_files = sorted(OUTPUT_DIR.rglob("watchlist*.parquet"))
        if not panel_files:
            raise HTTPException(
                status_code=404, detail="No panel parquet found for drift analysis"
            )

        df = pd.read_parquet(panel_files[-1])

        # Select numeric feature columns; if panel is raw OHLCV, derive rolling features
        skip = {
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
        }
        feature_cols = [c for c in df.select_dtypes("number").columns if c not in skip][
            :30
        ]

        if not feature_cols and "close" in df.columns:
            # Derive basic technical features from OHLCV for drift detection
            df["ret_1d"] = df["close"].pct_change(1)
            df["ret_5d"] = df["close"].pct_change(5)
            df["ret_20d"] = df["close"].pct_change(20)
            df["vol_20d"] = df["ret_1d"].rolling(20).std()
            df["vol_ratio"] = df["vol_20d"] / df["ret_1d"].rolling(60).std().replace(
                0, float("nan")
            )
            if "volume" in df.columns:
                df["vol_norm"] = df["volume"] / df["volume"].rolling(20).mean().replace(
                    0, float("nan")
                )
            feature_cols = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "vol_ratio"] + (
                ["vol_norm"] if "volume" in df.columns else []
            )

        if not feature_cols:
            raise HTTPException(
                status_code=404, detail="No feature columns found in panel"
            )

        # Sort by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        n = len(df)
        base_df = df.iloc[: max(n - 20, n // 2)]
        current_df = df.iloc[max(n - 20, n // 2) :]

        drift_items: list[FeatureDriftItem] = []
        for col in feature_cols:
            try:
                psi = compute_psi(
                    base_df[col].dropna(),
                    current_df[col].dropna(),
                )
                flag = (
                    "SEVERE" if psi >= 0.2 else ("MODERATE" if psi >= 0.1 else "NONE")
                )
                drift_items.append(
                    FeatureDriftItem(feature=col, psi=round(psi, 4), drift_flag=flag)
                )
            except Exception:
                pass

        drift_items.sort(key=lambda x: x.psi, reverse=True)
        n_drifted = sum(1 for d in drift_items if d.drift_flag != "NONE")
        worst = max(
            (d.drift_flag for d in drift_items),
            default="NONE",
            key=lambda f: {"NONE": 0, "MODERATE": 1, "SEVERE": 2}[f],
        )

        return FeatureDriftResponse(
            as_of=datetime.now(timezone.utc).isoformat(),
            n_features=len(drift_items),
            n_drifted=n_drifted,
            overall_severity=worst,
            features=drift_items,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("diagnostics/feature-drift error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
