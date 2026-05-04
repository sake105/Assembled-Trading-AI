"""Model Comparison Report: Candidate vs Deployed auf Recent-Daten.

Vergleicht zwei Modelle (z.B. neuer Candidate vs. aktuell deployed) auf
demselben aktuellen Testset. Liefert:

- IC / Hit-Rate / Sharpe pro Modell
- Diff-Metriken (Δ IC, Δ Hit-Rate, Δ Sharpe)
- Statistische Signifikanz via Diebold-Mariano-Test
- Prediction-Korrelation (hoher Wert → ähnlich)
- Entscheidungshilfe: Deploy / Hold / Reject

Verwendung:
    python scripts/analysis/compare_models.py \\
        --deployed models/meta/deployed.joblib \\
        --candidate models/meta/v3.joblib \\
        --panel output/factor_panels/full_panel_7y.parquet \\
        --label fwd_return_5d \\
        --n-days 60

Output: output/ops/model_comparison_{date}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_model(path: Path):
    import joblib

    try:
        return joblib.load(path)
    except (EOFError, Exception) as exc:
        raise RuntimeError(
            f"[compare_models] Failed to load model from {path}: {exc}"
        ) from exc


def _predict(model, X: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Robustes Prediction: versucht predict → predict_proba → attribute-Zugriff."""
    X_vals = X[feature_cols].fillna(0.0).values
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_vals)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return pd.Series(proba[:, 1], index=X.index)
        except Exception:
            pass
    if hasattr(model, "predict"):
        try:
            return pd.Series(model.predict(X_vals), index=X.index)
        except Exception:
            pass
    # StackingResult-ähnliches Objekt
    if hasattr(model, "feature_cols") and hasattr(model, "base_models"):
        try:
            return model.predict(X)
        except Exception:
            pass
    raise RuntimeError(f"Kann Modell {type(model).__name__} nicht predikten")


def _compute_metrics(preds: pd.Series, actuals: pd.Series) -> dict:
    preds = preds.dropna()
    actuals = actuals.loc[preds.index].dropna()
    preds = preds.loc[actuals.index]

    if len(preds) < 10 or preds.std() < 1e-9 or actuals.std() < 1e-9:
        return {
            "ic": 0.0,
            "hit_rate": 0.0,
            "sharpe": 0.0,
            "mse": (
                float(np.mean((preds.values - actuals.values) ** 2))
                if len(preds)
                else 0.0
            ),
            "n_obs": len(preds),
        }

    ic = float(np.corrcoef(preds.values, actuals.values)[0, 1])
    if np.isnan(ic):
        ic = 0.0

    # Hit rate: sign(pred) == sign(actual)
    hit = float((np.sign(preds.values) == np.sign(actuals.values)).mean())

    # Strategy returns = sign(pred) × actual
    strategy_rets = np.sign(preds.values) * actuals.values
    if strategy_rets.std() > 1e-9:
        sharpe = float(strategy_rets.mean() / strategy_rets.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    mse = float(np.mean((preds.values - actuals.values) ** 2))

    return {
        "ic": round(ic, 4),
        "hit_rate": round(hit, 4),
        "sharpe": round(sharpe, 3),
        "mse": round(mse, 6),
        "n_obs": len(preds),
    }


def _diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
) -> dict:
    """Diebold-Mariano-Test: sind zwei Forecast-Fehler-Serien signifikant verschieden?

    H_0: gleiche Forecast-Genauigkeit
    H_1: Modell A hat andere Fehler als Modell B
    Positive DM → B besser, negative → A besser.
    """
    d = errors_a**2 - errors_b**2
    n = len(d)
    if n < 10 or d.std() < 1e-12:
        return {"statistic": 0.0, "p_value": 1.0, "n_obs": n}

    d_mean = float(d.mean())
    d_var = float(d.var(ddof=1))
    se = np.sqrt(d_var / n)
    dm_stat = d_mean / se if se > 0 else 0.0

    # 2-sided p-value approximation via normal distribution
    try:
        from scipy.stats import norm

        p_val = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))
    except ImportError:
        p_val = 2.0 * np.exp(-0.5 * dm_stat**2) / np.sqrt(2 * np.pi)
        p_val = float(min(1.0, max(0.0, p_val)))

    return {
        "statistic": round(float(dm_stat), 3),
        "p_value": round(float(p_val), 4),
        "n_obs": n,
    }


def _decision(deployed_m: dict, candidate_m: dict, dm: dict, corr: float) -> str:
    """Heuristik für Deploy/Hold/Reject."""
    ic_delta = candidate_m["ic"] - deployed_m["ic"]
    sharpe_delta = candidate_m["sharpe"] - deployed_m["sharpe"]

    # Klarer Gewinn + signifikant
    if ic_delta > 0.02 and sharpe_delta > 0.2 and dm["p_value"] < 0.10:
        return "DEPLOY_CANDIDATE"
    # Leichte Verbesserung aber nicht signifikant
    if ic_delta > 0 and sharpe_delta > 0:
        return "HOLD_CONSIDER_MORE_DATA"
    # Schlechter → reject
    if ic_delta < -0.01 or sharpe_delta < -0.2:
        return "REJECT_CANDIDATE"
    return "HOLD"


def compare(
    deployed_path: Path,
    candidate_path: Path,
    panel_path: Path,
    label: str,
    n_days: int = 60,
    timestamp_col: str = "timestamp",
    feature_cols: list[str] | None = None,
) -> dict:
    deployed = _load_model(deployed_path)
    candidate = _load_model(candidate_path)

    # Panel laden + recent window
    panel = (
        pd.read_parquet(panel_path)
        if panel_path.suffix == ".parquet"
        else pd.read_csv(panel_path)
    )
    if timestamp_col in panel.columns:
        panel = panel.sort_values(timestamp_col).reset_index(drop=True)
        recent_cutoff = panel[timestamp_col].max() - pd.Timedelta(days=n_days * 2)
        panel = panel[panel[timestamp_col] >= recent_cutoff]

    panel = panel.dropna(subset=[label])
    logger.info("Recent panel: %d rows (label=%s)", len(panel), label)

    # Features: Prefer Modell-Attribute, falls vorhanden
    feats = feature_cols
    if feats is None:
        for model in (deployed, candidate):
            candidate_feats = getattr(model, "feature_names", None) or getattr(
                model, "feature_cols", None
            )
            if candidate_feats:
                feats = list(candidate_feats)
                break
    if feats is None:
        feats = [
            c
            for c in panel.select_dtypes(include="number").columns
            if c != label
            and not c.startswith("fwd_return")
            and not c.startswith("tb_")
            and c != timestamp_col
        ]
    logger.info("Using %d features", len(feats))

    # Predictions
    dep_preds = _predict(deployed, panel, feats)
    cand_preds = _predict(candidate, panel, feats)
    actuals = panel[label]

    # Metrics
    dep_metrics = _compute_metrics(dep_preds, actuals)
    cand_metrics = _compute_metrics(cand_preds, actuals)

    # DM-Test
    common_idx = dep_preds.index.intersection(cand_preds.index).intersection(
        actuals.index
    )
    dm = _diebold_mariano_test(
        dep_preds.loc[common_idx].values - actuals.loc[common_idx].values,
        cand_preds.loc[common_idx].values - actuals.loc[common_idx].values,
    )

    # Prediction-Korrelation
    corr = 0.0
    if (
        len(common_idx) > 10
        and dep_preds.loc[common_idx].std() > 1e-9
        and cand_preds.loc[common_idx].std() > 1e-9
    ):
        corr = float(
            np.corrcoef(
                dep_preds.loc[common_idx].values, cand_preds.loc[common_idx].values
            )[0, 1]
        )
        if np.isnan(corr):
            corr = 0.0

    decision = _decision(dep_metrics, cand_metrics, dm, corr)

    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "deployed_model_path": str(deployed_path),
        "candidate_model_path": str(candidate_path),
        "label": label,
        "n_days": n_days,
        "n_common_samples": len(common_idx),
        "deployed_metrics": dep_metrics,
        "candidate_metrics": cand_metrics,
        "deltas": {
            "ic": round(cand_metrics["ic"] - dep_metrics["ic"], 4),
            "hit_rate": round(cand_metrics["hit_rate"] - dep_metrics["hit_rate"], 4),
            "sharpe": round(cand_metrics["sharpe"] - dep_metrics["sharpe"], 3),
        },
        "prediction_correlation": round(corr, 4),
        "diebold_mariano": dm,
        "decision": decision,
        "decision_notes": _decision_explanation(decision),
    }


def _decision_explanation(decision: str) -> str:
    return {
        "DEPLOY_CANDIDATE": "Candidate ist signifikant besser (IC +0.02, Sharpe +0.2, DM p < 0.10).",
        "HOLD_CONSIDER_MORE_DATA": "Candidate leicht besser aber nicht signifikant — mehr Daten sammeln.",
        "REJECT_CANDIDATE": "Candidate schlechter als deployed — keine Aktion.",
        "HOLD": "Keine klare Verbesserung — deployed Modell beibehalten.",
    }.get(decision, "")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Model Comparison: Candidate vs Deployed"
    )
    parser.add_argument("--deployed", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--label", default="fwd_return_5d")
    parser.add_argument("--n-days", type=int, default=60)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            f"output/ops/model_comparison_{pd.Timestamp.now().strftime('%Y%m%d')}.json"
        ),
    )
    args = parser.parse_args()

    if not args.deployed.exists():
        logger.error("Deployed-Modell fehlt: %s", args.deployed)
        return 1
    if not args.candidate.exists():
        logger.error("Candidate-Modell fehlt: %s", args.candidate)
        return 1
    if not args.panel.exists():
        logger.error("Panel fehlt: %s", args.panel)
        return 1

    result = compare(
        deployed_path=args.deployed,
        candidate_path=args.candidate,
        panel_path=args.panel,
        label=args.label,
        n_days=args.n_days,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info("=" * 60)
    logger.info("Decision: %s", result["decision"])
    logger.info("%s", result["decision_notes"])
    logger.info("Report: %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
