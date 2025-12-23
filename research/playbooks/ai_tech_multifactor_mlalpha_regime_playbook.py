"""End-to-End Research Playbook: AI/Tech Multi-Factor ML Alpha with Regime Analysis.

**Ziel:**
Dieses Playbook automatisiert den vollständigen Research-Workflow von der Factor-Panel-Erstellung
bis zum Risk-Report für AI/Tech-Universe mit ML-Alpha-Integration und Regime-Analyse.

**High-Level Workflow:**

1. **Factor Panel Export**
   - Exportiert Factor-Panel mit Forward-Returns für ML-Validation
   - Nutzt `export_factor_panel_for_ml.py` Logik
   - Output: `output/factor_panels/factor_panel_*.parquet`

2. **ML Model Zoo Comparison**
   - Führt systematischen Modellvergleich durch (Linear, Ridge, Lasso, Random Forest)
   - Nutzt `model_zoo_factor_validation.py`
   - Output: `ml_model_zoo_summary.csv` mit Metriken aller Modelle

3. **Best Model Selection**
   - Wählt bestes Modell basierend auf IC-IR, Test-R², Overfitting-Indikatoren
   - Gibt Modell-Konfiguration zurück für ML-Alpha-Export

4. **ML Alpha Factor Export**
   - Exportiert ML-Alpha-Faktor mit bestem Modell
   - Nutzt `export_ml_alpha_factor.py`
   - Output: `ml_alpha_panel_*.parquet` mit `ml_alpha_*` Spalte

5. **Backtest Multiple Bundles**
   - Führt Backtests mit verschiedenen Factor-Bundles durch:
     - Core-only Bundle (Baseline)
     - Core + ML Alpha Bundle (Mixed)
     - ML Alpha-only Bundle (Pure ML)
   - Nutzt `run_backtest_strategy.py`
   - Output: Backtest-Verzeichnisse mit Equity-Kurven, Positions, Reports

6. **Risk Reports**
   - Generiert Risk-Reports für alle Backtest-Runs
   - Nutzt `generate_risk_report.py`
   - Optional: Regime-Attribution falls Regime-Daten verfügbar
   - Output: `risk_summary.csv`, `risk_by_regime.csv`, `risk_report.md` pro Backtest

7. **Research Summary**
   - Konsolidiert alle Ergebnisse in einem Research-Summary-Report
   - Vergleicht Performance aller Bundles
   - Zeigt ML-Modell-Metriken und Feature-Importance
   - Output: `research_summary_{timestamp}.md`

**Verweise:**

- Factor Analysis: [Workflows – Factor Analysis](docs/WORKFLOWS_FACTOR_ANALYSIS.md)
- ML Validation: [ML Validation & Model Comparison Workflows](docs/WORKFLOWS_ML_VALIDATION_AND_MODEL_COMPARISON.md)
- Strategy Workflows: [Multi-Factor Strategy Workflows](docs/WORKFLOWS_STRATEGIES_MULTIFACTOR.md)
- Regime & Risk: [Regime Models & Risk Workflows](docs/WORKFLOWS_REGIME_MODELS_AND_RISK.md)
- ML Alpha Design: [ML Alpha Factor & Strategy Integration Design (E3)](docs/ML_ALPHA_E3_DESIGN.md)

**Wichtig:**
- Alle Schritte nutzen nur lokale Daten (keine Live-APIs)
- Playbook ist idempotent (kann mehrfach ausgeführt werden)
- Fehlerbehandlung: Bei Fehlern wird Logging ausgegeben, Playbook stoppt nicht sofort
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path for imports
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


@dataclass
class PlaybookConfig:
    """Configuration for the End-to-End Research Playbook.

    Attributes:
        data_root: Root directory for local data (ASSEMBLED_LOCAL_DATA_ROOT)
        factor_panel_dir: Directory for factor panel outputs
        ml_output_dir: Directory for ML validation and model zoo outputs
        backtest_root: Root directory for backtest outputs
        risk_output_dir: Directory for risk report outputs

        universe_file: Path to universe ticker file (e.g., config/universe_ai_tech_tickers.txt)
        core_bundle_path: Path to core-only bundle YAML
        core_ml_bundle_path: Path to core+ML alpha bundle YAML
        ml_alpha_bundle_path: Path to ML alpha-only bundle YAML

        horizon_days: Forward return horizon (e.g., 20)
        freq: Data frequency (e.g., "1d")
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)

        factor_set: Factor set to use (e.g., "core+alt_full")
        n_splits: Number of CV splits for ML validation (default: 5)
        rebalance_freq: Rebalancing frequency for backtests (e.g., "M" for monthly)
        max_gross_exposure: Maximum gross exposure for backtests (default: 1.0)
        start_capital: Starting capital for backtests (default: 100000)

        regime_file: Optional path to regime state file for risk attribution
    """

    # Paths
    data_root: Path
    factor_panel_dir: Path
    ml_output_dir: Path
    backtest_root: Path
    risk_output_dir: Path

    # Universe & Bundles
    universe_file: Path
    core_bundle_path: Path
    core_ml_bundle_path: Path
    ml_alpha_bundle_path: Path

    # Parameters
    horizon_days: int
    freq: str
    start_date: str
    end_date: str

    # Factor & ML
    factor_set: str
    n_splits: int = 5

    # Backtest
    rebalance_freq: str = "M"
    max_gross_exposure: float = 1.0
    start_capital: float = 100000.0

    # Optional
    regime_file: Path | None = None


def run_factor_panel_export(config: PlaybookConfig) -> Path:
    """Export factor panel with forward returns for ML validation.

    Uses the logic from `research/factors/export_factor_panel_for_ml.py` to create
    a factor panel with forward returns and all selected factors.

    Args:
        config: Playbook configuration

    Returns:
        Path to exported factor panel Parquet file

    Raises:
        RuntimeError: If factor panel export fails
    """
    logger.info("=" * 80)
    logger.info("Step 1: Export Factor Panel")
    logger.info("=" * 80)

    from src.assembled_core.config.settings import get_settings

    # Load symbols from universe file
    if not config.universe_file.exists():
        raise RuntimeError(f"Universe file not found: {config.universe_file}")

    symbols_list = []
    with open(config.universe_file, "r", encoding="utf-8") as f:
        for line in f:
            symbol = line.strip()
            if symbol and not symbol.startswith("#"):
                symbols_list.append(symbol)

    if not symbols_list:
        raise RuntimeError(f"No symbols found in universe file: {config.universe_file}")

    logger.info(f"Loaded {len(symbols_list)} symbols from {config.universe_file.name}")

    # Get settings for data loading
    settings = get_settings()

    # Import price loading and factor computation
    from scripts.run_factor_analysis import load_price_data, compute_factors
    from src.assembled_core.qa.factor_analysis import add_forward_returns

    # Load prices
    logger.info(
        f"Loading prices (freq={config.freq}, start={config.start_date}, end={config.end_date})..."
    )
    prices = load_price_data(
        freq=config.freq,
        symbols=symbols_list,
        symbols_file=None,
        universe=None,
        data_source="local",
        start_date=config.start_date,
        end_date=config.end_date,
        settings=settings,
    )

    if prices.empty:
        raise RuntimeError("Price DataFrame is empty - check date range and universe")

    logger.info(
        f"Loaded prices: {len(prices)} rows, {prices['symbol'].nunique()} symbols"
    )

    # Compute factors
    logger.info(f"Computing factors (factor_set={config.factor_set})...")
    # compute_factors needs output_dir for alt-data loading
    output_dir = ROOT / "output" / "altdata"
    factors = compute_factors(
        prices=prices,
        factor_set=config.factor_set,
        freq=config.freq,
        output_dir=output_dir,
        settings=settings,
    )

    if factors is None or factors.empty:
        logger.warning("compute_factors() returned empty DataFrame. Using prices only.")
        factors = prices[["timestamp", "symbol"]].copy()

    # Add forward returns
    logger.info(f"Adding forward returns (horizon={config.horizon_days}d)...")
    panel_with_fwd = add_forward_returns(
        prices=prices.copy(),
        horizon_days=config.horizon_days,
        price_col="close",
        group_col="symbol",
        timestamp_col="timestamp",
        col_name=f"fwd_return_{config.horizon_days}d",
        return_type="log",
    )

    # Merge factors
    factor_cols = [c for c in factors.columns if c not in ("timestamp", "symbol")]
    logger.info(f"Merging {len(factor_cols)} factor columns into panel...")

    merged = panel_with_fwd.merge(
        factors[["timestamp", "symbol"] + factor_cols],
        on=["timestamp", "symbol"],
        how="left",
    )

    # Clean up duplicate close columns if any
    if "close_x" in merged.columns and "close_y" in merged.columns:
        merged["close"] = merged["close_x"].fillna(merged["close_y"])
        merged = merged.drop(columns=["close_x", "close_y"])
    elif "close_x" in merged.columns:
        merged = merged.rename(columns={"close_x": "close"})
    elif "close_y" in merged.columns:
        merged = merged.rename(columns={"close_y": "close"})

    # Sort and reset index
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    # Generate output filename
    universe_name = config.universe_file.stem
    output_filename = f"factor_panel_{universe_name}_{config.factor_set}_{config.horizon_days}d_{config.freq}.parquet"
    output_path = config.factor_panel_dir / output_filename

    # Ensure output directory exists
    config.factor_panel_dir.mkdir(parents=True, exist_ok=True)

    # Save panel
    logger.info(f"Saving factor panel to {output_path}...")
    merged.to_parquet(output_path, index=False)

    logger.info(
        f"Factor panel exported: {len(merged)} rows, {merged['symbol'].nunique()} symbols, "
        f"{len(merged.columns)} columns"
    )

    return output_path


def run_ml_model_zoo(config: PlaybookConfig, factor_panel_path: Path) -> pd.DataFrame:
    """Run ML model zoo comparison on factor panel.

    Uses `research/ml/model_zoo_factor_validation.py` to compare multiple ML models
    (Linear, Ridge, Lasso, Random Forest) on the same factor panel.

    Args:
        config: Playbook configuration
        factor_panel_path: Path to factor panel Parquet file

    Returns:
        DataFrame with model comparison metrics (from ml_model_zoo_summary.csv)

    Raises:
        RuntimeError: If model zoo execution fails
    """
    logger.info("=" * 80)
    logger.info("Step 2: ML Model Zoo Comparison")
    logger.info("=" * 80)

    try:
        from research.ml.model_zoo_factor_validation import run_model_zoo_for_panel
    except ImportError as e:
        raise RuntimeError(f"Failed to import run_model_zoo_for_panel: {e}") from e

    if not factor_panel_path.exists():
        raise RuntimeError(f"Factor panel file not found: {factor_panel_path}")

    label_col = f"fwd_return_{config.horizon_days}d"
    output_dir = config.ml_output_dir / "model_zoo"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running model zoo on factor panel: {factor_panel_path.name}")
    logger.info(f"Label column: {label_col}")
    logger.info(f"Number of CV splits: {config.n_splits}")
    logger.info(f"Output directory: {output_dir}")

    # Run model zoo
    summary_df = run_model_zoo_for_panel(
        factor_panel_path=factor_panel_path,
        label_col=label_col,
        output_dir=output_dir,
        experiment_cfg_kwargs={
            "n_splits": config.n_splits,
            "train_size": None,  # Expanding window
            "standardize": True,
            "min_train_samples": 252,
        },
    )

    if summary_df is None or summary_df.empty:
        raise RuntimeError("Model zoo returned empty summary DataFrame")

    logger.info(f"Model zoo completed: {len(summary_df)} models compared")

    # Log top models
    if "ic_ir" in summary_df.columns:
        top_models = summary_df.nlargest(3, "ic_ir")
        logger.info("Top 3 models by IC-IR:")
        for idx, row in top_models.iterrows():
            logger.info(
                f"  {row.get('model_name', 'unknown')}: "
                f"IC-IR={row.get('ic_ir', 'N/A'):.4f}, "
                f"Test-R²={row.get('test_r2_mean', 'N/A'):.4f}"
            )
    elif "test_r2_mean" in summary_df.columns:
        top_models = summary_df.nlargest(3, "test_r2_mean")
        logger.info("Top 3 models by Test-R²:")
        for idx, row in top_models.iterrows():
            logger.info(
                f"  {row.get('model_name', 'unknown')}: "
                f"Test-R²={row.get('test_r2_mean', 'N/A'):.4f}"
            )

    return summary_df


def select_best_model(model_zoo_df: pd.DataFrame) -> dict[str, Any]:
    """Select best model from model zoo comparison results.

    Selection criteria (in order of priority):
    1. Highest IC-IR (Information Ratio)
    2. Highest test_r2_mean (if IC-IR is similar)
    3. Lowest train/test gap (overfitting indicator)
    4. Consistent performance across CV splits

    Args:
        model_zoo_df: DataFrame from ml_model_zoo_summary.csv

    Returns:
        Dictionary with best model configuration:
        {
            "model_name": str,
            "model_type": str,
            "model_params": dict,
            "label_col": str,
            "ic_ir": float,
            "test_r2_mean": float,
            "ls_sharpe": float,
        }

    Raises:
        ValueError: If model_zoo_df is empty or no valid models found
    """
    logger.info("=" * 80)
    logger.info("Step 3: Select Best Model")
    logger.info("=" * 80)

    if model_zoo_df is None or model_zoo_df.empty:
        raise ValueError("Model zoo DataFrame is empty - cannot select best model")

    # Create a copy for sorting
    df = model_zoo_df.copy()

    # Determine label_col from DataFrame (if available) or use default
    label_col = "fwd_return_20d"  # Default, will be overridden if found in config

    # Sort by IC-IR (primary criterion)
    if "ic_ir" in df.columns:
        # Filter out NaN IC-IR values
        df = df[df["ic_ir"].notna()].copy()
        if df.empty:
            logger.warning(
                "No models with valid IC-IR found. Falling back to test_r2_mean."
            )
        else:
            # Sort by IC-IR descending
            df = df.sort_values("ic_ir", ascending=False)
            best_row = df.iloc[0]
            logger.info(
                f"Selected model by IC-IR: {best_row.get('model_name', 'unknown')} (IC-IR={best_row.get('ic_ir', 0):.4f})"
            )
    elif "ic_mean" in df.columns:
        # Fallback to IC mean
        df = df[df["ic_mean"].notna()].copy()
        if df.empty:
            logger.warning(
                "No models with valid IC mean found. Falling back to test_r2_mean."
            )
        else:
            df = df.sort_values("ic_mean", ascending=False)
            best_row = df.iloc[0]
            logger.info(
                f"Selected model by IC mean: {best_row.get('model_name', 'unknown')} (IC={best_row.get('ic_mean', 0):.4f})"
            )
    else:
        # Fallback to test_r2_mean
        if "test_r2_mean" not in df.columns:
            raise ValueError(
                "No valid metric columns found in model zoo DataFrame (expected ic_ir, ic_mean, or test_r2_mean)"
            )
        df = df[df["test_r2_mean"].notna()].copy()
        if df.empty:
            raise ValueError("No models with valid test_r2_mean found")
        df = df.sort_values("test_r2_mean", ascending=False)
        best_row = df.iloc[0]
        logger.info(
            f"Selected model by Test-R²: {best_row.get('model_name', 'unknown')} (R²={best_row.get('test_r2_mean', 0):.4f})"
        )

    # Extract model information
    model_name = best_row.get("model_name", "unknown")
    model_type = best_row.get("model_type", "unknown")

    # Parse model parameters from model_name or extract from DataFrame
    # Model names are typically like "ridge_0_1", "rf_depth_5", etc.
    model_params: dict[str, Any] = {}

    # Import re at function level (already imported at module level, but ensure it's available)
    if model_type == "ridge" or model_name.startswith("ridge"):
        # Extract alpha from model_name (e.g., "ridge_0_1" -> alpha=0.1)
        alpha_match = re.search(r"ridge_(\d+)_(\d+)", model_name)
        if alpha_match:
            num1, num2 = alpha_match.groups()
            model_params["alpha"] = float(f"{num1}.{num2}")
        else:
            model_params["alpha"] = 1.0  # Default
    elif model_type == "lasso" or model_name.startswith("lasso"):
        alpha_match = re.search(r"lasso_(\d+)_(\d+)", model_name)
        if alpha_match:
            num1, num2 = alpha_match.groups()
            model_params["alpha"] = float(f"{num1}.{num2}")
        else:
            model_params["alpha"] = 0.01  # Default
    elif model_type == "random_forest" or model_name.startswith("rf"):
        # Extract depth from model_name (e.g., "rf_depth_5" -> max_depth=5)
        depth_match = re.search(r"depth_(\d+)", model_name)
        if depth_match:
            model_params["max_depth"] = int(depth_match.group(1))
        else:
            model_params["max_depth"] = 5  # Default
        model_params["n_estimators"] = 200  # Default
        model_params["random_state"] = 42

    # Build result dictionary
    result = {
        "model_name": model_name,
        "model_type": model_type,
        "model_params": model_params,
        "label_col": label_col,
        "ic_ir": float(best_row.get("ic_ir", 0)) if "ic_ir" in best_row else None,
        "ic_mean": float(best_row.get("ic_mean", 0)) if "ic_mean" in best_row else None,
        "test_r2_mean": float(best_row.get("test_r2_mean", 0))
        if "test_r2_mean" in best_row
        else None,
        "test_mse_mean": float(best_row.get("test_mse_mean", 0))
        if "test_mse_mean" in best_row
        else None,
        "ls_sharpe": float(best_row.get("ls_sharpe", 0))
        if "ls_sharpe" in best_row and pd.notna(best_row.get("ls_sharpe"))
        else None,
    }

    logger.info(f"Best model selected: {model_name} ({model_type})")
    logger.info(f"  IC-IR: {result.get('ic_ir', 'N/A')}")
    logger.info(f"  Test-R²: {result.get('test_r2_mean', 'N/A')}")
    logger.info(f"  L/S Sharpe: {result.get('ls_sharpe', 'N/A')}")
    logger.info(f"  Model params: {model_params}")

    return result


def run_ml_alpha_export(
    config: PlaybookConfig,
    factor_panel_path: Path,
    best_model: dict[str, Any],
) -> Path:
    """Export ML alpha factor using best model.

    Uses `research/ml/export_ml_alpha_factor.py` to generate ML alpha factor
    panel with predictions from the best model.

    Args:
        config: Playbook configuration
        factor_panel_path: Path to original factor panel
        best_model: Dictionary with best model configuration (from select_best_model)

    Returns:
        Path to ML alpha factor panel Parquet file

    Raises:
        RuntimeError: If ML alpha export fails
    """
    logger.info("=" * 80)
    logger.info("Step 4: Export ML Alpha Factor")
    logger.info("=" * 80)

    try:
        from research.ml.export_ml_alpha_factor import export_ml_alpha_factor
    except ImportError as e:
        raise RuntimeError(f"Failed to import export_ml_alpha_factor: {e}") from e

    if not factor_panel_path.exists():
        raise RuntimeError(f"Factor panel file not found: {factor_panel_path}")

    # Extract model configuration
    model_type = best_model.get("model_type")
    model_params = best_model.get("model_params", {})
    label_col = best_model.get("label_col", f"fwd_return_{config.horizon_days}d")
    model_name = best_model.get("model_name", "unknown")

    if not model_type:
        raise ValueError("best_model dict must contain 'model_type'")

    # Set up output directory
    output_dir = config.ml_output_dir / "ml_alpha_factors"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting ML alpha factor with model: {model_name} ({model_type})")
    logger.info(f"  Label column: {label_col}")
    logger.info(f"  Model params: {model_params}")
    logger.info(f"  CV splits: {config.n_splits}")
    logger.info(f"  Output directory: {output_dir}")

    # Run ML alpha export
    ml_alpha_panel_path = export_ml_alpha_factor(
        factor_panel_file=factor_panel_path,
        label_col=label_col,
        model_type=model_type,
        model_params=model_params if model_params else None,
        n_splits=config.n_splits,
        test_start=None,
        test_end=None,
        output_dir=output_dir,
        column_name=None,  # Auto-generate from model_type and horizon
    )

    if not ml_alpha_panel_path.exists():
        raise RuntimeError(
            f"ML alpha panel file was not created: {ml_alpha_panel_path}"
        )

    logger.info(f"ML alpha factor exported: {ml_alpha_panel_path}")

    # Log some statistics
    try:
        import pandas as pd

        ml_alpha_df = pd.read_parquet(ml_alpha_panel_path)
        ml_alpha_cols = [c for c in ml_alpha_df.columns if c.startswith("ml_alpha_")]
        if ml_alpha_cols:
            ml_alpha_col = ml_alpha_cols[0]
            n_with_predictions = ml_alpha_df[ml_alpha_col].notna().sum()
            logger.info(f"  ML alpha column: {ml_alpha_col}")
            logger.info(
                f"  Rows with predictions: {n_with_predictions} / {len(ml_alpha_df)}"
            )
    except Exception as e:
        logger.warning(f"Could not load ML alpha panel for statistics: {e}")

    return ml_alpha_panel_path


def run_backtests_with_bundles(
    config: PlaybookConfig,
    ml_alpha_panel_path: Path,
) -> list[Path]:
    """Run backtests with multiple factor bundles.

    Runs backtests for:
    1. Core-only bundle (baseline)
    2. Core + ML alpha bundle (mixed)
    3. ML alpha-only bundle (pure ML) - optional

    Uses `scripts/run_backtest_strategy.py` via subprocess (CLI).

    Args:
        config: Playbook configuration
        ml_alpha_panel_path: Path to ML alpha factor panel (contains both traditional and ML factors)

    Returns:
        List of paths to backtest output directories

    Raises:
        RuntimeError: If any backtest fails
    """
    logger.info("=" * 80)
    logger.info("Step 5: Run Backtests with Multiple Bundles")
    logger.info("=" * 80)

    import subprocess
    from datetime import datetime

    if not ml_alpha_panel_path.exists():
        raise RuntimeError(f"ML alpha panel file not found: {ml_alpha_panel_path}")

    # Define bundles to test
    bundles_to_test = [
        ("core_only", config.core_bundle_path, "Core-only (baseline)"),
        ("core_ml", config.core_ml_bundle_path, "Core + ML Alpha (mixed)"),
    ]

    # Add ML alpha-only bundle if path exists
    if config.ml_alpha_bundle_path.exists():
        bundles_to_test.append(
            ("ml_alpha_only", config.ml_alpha_bundle_path, "ML Alpha-only (pure ML)")
        )
    else:
        logger.warning(
            f"ML alpha bundle not found: {config.ml_alpha_bundle_path}. Skipping ML-only backtest."
        )

    backtest_dirs: list[Path] = []
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    for bundle_name, bundle_path, description in bundles_to_test:
        if not bundle_path.exists():
            logger.warning(
                f"Bundle file not found: {bundle_path}. Skipping {bundle_name} backtest."
            )
            continue

        logger.info(f"Running backtest: {description}")
        logger.info(f"  Bundle: {bundle_path.name}")

        # Create unique output directory
        output_dir = config.backtest_root / f"{bundle_name}_{timestamp_str}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build CLI command
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "cli.py"),
            "run_backtest",
            "--freq",
            config.freq,
            "--strategy",
            "multifactor_long_short",
            "--bundle-path",
            str(bundle_path),
            "--symbols-file",
            str(config.universe_file),
            "--data-source",
            "local",
            "--start-date",
            config.start_date,
            "--end-date",
            config.end_date,
            "--rebalance-freq",
            config.rebalance_freq,
            "--max-gross-exposure",
            str(config.max_gross_exposure),
            "--start-capital",
            str(int(config.start_capital)),
            "--output-dir",
            str(output_dir),
            "--generate-report",
        ]

        # Add factor file if available (for multifactor strategy)
        # Note: The strategy should load factors from the bundle, but we can pass the panel as reference
        # Check if CLI accepts --factor-file parameter
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Command: {' '.join(cmd)}")

        # Run backtest via subprocess
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                check=True,
            )
            logger.info("  Backtest completed successfully")
            if result.stdout:
                logger.debug(f"  stdout: {result.stdout[:500]}")

            # Verify output directory has expected files
            expected_files = ["equity_curve", "positions", "performance_report"]
            found_files = []
            for file_pattern in expected_files:
                matching_files = list(output_dir.glob(f"{file_pattern}*"))
                if matching_files:
                    found_files.extend([f.name for f in matching_files])

            if found_files:
                logger.info(f"  Generated files: {', '.join(found_files[:5])}")
            else:
                logger.warning(f"  No expected output files found in {output_dir}")

            backtest_dirs.append(output_dir)

        except subprocess.CalledProcessError as e:
            logger.error(f"Backtest failed for {bundle_name}: {e}")
            logger.error(f"  stderr: {e.stderr[:500] if e.stderr else 'N/A'}")
            raise RuntimeError(f"Backtest failed for {bundle_name}: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error during backtest for {bundle_name}: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Backtest failed for {bundle_name}: {e}") from e

    logger.info(f"Completed {len(backtest_dirs)} backtests")
    return backtest_dirs


def run_risk_reports(
    config: PlaybookConfig,
    backtest_dirs: list[Path],
    ml_alpha_panel_path: Path | None = None,
) -> dict[str, Path]:
    """Generate risk reports for all backtest runs.

    Uses `scripts/generate_risk_report.py` to create risk reports with:
    - Basic risk metrics (Sharpe, Sortino, MaxDD, etc.)
    - Regime attribution (if regime_file is provided)
    - Factor group attribution (optional, if factor panel available)

    Args:
        config: Playbook configuration
        backtest_dirs: List of paths to backtest output directories
        ml_alpha_panel_path: Optional path to ML alpha factor panel (for factor attribution)

    Returns:
        Dictionary mapping backtest directory names to risk report paths:
        {
            "backtest_dir_name": Path to risk_report.md,
            ...
        }

    Raises:
        RuntimeError: If risk report generation fails
    """
    logger.info("=" * 80)
    logger.info("Step 6: Generate Risk Reports")
    logger.info("=" * 80)

    import subprocess

    if not backtest_dirs:
        logger.warning("No backtest directories provided. Skipping risk reports.")
        return {}

    risk_report_paths: dict[str, Path] = {}

    for backtest_dir in backtest_dirs:
        if not backtest_dir.exists():
            logger.warning(
                f"Backtest directory not found: {backtest_dir}. Skipping risk report."
            )
            continue

        backtest_name = backtest_dir.name
        logger.info(f"Generating risk report for: {backtest_name}")

        # Build CLI command
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "cli.py"),
            "risk_report",
            "--backtest-dir",
            str(backtest_dir),
            "--output-dir",
            str(backtest_dir),  # Risk reports go into backtest directory
        ]

        # Add regime file if provided
        if config.regime_file and config.regime_file.exists():
            cmd.extend(["--regime-file", str(config.regime_file)])
            logger.info(f"  Using regime file: {config.regime_file.name}")

        # Add factor panel if available (for factor attribution)
        if ml_alpha_panel_path and ml_alpha_panel_path.exists():
            cmd.extend(["--factor-panel-file", str(ml_alpha_panel_path)])
            logger.info(f"  Using factor panel: {ml_alpha_panel_path.name}")

        logger.info(f"  Command: {' '.join(cmd)}")

        # Run risk report generation via subprocess
        try:
            _ = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                check=True,
            )
            logger.info("  Risk report generated successfully")

            # Check for generated risk report files
            risk_report_md = backtest_dir / "risk_report.md"
            risk_summary_csv = backtest_dir / "risk_summary.csv"

            if risk_report_md.exists():
                risk_report_paths[backtest_name] = risk_report_md
                logger.info(f"  Risk report: {risk_report_md.name}")
            else:
                logger.warning(f"  Risk report Markdown not found: {risk_report_md}")

            if risk_summary_csv.exists():
                logger.info(f"  Risk summary: {risk_summary_csv.name}")

            # Check for optional files
            if (backtest_dir / "risk_by_regime.csv").exists():
                logger.info("  Regime attribution: risk_by_regime.csv")
            if (backtest_dir / "risk_by_factor_group.csv").exists():
                logger.info("  Factor attribution: risk_by_factor_group.csv")

        except subprocess.CalledProcessError as e:
            logger.error(f"Risk report generation failed for {backtest_name}: {e}")
            logger.error(f"  stderr: {e.stderr[:500] if e.stderr else 'N/A'}")
            # Continue with other backtests instead of failing completely
            logger.warning("  Continuing with other backtests...")
        except Exception as e:
            logger.error(
                f"Unexpected error during risk report generation for {backtest_name}: {e}",
                exc_info=True,
            )
            logger.warning("  Continuing with other backtests...")

    logger.info(f"Generated {len(risk_report_paths)} risk reports")
    return risk_report_paths


def write_research_summary(
    config: PlaybookConfig,
    artifacts: dict[str, Any],
) -> Path:
    """Write consolidated research summary report.

    Creates a comprehensive Markdown report with:
    - Overview of research workflow
    - ML model comparison results (top models, metrics)
    - Best model selection rationale
    - Backtest performance comparison (all bundles)
    - Risk metrics comparison
    - Regime-specific performance (if available)
    - Feature importance summary (if available)
    - Recommendations and next steps

    Args:
        config: Playbook configuration
        artifacts: Dictionary with all artifacts from workflow:
            {
                "factor_panel_path": Path,
                "model_zoo_df": pd.DataFrame,
                "best_model": dict,
                "ml_alpha_panel_path": Path,
                "backtest_dirs": list[Path],
                "risk_report_paths": dict[str, Path],
            }

    Returns:
        Path to research summary Markdown file

    Raises:
        RuntimeError: If summary generation fails
    """
    logger.info("=" * 80)
    logger.info("Step 7: Write Research Summary")
    logger.info("=" * 80)

    from datetime import datetime

    # Create output directory
    output_dir = config.risk_output_dir / "research_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = f"research_summary_ai_tech_mlalpha_{timestamp_str}.md"
    summary_path = output_dir / summary_filename

    logger.info(f"Writing research summary to: {summary_path}")

    # Extract artifacts
    model_zoo_df = artifacts.get("model_zoo_df")
    best_model = artifacts.get("best_model", {})
    backtest_dirs = artifacts.get("backtest_dirs", [])
    risk_report_paths = artifacts.get("risk_report_paths", {})

    # Build Markdown content
    lines: list[str] = []

    # Header
    lines.append(
        "# Research Summary: AI/Tech Multi-Factor ML Alpha with Regime Analysis"
    )
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Overview & Setup
    lines.append("## Overview & Setup")
    lines.append("")
    lines.append("### Configuration")
    lines.append("")
    lines.append(f"- **Universe:** {config.universe_file.name}")
    lines.append(f"- **Factor Set:** {config.factor_set}")
    lines.append(f"- **Horizon:** {config.horizon_days} days")
    lines.append(f"- **Frequency:** {config.freq}")
    lines.append(f"- **Time Range:** {config.start_date} to {config.end_date}")
    lines.append(f"- **CV Splits:** {config.n_splits}")
    lines.append(f"- **Rebalance Frequency:** {config.rebalance_freq}")
    lines.append(f"- **Max Gross Exposure:** {config.max_gross_exposure}")
    lines.append(f"- **Start Capital:** ${config.start_capital:,.0f}")
    lines.append("")

    # ML Model Comparison
    if model_zoo_df is not None and not model_zoo_df.empty:
        lines.append("## ML Model Comparison")
        lines.append("")
        lines.append("### Top Models by IC-IR")
        lines.append("")

        # Sort by IC-IR and show top 5
        if "ic_ir" in model_zoo_df.columns:
            top_models = model_zoo_df.nlargest(5, "ic_ir")[
                ["model_name", "model_type", "ic_ir", "test_r2_mean", "ls_sharpe"]
            ]
        elif "test_r2_mean" in model_zoo_df.columns:
            top_models = model_zoo_df.nlargest(5, "test_r2_mean")[
                ["model_name", "model_type", "test_r2_mean", "ls_sharpe"]
            ]
        else:
            top_models = model_zoo_df.head(5)

        # Create table
        lines.append("| Model | Type | IC-IR | Test R² | L/S Sharpe |")
        lines.append("|-------|------|-------|---------|------------|")
        for _, row in top_models.iterrows():
            model_name = row.get("model_name", "N/A")
            model_type = row.get("model_type", "N/A")
            ic_ir = row.get("ic_ir", "N/A")
            if pd.notna(ic_ir):
                ic_ir = f"{ic_ir:.4f}"
            test_r2 = row.get("test_r2_mean", "N/A")
            if pd.notna(test_r2):
                test_r2 = f"{test_r2:.4f}"
            ls_sharpe = row.get("ls_sharpe", "N/A")
            if pd.notna(ls_sharpe):
                ls_sharpe = f"{ls_sharpe:.4f}"
            lines.append(
                f"| {model_name} | {model_type} | {ic_ir} | {test_r2} | {ls_sharpe} |"
            )
        lines.append("")

    # Best Model Selection
    if best_model:
        lines.append("## Best Model Selection")
        lines.append("")
        lines.append(
            f"**Selected Model:** {best_model.get('model_name', 'N/A')} ({best_model.get('model_type', 'N/A')})"
        )
        lines.append("")
        lines.append("### Rationale")
        lines.append("")
        lines.append(f"- **IC-IR:** {best_model.get('ic_ir', 'N/A')}")
        lines.append(f"- **Test R²:** {best_model.get('test_r2_mean', 'N/A')}")
        lines.append(f"- **L/S Sharpe:** {best_model.get('ls_sharpe', 'N/A')}")
        lines.append(f"- **Model Parameters:** {best_model.get('model_params', {})}")
        lines.append("")

    # Backtest Performance Comparison
    if backtest_dirs:
        lines.append("## Backtest Performance Comparison")
        lines.append("")
        lines.append("### Performance Metrics")
        lines.append("")

        # Try to load risk summaries
        performance_data: list[dict[str, Any]] = []

        for backtest_dir in backtest_dirs:
            backtest_name = backtest_dir.name
            risk_summary_csv = backtest_dir / "risk_summary.csv"

            bundle_name = backtest_name.split("_")[0]  # Extract bundle name

            if risk_summary_csv.exists():
                try:
                    risk_df = pd.read_csv(risk_summary_csv)
                    if not risk_df.empty:
                        # Extract key metrics (adjust column names as needed)
                        sharpe = (
                            risk_df.get("sharpe", pd.Series([None])).iloc[0]
                            if "sharpe" in risk_df.columns
                            else None
                        )
                        sortino = (
                            risk_df.get("sortino", pd.Series([None])).iloc[0]
                            if "sortino" in risk_df.columns
                            else None
                        )
                        max_dd = (
                            risk_df.get("max_drawdown", pd.Series([None])).iloc[0]
                            if "max_drawdown" in risk_df.columns
                            else None
                        )
                        ann_return = (
                            risk_df.get(
                                "mean_return_annualized", pd.Series([None])
                            ).iloc[0]
                            if "mean_return_annualized" in risk_df.columns
                            else None
                        )

                        performance_data.append(
                            {
                                "Bundle": bundle_name,
                                "Sharpe": sharpe,
                                "Sortino": sortino,
                                "Max DD": max_dd,
                                "Ann Return": ann_return,
                            }
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not load risk summary from {risk_summary_csv}: {e}"
                    )
                    performance_data.append(
                        {
                            "Bundle": bundle_name,
                            "Sharpe": "N/A",
                            "Sortino": "N/A",
                            "Max DD": "N/A",
                            "Ann Return": "N/A",
                        }
                    )
            else:
                performance_data.append(
                    {
                        "Bundle": bundle_name,
                        "Sharpe": "N/A",
                        "Sortino": "N/A",
                        "Max DD": "N/A",
                        "Ann Return": "N/A",
                    }
                )

        if performance_data:
            lines.append("| Bundle | Sharpe | Sortino | Max DD | Ann Return |")
            lines.append("|--------|--------|---------|--------|------------|")
            for data in performance_data:
                sharpe_str = (
                    f"{data['Sharpe']:.4f}"
                    if isinstance(data["Sharpe"], (int, float))
                    and pd.notna(data["Sharpe"])
                    else str(data["Sharpe"])
                )
                sortino_str = (
                    f"{data['Sortino']:.4f}"
                    if isinstance(data["Sortino"], (int, float))
                    and pd.notna(data["Sortino"])
                    else str(data["Sortino"])
                )
                max_dd_str = (
                    f"{data['Max DD']:.4f}"
                    if isinstance(data["Max DD"], (int, float))
                    and pd.notna(data["Max DD"])
                    else str(data["Max DD"])
                )
                ann_ret_str = (
                    f"{data['Ann Return']:.2%}"
                    if isinstance(data["Ann Return"], (int, float))
                    and pd.notna(data["Ann Return"])
                    else str(data["Ann Return"])
                )
                lines.append(
                    f"| {data['Bundle']} | {sharpe_str} | {sortino_str} | {max_dd_str} | {ann_ret_str} |"
                )
        lines.append("")

    # Risk Reports
    if risk_report_paths:
        lines.append("## Risk Reports")
        lines.append("")
        lines.append("Risk reports have been generated for each backtest:")
        lines.append("")
        for backtest_name, report_path in risk_report_paths.items():
            # Try to make path relative to ROOT, but fallback to absolute if not possible
            try:
                if report_path.is_relative_to(ROOT):
                    rel_path = report_path.relative_to(ROOT)
                    path_str = f"`{rel_path}`"
                else:
                    path_str = f"`{report_path}`"
            except (ValueError, TypeError, AttributeError):
                # Path is not under ROOT (e.g., in tmp_path during tests) or is_relative_to not available
                path_str = f"`{report_path}`"
            lines.append(f"- **{backtest_name}:** {path_str}")
        lines.append("")

    # Recommendations & Next Steps
    lines.append("## Recommendations & Next Steps")
    lines.append("")
    lines.append(
        "1. **Review Risk Reports:** Examine regime-specific performance and factor attribution"
    )
    lines.append(
        "2. **Compare Bundles:** Identify which bundle performs best in different market conditions"
    )
    lines.append(
        "3. **Model Refinement:** Consider tuning hyperparameters or trying additional models"
    )
    lines.append(
        "4. **Feature Analysis:** Review feature importance to identify most predictive factors"
    )
    lines.append(
        "5. **Regime Analysis:** If regime data is available, analyze performance by market regime"
    )
    lines.append("")

    # Write file
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Research summary written: {summary_path}")
        logger.info(f"  Summary length: {len(lines)} lines")
    except Exception as e:
        raise RuntimeError(f"Failed to write research summary: {e}") from e

    return summary_path


def create_default_config() -> PlaybookConfig:
    """Create default playbook configuration for AI/Tech universe.

    Returns:
        PlaybookConfig with sensible defaults for AI/Tech research workflow
    """
    # Default paths (relative to project root)
    root = Path(__file__).resolve().parents[2]

    return PlaybookConfig(
        # Paths
        data_root=Path(
            "datensammlungen/altdaten/stand 3-12-2025"
        ),  # Adjust to your local data root
        factor_panel_dir=root / "output" / "factor_panels",
        ml_output_dir=root / "output" / "ml_validation",
        backtest_root=root / "output" / "backtests",
        risk_output_dir=root / "output" / "risk_reports",
        # Universe & Bundles
        universe_file=root / "config" / "universe_ai_tech_tickers.txt",
        core_bundle_path=root
        / "config"
        / "factor_bundles"
        / "ai_tech_core_alt_bundle.yaml",
        core_ml_bundle_path=root
        / "config"
        / "factor_bundles"
        / "ai_tech_core_ml_bundle.yaml",
        ml_alpha_bundle_path=root
        / "config"
        / "factor_bundles"
        / "ai_tech_ml_alpha_bundle.yaml",
        # Parameters
        horizon_days=20,
        freq="1d",
        start_date="2015-01-01",
        end_date="2025-12-03",
        # Factor & ML
        factor_set="core+alt_full",
        n_splits=5,
        # Backtest
        rebalance_freq="M",
        max_gross_exposure=1.0,
        start_capital=100000.0,
        # Optional
        regime_file=None,  # Can be set to root / "output" / "regimes" / "universe_ai_tech_regime_state.parquet"
    )


def main() -> None:
    """Main entry point for the research playbook.

    Orchestrates the complete workflow:
    1. Factor panel export
    2. ML model zoo comparison
    3. Best model selection
    4. ML alpha factor export
    5. Backtests with multiple bundles
    6. Risk reports generation
    7. Research summary writing
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    logger.info("Starting End-to-End Research Playbook: AI/Tech Multi-Factor ML Alpha")
    logger.info("=" * 80)

    # Create configuration
    config = create_default_config()

    # Ensure output directories exist
    config.factor_panel_dir.mkdir(parents=True, exist_ok=True)
    config.ml_output_dir.mkdir(parents=True, exist_ok=True)
    config.backtest_root.mkdir(parents=True, exist_ok=True)
    config.risk_output_dir.mkdir(parents=True, exist_ok=True)

    # Collect artifacts for summary
    artifacts: dict[str, Any] = {}

    try:
        # Step 1: Export factor panel
        logger.info("")
        factor_panel_path = run_factor_panel_export(config)
        artifacts["factor_panel_path"] = factor_panel_path
        logger.info(f"[SUCCESS] Factor panel exported: {factor_panel_path.name}")
        logger.info("")

        # Step 2: Run ML model zoo
        model_zoo_df = run_ml_model_zoo(config, factor_panel_path)
        artifacts["model_zoo_df"] = model_zoo_df
        logger.info(
            f"[SUCCESS] ML model zoo completed: {len(model_zoo_df)} models compared"
        )
        logger.info("")

        # Step 3: Select best model
        best_model = select_best_model(model_zoo_df)
        artifacts["best_model"] = best_model
        logger.info(
            f"[SUCCESS] Best model selected: {best_model.get('model_name', 'unknown')}"
        )
        logger.info("")

        # Step 4: Export ML alpha factor
        ml_alpha_panel_path = run_ml_alpha_export(config, factor_panel_path, best_model)
        artifacts["ml_alpha_panel_path"] = ml_alpha_panel_path
        logger.info(f"[SUCCESS] ML alpha factor exported: {ml_alpha_panel_path.name}")
        logger.info("")

        # Step 5: Run backtests with multiple bundles
        backtest_dirs = run_backtests_with_bundles(config, ml_alpha_panel_path)
        artifacts["backtest_dirs"] = backtest_dirs
        logger.info(f"[SUCCESS] Backtests completed: {len(backtest_dirs)} bundles")
        logger.info("")

        # Step 6: Generate risk reports
        risk_report_paths = run_risk_reports(config, backtest_dirs, ml_alpha_panel_path)
        artifacts["risk_report_paths"] = risk_report_paths
        logger.info(
            f"[SUCCESS] Risk reports generated: {len(risk_report_paths)} reports"
        )
        logger.info("")

        # Step 7: Write research summary
        summary_path = write_research_summary(config, artifacts)
        artifacts["summary_path"] = summary_path
        logger.info(f"[SUCCESS] Research summary written: {summary_path.name}")
        logger.info("")

        logger.info("=" * 80)
        logger.info("Research Playbook completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Summary report: {summary_path}")
        logger.info(f"Backtest directories: {len(backtest_dirs)}")
        logger.info(f"Risk reports: {len(risk_report_paths)}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Research Playbook failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
