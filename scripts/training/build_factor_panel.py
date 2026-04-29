"""Build a complete factor panel (all 30+ factors + forward returns) for ML training.

Usage:
    python scripts/training/build_factor_panel.py
    python scripts/training/build_factor_panel.py --price-dir data/raw/equities_eod/yfinance
    python scripts/training/build_factor_panel.py --horizons 5 10 20 --output output/factor_panels/panel.parquet

Log prefix: [PANEL]

PIT-safety note:
    Forward returns are appended AFTER all feature computation. Features are derived
    only from data at or before each row's timestamp. No look-ahead bias is introduced.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_PREFIX = "[PANEL]"


def _log(level: str, msg: str) -> None:
    getattr(logger, level)(f"{_PREFIX} {msg}")


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------

def load_price_data(price_dir: Path) -> pd.DataFrame:
    """Load and concatenate all parquet files from price_dir.

    Returns a DataFrame with columns: timestamp, symbol, open, high, low, close, volume.
    timestamp is coerced to tz-naive UTC (date-level) so downstream modules
    that do not expect tz-aware series work without modification.
    """
    parquet_files = sorted(price_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {price_dir}")

    _log("info", f"[START] Loading {len(parquet_files)} parquet files from {price_dir}")

    frames: list[pd.DataFrame] = []
    for fp in parquet_files:
        try:
            df = pd.read_parquet(fp)
            frames.append(df)
        except Exception as exc:
            _log("warning", f"[SKIP] {fp.name} -- could not read: {exc}")

    if not frames:
        raise RuntimeError("No parquet files could be loaded.")

    prices = pd.concat(frames, ignore_index=True)
    _log("info", f"[OK] Raw rows loaded: {len(prices):,} across {prices['symbol'].nunique()} symbols")

    # Normalise timestamp: coerce to tz-naive datetime (keep date precision)
    if "timestamp" in prices.columns:
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True).dt.tz_localize(None)
    else:
        raise KeyError("Expected 'timestamp' column not found in price data.")

    # Sort for deterministic downstream behaviour
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return prices


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _safe_merge(
    base: pd.DataFrame,
    result: pd.DataFrame,
    module_name: str,
    on: list[str] | None = None,
) -> pd.DataFrame:
    """Left-merge result into base on ['timestamp', 'symbol'], ignoring duplicates."""
    merge_keys = on or ["timestamp", "symbol"]
    # Drop columns already present in base (except merge keys) to avoid collisions
    existing = set(base.columns) - set(merge_keys)
    new_cols = [c for c in result.columns if c not in existing or c in merge_keys]
    result_trimmed = result[new_cols].copy()
    try:
        merged = base.merge(result_trimmed, on=merge_keys, how="left")
        added = set(merged.columns) - set(base.columns)
        _log("info", f"[OK] {module_name}: merged {len(added)} new column(s): {sorted(added)}")
        return merged
    except Exception as exc:
        _log("warning", f"[WARN] {module_name}: merge failed -- {exc}; skipping")
        return base


def _try_compute_ta_features(prices: pd.DataFrame) -> pd.DataFrame | None:
    """src/assembled_core/features/ta_features -> add_all_features.

    ta_features works IN-PLACE (adds columns to the input DF). We return
    the enriched DF directly; the caller must replace (not merge) the panel.
    We tag the result with _INPLACE = True so build_full_factor_panel knows.
    """
    try:
        from src.assembled_core.features.ta_features import add_all_features
        result = add_all_features(prices.copy())
        result._inplace_result = True  # type: ignore[attr-defined]
        return result
    except ImportError as exc:
        _log("warning", f"[SKIP] ta_features import failed: {exc}")
        return None
    except Exception as exc:
        _log("warning", f"[SKIP] ta_features compute failed: {exc}")
        return None


def _try_compute_volatility_features(prices: pd.DataFrame) -> pd.DataFrame | None:
    """src/assembled_core/features/volatility_features -> compute_garch_features.

    NOTE: GARCH fitting for 90+ tickers takes 10-30 min (arch library holds GIL).
    Skipped in panel build; use pre-computed GARCH features or add separately.
    """
    _log("warning", "[SKIP] volatility_features: GARCH too slow for panel build (93 tickers)")
    return None


def _try_compute_correlation_features(prices: pd.DataFrame) -> pd.DataFrame | None:
    """src/assembled_core/features/correlation_features -> build_correlation_features_panel"""
    try:
        from src.assembled_core.features.correlation_features import (
            build_correlation_features_panel,
        )
        result = build_correlation_features_panel(prices.copy())
        return result
    except ImportError as exc:
        _log("warning", f"[SKIP] correlation_features import failed: {exc}")
        return None
    except Exception as exc:
        _log("warning", f"[SKIP] correlation_features compute failed: {exc}")
        return None


def _try_compute_macro_features(prices: pd.DataFrame) -> pd.DataFrame | None:
    """src/assembled_core/features/macro_features -> add_latest_macro_value.

    macro_features is a panel-join helper; if it cannot connect to external
    data it raises -- we catch and skip gracefully.
    """
    try:
        from src.assembled_core.features.macro_features import add_latest_macro_value
        result = add_latest_macro_value(prices.copy())
        return result
    except ImportError as exc:
        _log("warning", f"[SKIP] macro_features import failed: {exc}")
        return None
    except Exception as exc:
        _log("warning", f"[SKIP] macro_features compute failed: {exc}")
        return None


def _try_compute_mean_reversion_factors(prices: pd.DataFrame) -> pd.DataFrame | None:
    """src/assembled_core/features/mean_reversion_factors -> compute_mean_reversion_factors"""
    try:
        from src.assembled_core.features.mean_reversion_factors import (
            compute_mean_reversion_factors,
        )
        result = compute_mean_reversion_factors(prices.copy())
        return result
    except ImportError as exc:
        _log("warning", f"[SKIP] mean_reversion_factors import failed: {exc}")
        return None
    except Exception as exc:
        _log("warning", f"[SKIP] mean_reversion_factors compute failed: {exc}")
        return None


def _try_compute_liquidity_vol_factors(prices: pd.DataFrame) -> pd.DataFrame | None:
    """src/assembled_core/features/ta_liquidity_vol_factors.

    Calls multiple add_* helpers sequentially; returns the enriched frame.
    Works IN-PLACE -- tagged so caller replaces panel rather than merging.
    """
    try:
        from src.assembled_core.features.ta_liquidity_vol_factors import (
            add_amihud_illiquidity,
            add_realized_volatility,
            add_turnover_and_liquidity_proxies,
            add_vol_of_vol,
        )
        df = prices.copy()
        df = add_realized_volatility(df)
        df = add_vol_of_vol(df)
        df = add_turnover_and_liquidity_proxies(df)
        df = add_amihud_illiquidity(df)
        df._inplace_result = True  # type: ignore[attr-defined]
        return df
    except ImportError as exc:
        _log("warning", f"[SKIP] ta_liquidity_vol_factors import failed: {exc}")
        return None
    except Exception as exc:
        _log("warning", f"[SKIP] ta_liquidity_vol_factors compute failed: {exc}")
        return None


def _try_compute_market_breadth(prices: pd.DataFrame) -> pd.DataFrame | None:
    """src/assembled_core/features/market_breadth -> compute_market_breadth_ma.

    Returns a date-level frame (one row per date, not per symbol).
    The merge in build_full_factor_panel uses a date-level left join.
    """
    try:
        from src.assembled_core.features.market_breadth import compute_market_breadth_ma
        result = compute_market_breadth_ma(prices.copy())
        return result
    except ImportError as exc:
        _log("warning", f"[SKIP] market_breadth import failed: {exc}")
        return None
    except Exception as exc:
        _log("warning", f"[SKIP] market_breadth compute failed: {exc}")
        return None


def _try_compute_geopolitical_features(prices: pd.DataFrame) -> pd.DataFrame | None:
    """src/assembled_core/features/geopolitical_features -> compute_gpr_proxy"""
    try:
        from src.assembled_core.features.geopolitical_features import compute_gpr_proxy
        result = compute_gpr_proxy(prices.copy())
        return result
    except ImportError as exc:
        _log("warning", f"[SKIP] geopolitical_features import failed: {exc}")
        return None
    except Exception as exc:
        _log("warning", f"[SKIP] geopolitical_features compute failed: {exc}")
        return None


def _try_compute_altdata_factors(prices: pd.DataFrame) -> pd.DataFrame | None:
    """src/assembled_core/features/altdata_earnings_insider_factors -> build_earnings_surprise_factors"""
    try:
        from src.assembled_core.features.altdata_earnings_insider_factors import (
            build_earnings_surprise_factors,
        )
        result = build_earnings_surprise_factors(prices.copy())
        return result
    except ImportError as exc:
        _log("warning", f"[SKIP] altdata_earnings_insider_factors import failed: {exc}")
        return None
    except Exception as exc:
        _log("warning", f"[SKIP] altdata_earnings_insider_factors compute failed: {exc}")
        return None


def _try_compute_supply_chain_features(prices: pd.DataFrame) -> pd.DataFrame | None:
    """src/assembled_core/features/supply_chain_features -> build_supply_chain_features"""
    try:
        from src.assembled_core.features.supply_chain_features import (
            build_supply_chain_features,
        )
        result = build_supply_chain_features(prices.copy())
        return result
    except ImportError as exc:
        _log("warning", f"[SKIP] supply_chain_features import failed: {exc}")
        return None
    except Exception as exc:
        _log("warning", f"[SKIP] supply_chain_features compute failed: {exc}")
        return None


def _try_compute_intermarket_factors(prices: pd.DataFrame) -> pd.DataFrame | None:
    """src/assembled_core/features/intermarket_factors -> build_intermarket_factors + align."""
    try:
        import concurrent.futures

        from src.assembled_core.features.intermarket_factors import (
            align_intermarket_factors_to_panel,
            build_intermarket_factors,
        )
        date_min = prices["timestamp"].min().strftime("%Y-%m-%d")
        date_max = prices["timestamp"].max().strftime("%Y-%m-%d")
        def _run():
            im_df = build_intermarket_factors(start_date=date_min, end_date=date_max)
            return align_intermarket_factors_to_panel(prices.copy(), im_df)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_run)
            result = fut.result(timeout=120)
        return result
    except concurrent.futures.TimeoutError:
        _log("warning", "[SKIP] intermarket_factors: timed out after 120s")
        return None
    except ImportError as exc:
        _log("warning", f"[SKIP] intermarket_factors import failed: {exc}")
        return None
    except Exception as exc:
        _log("warning", f"[SKIP] intermarket_factors compute failed: {exc}")
        return None


def _try_compute_congress_features(prices: pd.DataFrame) -> pd.DataFrame | None:
    """src/assembled_core/features/congress_features -> add_congress_features"""
    try:
        from src.assembled_core.features.congress_features import add_congress_features
        result = add_congress_features(prices.copy())
        return result
    except ImportError as exc:
        _log("warning", f"[SKIP] congress_features import failed: {exc}")
        return None
    except Exception as exc:
        _log("warning", f"[SKIP] congress_features compute failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Forward returns (PIT-safe -- must be called AFTER all features)
# ---------------------------------------------------------------------------

def _add_forward_returns(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Append simple forward returns for each horizon.

    Column naming: fwd_return_5d, fwd_return_10d, fwd_return_20d, etc.
    Uses simple returns: price[t+N] / price[t] - 1.

    CRITICAL: This function is called AFTER all feature computation so that
    forward-price information cannot leak into feature columns (PIT-safe).
    """
    try:
        from src.assembled_core.qa.factor_analysis import add_forward_returns

        # add_forward_returns with a list produces fwd_ret_N columns; we want fwd_return_Nd.
        # Call once per horizon to get the canonical naming.
        result = panel.copy()
        for h in horizons:
            col_name = f"fwd_return_{h}d"
            result = add_forward_returns(
                result,
                horizon_days=h,
                price_col="close",
                group_col="symbol",
                timestamp_col="timestamp",
                col_name=col_name,
                return_type="simple",
            )
            _log("info", f"[OK] Forward return added: {col_name}")
        return result

    except Exception as exc:
        _log("warning", f"[WARN] add_forward_returns failed ({exc}); computing inline fallback")
        result = panel.copy()
        result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        for h in horizons:
            col_name = f"fwd_return_{h}d"
            fwd_price = result.groupby("symbol", group_keys=False)["close"].shift(-h)
            result[col_name] = fwd_price / result["close"] - 1.0
            # PIT guard: last h rows per symbol must be NaN (no future data)
            result[col_name] = result.groupby("symbol", group_keys=False)[col_name].transform(
                lambda s: s.where(pd.Series(range(len(s)), index=s.index) < len(s) - h)
            )
            _log("info", f"[OK] Forward return added (fallback): {col_name}")
        return result


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_full_factor_panel(
    price_dir: Path = Path("data/raw/equities_eod/yfinance"),
    horizons: list[int] = [1, 5, 10, 20],
    output_path: Path = Path("output/factor_panels/full_panel_7y.parquet"),
    use_registry: bool = True,
    triple_barrier: bool = False,
    tb_upper_mult: float = 2.0,
    tb_lower_mult: float = 2.0,
) -> pd.DataFrame:
    """Build a complete factor panel with all available features and forward returns.

    Steps:
        1. Load all parquet price files from price_dir.
        2. Attempt to compute each feature module; skip on any failure.
        3. Merge successful features into a unified panel.
        4. Append forward returns (AFTER features -- PIT-safe).
        5. Save panel as parquet and write a summary JSON sidecar.

    Args:
        price_dir:   Directory containing per-symbol parquet files.
        horizons:    Forward return horizons in trading days.
        output_path: Destination parquet path.
        use_registry: Placeholder for future registry-driven feature selection.

    Returns:
        The completed factor panel DataFrame.
    """
    t_start = time.time()
    _log("info", f"[START] build_full_factor_panel | price_dir={price_dir} | horizons={horizons}")

    # ------------------------------------------------------------------
    # 1. Load prices
    # ------------------------------------------------------------------
    prices = load_price_data(price_dir)
    _log("info", f"[OK] Prices loaded: {prices.shape} | date range: "
                 f"{prices['timestamp'].min().date()} - {prices['timestamp'].max().date()}")

    # ------------------------------------------------------------------
    # 2. Feature modules: try each, collect results
    # ------------------------------------------------------------------
    MODULE_JOBS: list[tuple[str, object]] = [
        ("ta_features",                      _try_compute_ta_features),
        ("volatility_features",              _try_compute_volatility_features),
        ("correlation_features",             _try_compute_correlation_features),
        ("macro_features",                   _try_compute_macro_features),
        ("mean_reversion_factors",           _try_compute_mean_reversion_factors),
        ("ta_liquidity_vol_factors",         _try_compute_liquidity_vol_factors),
        ("market_breadth",                   _try_compute_market_breadth),
        ("geopolitical_features",            _try_compute_geopolitical_features),
        ("altdata_earnings_insider_factors", _try_compute_altdata_factors),
        ("supply_chain_features",            _try_compute_supply_chain_features),
        ("intermarket_factors",              _try_compute_intermarket_factors),
        ("congress_features",               _try_compute_congress_features),
    ]

    # Base panel starts as prices (timestamp + symbol + ohlcv)
    panel = prices.copy()
    base_cols = set(prices.columns)
    succeeded_modules: list[str] = []
    failed_modules: list[str] = []

    for module_name, compute_fn in MODULE_JOBS:
        _log("info", f"[START] Computing {module_name}...")
        try:
            # In-place modules receive the current panel; merge modules receive original prices
            is_inplace_module = module_name in ("ta_features", "ta_liquidity_vol_factors")
            result = compute_fn(panel if is_inplace_module else prices)
        except Exception as exc:
            _log("warning", f"[SKIP] {module_name}: unexpected error -- {exc}")
            failed_modules.append(module_name)
            continue

        if result is None:
            failed_modules.append(module_name)
            continue

        if not isinstance(result, pd.DataFrame) or result.empty:
            _log("warning", f"[SKIP] {module_name}: returned empty or non-DataFrame result")
            failed_modules.append(module_name)
            continue

        # In-place modules return the full enriched panel -- replace directly
        if getattr(result, "_inplace_result", False):
            new_cols = set(result.columns) - set(panel.columns)
            panel = result
            _log("info", f"[OK] {module_name}: in-place, added {len(new_cols)} column(s): {sorted(new_cols)}")
            succeeded_modules.append(module_name)
            continue

        # Determine merge keys based on what columns the result has
        has_symbol = "symbol" in result.columns
        has_timestamp = "timestamp" in result.columns

        if has_symbol and has_timestamp:
            merge_keys = ["timestamp", "symbol"]
        elif has_timestamp and not has_symbol:
            merge_keys = ["timestamp"]
        else:
            _log("warning", f"[SKIP] {module_name}: result missing 'timestamp' column; cannot merge")
            failed_modules.append(module_name)
            continue

        panel = _safe_merge(panel, result, module_name, on=merge_keys)
        succeeded_modules.append(module_name)

    _log("info", f"[OK] Feature modules succeeded: {succeeded_modules}")
    _log("info", f"[OK] Feature modules skipped/failed: {failed_modules}")

    feature_cols = [c for c in panel.columns if c not in base_cols]
    _log("info", f"[OK] Factor columns before forward returns: {len(feature_cols)}")

    # ------------------------------------------------------------------
    # 3. Forward returns -- MUST happen AFTER all feature computation
    # ------------------------------------------------------------------
    _log("info", "[START] Adding forward returns (PIT-safe, post-feature step)...")
    panel = _add_forward_returns(panel, horizons)
    fwd_cols = [f"fwd_return_{h}d" for h in horizons]

    # ------------------------------------------------------------------
    # 3.5. Triple-Barrier Labels (optional)
    # ------------------------------------------------------------------
    if triple_barrier:
        try:
            from src.assembled_core.ml.triple_barrier import build_triple_barrier_labels
            _log("info", f"[START] Adding triple-barrier labels (upper={tb_upper_mult}, lower={tb_lower_mult})...")
            for h in horizons:
                panel = build_triple_barrier_labels(
                    panel,
                    price_col="close",
                    horizon_days=h,
                    upper_mult=tb_upper_mult,
                    lower_mult=tb_lower_mult,
                )
            _log("info", "[OK] Triple-barrier labels added")
        except Exception as exc:
            _log("warning", f"[WARN] Triple-barrier failed: {exc}")

    # ------------------------------------------------------------------
    # 4. Final column ordering:
    #    date | symbol | <factors> | fwd_return_5d | fwd_return_10d | fwd_return_20d
    # ------------------------------------------------------------------
    non_fwd_cols = [c for c in panel.columns if c not in fwd_cols]
    panel = panel[non_fwd_cols + [c for c in fwd_cols if c in panel.columns]]

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panel.to_parquet(output_path, index=False)
    _log("info", f"[OK] Panel saved -> {output_path}  shape={panel.shape}")

    # ------------------------------------------------------------------
    # 6. Summary JSON sidecar
    # ------------------------------------------------------------------
    all_factor_cols = [c for c in panel.columns if c not in {"timestamp", "symbol", "open", "high", "low", "close", "volume"}]
    nan_rates: dict[str, float] = {}
    for col in all_factor_cols:
        nan_rates[col] = round(float(panel[col].isna().mean()), 4)

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "price_dir": str(price_dir),
        "output_path": str(output_path),
        "shape": {"rows": int(panel.shape[0]), "cols": int(panel.shape[1])},
        "symbols": int(panel["symbol"].nunique()),
        "date_range": {
            "start": str(panel["timestamp"].min().date()),
            "end": str(panel["timestamp"].max().date()),
        },
        "horizons_days": horizons,
        "forward_return_cols": fwd_cols,
        "factor_count": len(all_factor_cols),
        "factor_names": all_factor_cols,
        "nan_rates": nan_rates,
        "modules_succeeded": succeeded_modules,
        "modules_failed": failed_modules,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    _log("info", f"[OK] Summary JSON -> {summary_path}")

    _log(
        "info",
        f"[DONE] Panel complete | rows={panel.shape[0]:,} | factors={len(all_factor_cols)} | "
        f"elapsed={summary['elapsed_seconds']}s",
    )
    return panel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a complete ML factor panel (all 30+ factors + forward returns).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--price-dir",
        type=Path,
        default=Path("data/raw/equities_eod/yfinance"),
        help="Directory containing per-symbol EOD parquet files.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        metavar="N",
        help="Forward return horizons in trading days (default: 1 5 10 20).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Nur 5d-Horizon für schnelle Iteration (überschreibt --horizons).",
    )
    parser.add_argument(
        "--triple-barrier",
        action="store_true",
        help="Fügt tb_label_Nd / tb_ret_Nd / tb_barrier_Nd Spalten hinzu (Lopez de Prado).",
    )
    parser.add_argument(
        "--tb-upper-mult",
        type=float,
        default=2.0,
        help="Triple-Barrier Upper-Multiplikator × σ (default: 2.0).",
    )
    parser.add_argument(
        "--tb-lower-mult",
        type=float,
        default=2.0,
        help="Triple-Barrier Lower-Multiplikator × σ (default: 2.0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/factor_panels/full_panel_7y.parquet"),
        help="Destination parquet path for the factor panel.",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        default=False,
        help="Disable registry-driven feature selection (currently informational only).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity level.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.fast:
        args.horizons = [5]
        logging.getLogger().info("[Panel] --fast Mode: nur 5d Horizon")

    # Resolve relative paths relative to repo root (two levels up from this file)
    _repo_root = Path(__file__).resolve().parents[2]
    price_dir = args.price_dir if args.price_dir.is_absolute() else _repo_root / args.price_dir
    output_path = args.output if args.output.is_absolute() else _repo_root / args.output

    # Ensure src is importable
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    panel = build_full_factor_panel(
        price_dir=price_dir,
        horizons=args.horizons,
        output_path=output_path,
        use_registry=not args.no_registry,
        triple_barrier=args.triple_barrier,
        tb_upper_mult=args.tb_upper_mult,
        tb_lower_mult=args.tb_lower_mult,
    )

    print(f"\nFactor panel shape: {panel.shape}")
    print(f"Columns ({len(panel.columns)}): {list(panel.columns)[:10]} ... (first 10 shown)")
