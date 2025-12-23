# research/factors/export_factor_panel_for_ml.py
"""Export Factor Panels for ML Validation Experiments.

This script exports factor panels (with forward returns and factor columns) 
for use in ML validation workflows. It reuses existing factor computation logic
from run_factor_analysis.py and saves panels as Parquet files.

Usage:
    python research/factors/export_factor_panel_for_ml.py \
      --freq 1d \
      --universe config/macro_world_etfs_tickers.txt \
      --factor-set core \
      --horizon-days 20 \
      --start-date 2010-01-01 \
      --end-date 2025-12-03
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.factor_analysis import add_forward_returns

# Reuse existing logic from run_factor_analysis
# We want NO price APIs here, only the existing price loading from run_factor_analysis
# that works with data_source="local" by default
from scripts.run_factor_analysis import (
    load_price_data,
    compute_factors,
    list_available_factor_sets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exportiert ein Factor-Panel für ML-Experimente "
            "(Forward-Returns + Faktor-Spalten) als Parquet-Datei.\n"
            "Hinweis: Preise werden über data_source='local' geladen, "
            "es werden KEINE Live-Preis-APIs aufgerufen."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--freq",
        required=True,
        choices=["1d", "5min"],
        help="Datenfrequenz, z.B. '1d'. Muss zu deinen lokalen Snapshots passen.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--symbols",
        nargs="+",
        help="Liste von Symbolen (manuell).",
    )
    group.add_argument(
        "--symbols-file",
        type=Path,
        help="Pfad zu einer Textdatei mit Symbolen (ein Symbol pro Zeile).",
    )
    group.add_argument(
        "--universe",
        type=Path,
        help="Pfad zu einer Universe-Datei (z.B. config/macro_world_etfs_tickers.txt).",
    )

    parser.add_argument(
        "--data-source",
        default="local",
        choices=["local", "yahoo", "finnhub", "twelve_data"],
        help=(
            "Data Source für Preise. Standard: 'local'. "
            "Andere Provider werden nicht empfohlen."
        ),
    )

    parser.add_argument(
        "--factor-set",
        required=True,
        choices=list_available_factor_sets(),
        help="Welches Factor-Set berechnet werden soll (Single Source of Truth aus run_factor_analysis).",
    )

    parser.add_argument(
        "--horizon-days",
        type=int,
        default=20,
        help="Forward-Return-Horizont in Tagen, z.B. 20.",
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Startdatum (YYYY-MM-DD) für den Datenbereich.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Enddatum (YYYY-MM-DD) für den Datenbereich.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Optional: expliziter Pfad zur Output-Parquet-Datei. "
            "Wenn nicht gesetzt, wird automatisch unter "
            "output/factor_panels/ factor_panel_*.parquet gespeichert."
        ),
    )

    return parser.parse_args()


def _derive_universe_tag(args: argparse.Namespace) -> str:
    """Derive a short tag for the universe from arguments."""
    if args.universe:
        return Path(args.universe).stem
    if args.symbols_file:
        return Path(args.symbols_file).stem
    if args.symbols:
        # Short tag if only few symbols
        if len(args.symbols) <= 3:
            return "custom_" + "_".join(args.symbols)
        return f"custom_{len(args.symbols)}_symbols"
    return "unknown_universe"


def _make_output_path(args: argparse.Namespace) -> Path:
    """Generate output path for factor panel file."""
    if args.output_file:
        return Path(args.output_file)

    universe_tag = _derive_universe_tag(args)
    out_dir = ROOT / "output" / "factor_panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = (
        f"factor_panel_{universe_tag}_{args.factor_set}_"
        f"{args.horizon_days}d_{args.freq}.parquet"
    )
    return out_dir / fname


def _merge_factors_with_forward_returns(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Erzeugt ein Panel mit Forward-Returns + Faktor-Spalten.

    Strategie:
    - Forward-Returns werden aus dem PRICE-Panel berechnet
    - Faktoren werden über (timestamp, symbol) zugejoint
    - Kein aggressives Dropna: Cleaning passiert später in der ML-Pipeline

    Args:
        prices: Price DataFrame with timestamp, symbol, close columns
        factors: Factor DataFrame with timestamp, symbol, and factor columns
        horizon_days: Forward return horizon in days

    Returns:
        Merged DataFrame with forward returns and factor columns
    """
    if not {"timestamp", "symbol", "close"}.issubset(prices.columns):
        missing = {"timestamp", "symbol", "close"} - set(prices.columns)
        raise ValueError(f"prices DataFrame fehlt Spalten: {missing}")

    logger.info("Berechne Forward-Returns für horizon_days=%s ...", horizon_days)

    panel_with_fwd = add_forward_returns(
        prices.copy(),
        horizon_days=horizon_days,
        price_col="close",
        group_col="symbol",
        timestamp_col="timestamp",
        col_name=f"fwd_return_{horizon_days}d",
        return_type="log",
    )

    # Nur wirklich benötigte Spalten aus factors mitnehmen
    factor_cols: list[str] = [
        c for c in factors.columns if c not in ("timestamp", "symbol")
    ]

    if not factor_cols:
        logger.warning(
            "Im factors-DataFrame wurden keine Faktor-Spalten gefunden "
            "(nur timestamp/symbol). Panel enthält nur Forward-Returns."
        )
        return panel_with_fwd

    logger.info("Mergen von %d Faktor-Spalten in das Panel ...", len(factor_cols))

    # Merge auf timestamp/symbol, linke Seite = Forward-Return-Panel
    merged = panel_with_fwd.merge(
        factors[["timestamp", "symbol"] + factor_cols],
        on=["timestamp", "symbol"],
        how="left",
    )

    # Clean up duplicate 'close' columns from merge (close_x, close_y)
    # Keep the first one (from panel_with_fwd) and rename to 'close'
    if "close_x" in merged.columns and "close_y" in merged.columns:
        # Keep close_x (from panel_with_fwd, which has forward returns)
        merged["close"] = merged["close_x"]
        merged = merged.drop(columns=["close_x", "close_y"])
        logger.info(
            "Removed duplicate close columns (close_x, close_y) → kept as 'close'"
        )
    elif "close_x" in merged.columns:
        # Only close_x exists, rename to close
        merged = merged.rename(columns={"close_x": "close"})
        logger.info("Renamed 'close_x' to 'close'")
    elif "close_y" in merged.columns:
        # Only close_y exists, rename to close
        merged = merged.rename(columns={"close_y": "close"})
        logger.info("Renamed 'close_y' to 'close'")
    # If neither close_x nor close_y exists, assume 'close' already exists or is not needed

    # Sortierung für spätere Nutzung (optional, aber nice-to-have)
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return merged


def export_factor_panel_for_ml(args: argparse.Namespace) -> Path:
    """Main export function."""
    if args.data_source != "local":
        logger.warning(
            "data_source='%s' gesetzt. Empfohlen ist 'local', um keine Live-Preis-APIs "
            "zu verwenden.",
            args.data_source,
        )

    logger.info(
        "Lade Preise (data_source=%s, freq=%s) ...", args.data_source, args.freq
    )

    prices = load_price_data(
        freq=args.freq,
        symbols=args.symbols,
        symbols_file=args.symbols_file,
        universe=args.universe,
        data_source=args.data_source,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if prices.empty:
        raise RuntimeError(
            "Price-DataFrame ist leer – bitte Datumsbereich/Universe prüfen."
        )

    logger.info("Berechne Faktoren (factor_set=%s) ...", args.factor_set)

    # compute_factors needs output_dir for alt-data loading
    # Use default output/altdata directory
    output_dir = ROOT / "output" / "altdata"

    factors = compute_factors(
        prices=prices,
        factor_set=args.factor_set,
        output_dir=output_dir,
    )

    if factors is None or factors.empty:
        logger.warning(
            "compute_factors() hat ein leeres DataFrame geliefert. "
            "Exportiere nur Forward-Returns ohne zusätzliche Faktor-Spalten."
        )
        factors = prices[["timestamp", "symbol"]].copy()

    panel = _merge_factors_with_forward_returns(
        prices=prices,
        factors=factors,
        horizon_days=args.horizon_days,
    )

    out_path = _make_output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Speichere Factor-Panel nach %s ...", out_path)
    panel.to_parquet(out_path, index=False)

    logger.info(
        "Fertig. Factor-Panel: %d Zeilen, %d Spalten. Spaltenbeispiele: %s",
        len(panel),
        panel.shape[1],
        ", ".join(list(panel.columns[:10])),
    )

    return out_path


def main() -> int:
    """Main entry point."""
    try:
        args = _parse_args()
        out_path = export_factor_panel_for_ml(args)
        print(f"\nFactor panel exported successfully: {out_path}")
        return 0
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
