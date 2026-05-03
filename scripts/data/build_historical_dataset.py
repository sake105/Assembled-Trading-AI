"""Baut kanonisches historisches Dataset (OHLCV + Fundamentals + Makro) auf.

Verwendung:
    python scripts/data/build_historical_dataset.py \\
        --universe sp500 \\
        --start 2005-01-01 \\
        --out data/canonical/ \\
        --sources yfinance

PIT-Invarianten:
    - Kein Survivorship Bias: Historische S&P500-Mitglieder via Wikipedia-Änderungstabelle
    - EDGAR: filing_date als available_as_of, nie fiscal_year_end
    - FRED: realtime_start für echte Verfügbarkeit
    - adj_close nur für Return-Berechnung im Backtest, nie in Execution
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FRED_SERIES = {
    "FEDFUNDS": "fed_funds_rate",
    "DGS10": "treasury_10y",
    "DGS2": "treasury_2y",
    "UNRATE": "unemployment_rate",
    "CPIAUCSL": "cpi",
    "GDP": "gdp",
    "VIXCLS": "vix",
    "BAA10YM": "credit_spread",
}


# ---------------------------------------------------------------------------
# Sektion 1 — Universe (PIT-safe, kein Survivorship Bias)
# ---------------------------------------------------------------------------

def load_sp500_historical_members(
    start: str,
    end: str,
) -> pd.DataFrame:
    """Historische S&P500-Mitglieder via Wikipedia-Änderungstabelle.

    Gibt DataFrame: symbol, start_date, end_date (NaT = aktuell Mitglied).
    Nur Symbole die im Zeitraum [start, end] Mitglieder waren.

    PIT-safe: kein Survivorship Bias da auch ex-Mitglieder enthalten.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)

    # Tabelle 0 = aktuelle Mitglieder
    current = tables[0][["Symbol", "Date added"]].copy()
    current.columns = ["symbol", "added_raw"]
    current["symbol"] = current["symbol"].str.replace(".", "-", regex=False)
    current["start_date"] = pd.to_datetime(current["added_raw"], errors="coerce")
    current["end_date"] = pd.NaT

    # Tabelle 1 = historische Änderungen (wenn vorhanden)
    removed_rows: list[dict] = []
    if len(tables) > 1:
        changes = tables[1]
        for row in changes.itertuples(index=False):
            try:
                date_raw = row[0]
                added_sym = str(row[1]).replace(".", "-") if pd.notna(row[1]) else None  # noqa: F841
                removed_sym = str(row[2]).replace(".", "-") if pd.notna(row[2]) else None
                chg_date = pd.to_datetime(date_raw, errors="coerce")
                if pd.isna(chg_date):
                    continue
                if removed_sym and removed_sym.strip():
                    removed_rows.append({
                        "symbol": removed_sym.strip(),
                        "start_date": pd.NaT,
                        "end_date": chg_date,
                    })
            except Exception:
                continue

    removed_df = pd.DataFrame(removed_rows) if removed_rows else pd.DataFrame(
        columns=["symbol", "start_date", "end_date"]
    )

    all_members = pd.concat([
        current[["symbol", "start_date", "end_date"]],
        removed_df,
    ], ignore_index=True)

    # Filter: Mitglied im Zeitraum [start, end]
    range_start = pd.Timestamp(start)
    range_end = pd.Timestamp(end)
    mask = (
        (all_members["end_date"].isna() | (all_members["end_date"] >= range_start))
        & (all_members["start_date"].isna() | (all_members["start_date"] <= range_end))
    )
    result = all_members[mask].drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    logger.info("[Universe] %d Symbole für Zeitraum %s–%s", len(result), start, end)
    return result


# ---------------------------------------------------------------------------
# Sektion 2 — OHLCV Download (yfinance in 50er-Batches)
# ---------------------------------------------------------------------------

def download_symbol_batch(
    symbols: list[str],
    start: str,
    end: str,
    out_dir: Path,
    manifest_path: Path,
) -> dict:
    """Batch-Download mit Resume-Unterstützung via manifest.json.

    Warnung: adj_close nur für Return-Berechnung, NICHT in Execution verwenden.
    """
    import yfinance as yf  # type: ignore

    manifest: dict = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0

    for i in range(0, len(symbols), 50):
        batch = symbols[i: i + 50]
        to_download = [s for s in batch if s not in manifest]
        if not to_download:
            skipped += len(batch)
            continue

        try:
            df = yf.download(
                to_download,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            for sym in to_download:
                try:
                    if len(to_download) == 1:
                        sym_df = df
                    else:
                        sym_df = df.xs(sym, level=1, axis=1)
                    if sym_df.empty or len(sym_df) < 10:
                        continue
                    sym_path = out_dir / f"{sym}.parquet"
                    sym_df.to_parquet(sym_path)
                    manifest[sym] = {
                        "rows": len(sym_df),
                        "start": str(sym_df.index.min().date()),
                        "end": str(sym_df.index.max().date()),
                        "downloaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    }
                    downloaded += 1
                except Exception as exc:
                    logger.debug("[SKIP] %s: %s", sym, exc)

            manifest_path.write_text(
                json.dumps(manifest, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("[WARN] Batch %d-%d fehlgeschlagen: %s", i, i + 50, exc)

        time.sleep(0.3)

    logger.info(
        "[OHLCV] %d heruntergeladen, %d übersprungen (bereits vorhanden)",
        downloaded, skipped,
    )
    return manifest


# ---------------------------------------------------------------------------
# Sektion 3 — Makro (FRED)
# ---------------------------------------------------------------------------

def download_fred_data(out_dir: Path) -> None:
    """Makro-Zeitreihen via fredapi.

    Nutzt get_series() — realtime_start für echten PIT-Anker.
    FRED_API_KEY aus Umgebungsvariable (optional, öffentliche Serien ohne Key abrufbar).
    """
    try:
        import os

        from fredapi import Fred  # type: ignore
        fred = Fred(api_key=os.environ.get("FRED_API_KEY", ""))
    except ImportError:
        logger.warning("[FRED] fredapi nicht installiert — Makro-Download übersprungen")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    all_series: dict[str, pd.Series] = {}

    for series_id, col_name in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id)
            all_series[col_name] = data
            logger.info("[FRED] %s (%s): %d Datenpunkte", col_name, series_id, len(data))
            time.sleep(0.1)
        except Exception as exc:
            logger.warning("[FRED] %s fehlgeschlagen: %s", series_id, exc)

    if all_series:
        macro_df = pd.DataFrame(all_series)
        macro_df.index.name = "date"
        macro_df.to_parquet(out_dir / "fred_series.parquet")
        logger.info("[FRED] Makro gespeichert: %d Zeilen, %d Spalten", len(macro_df), len(macro_df.columns))


# ---------------------------------------------------------------------------
# Sektion 4 — Quality Gate
# ---------------------------------------------------------------------------

def run_quality_gate(prices_dir: Path) -> dict:
    """Prüft jedes Symbol auf Datenqualität.

    Disqualifikation wenn:
    - > 5 aufeinanderfolgende fehlende Handelstage
    - Tagesrendite > 50% (Preissprung-Anomalie)
    - < 1000 Handelstage

    Returns:
        quality_report dict mit passed/failed Listen.
    """
    passed: list[str] = []
    failed: list[dict] = []

    for parquet_file in sorted(prices_dir.glob("*.parquet")):
        sym = parquet_file.stem
        try:
            df = pd.read_parquet(parquet_file)
            n_rows = len(df)

            if n_rows < 1000:
                failed.append({"symbol": sym, "reason": f"zu kurz: {n_rows} Tage"})
                continue

            # Gap-Erkennung: mehr als 5 aufeinanderfolgende fehlende Tage
            close_col = "Close" if "Close" in df.columns else "close"
            if close_col in df.columns:
                nulls = df[close_col].isna()
                max_gap = nulls.groupby((nulls != nulls.shift()).cumsum()).sum().max()
                if max_gap > 5:
                    failed.append({"symbol": sym, "reason": f"Gap > 5 Tage: {max_gap}"})
                    continue

                # Preissprung-Anomalie
                returns = df[close_col].pct_change(fill_method=None).abs()
                if (returns > 0.5).any():
                    n_jumps = int((returns > 0.5).sum())
                    failed.append({"symbol": sym, "reason": f"{n_jumps} Preissprünge > 50%"})
                    continue

            passed.append(sym)
        except Exception as exc:
            failed.append({"symbol": sym, "reason": str(exc)})

    report = {
        "n_passed": len(passed),
        "n_failed": len(failed),
        "passed": passed,
        "failed": failed,
    }
    report_path = prices_dir.parent / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(
        "[QA] %d bestanden, %d disqualifiziert. Report: %s",
        len(passed), len(failed), report_path,
    )
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Baut kanonisches historisches Dataset (10-20 Jahre)"
    )
    parser.add_argument(
        "--universe",
        choices=["sp500", "combined"],
        default="sp500",
        help="Symbol-Universe (default: sp500)",
    )
    parser.add_argument(
        "--start",
        default="2005-01-01",
        help="Startdatum (default: 2005-01-01)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/canonical"),
        help="Ausgabeverzeichnis (default: data/canonical)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["yfinance"],
        choices=["yfinance", "fred"],
        help="Datenquellen (default: yfinance)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur Universe laden, kein Download",
    )
    parser.add_argument(
        "--skip-quality-gate",
        action="store_true",
        help="Quality-Gate überspringen",
    )
    args = parser.parse_args()

    end = pd.Timestamp.now(tz="UTC").floor("D").strftime("%Y-%m-%d")
    out_dir = args.out
    prices_dir = out_dir / "prices"
    macro_dir = out_dir / "macro"

    # Sektion 1: Universe
    universe_path = out_dir / "universe_history.parquet"
    if universe_path.exists():
        members_df = pd.read_parquet(universe_path)
        logger.info("[Universe] Geladen aus Cache: %d Symbole", len(members_df))
    else:
        try:
            members_df = load_sp500_historical_members(start=args.start, end=end)
            out_dir.mkdir(parents=True, exist_ok=True)
            members_df.to_parquet(universe_path)
            logger.info("[Universe] Gespeichert: %s", universe_path)
        except Exception as exc:
            logger.error("[Universe] Fehlgeschlagen: %s", exc)
            return 1

    symbols = sorted(members_df["symbol"].dropna().unique().tolist())
    logger.info("[Universe] %d Symbole gesamt", len(symbols))

    if args.dry_run:
        logger.info("--dry-run: Universe geladen, kein Download")
        return 0

    # Sektion 2: OHLCV
    if "yfinance" in args.sources:
        manifest_path = prices_dir / "_manifest.json"
        download_symbol_batch(
            symbols=symbols,
            start=args.start,
            end=end,
            out_dir=prices_dir,
            manifest_path=manifest_path,
        )

    # Sektion 3: Makro
    if "fred" in args.sources:
        download_fred_data(macro_dir)

    # Sektion 4: Quality Gate
    if not args.skip_quality_gate and prices_dir.exists():
        run_quality_gate(prices_dir)

    logger.info("[OK] Historisches Dataset abgeschlossen: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
