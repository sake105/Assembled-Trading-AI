"""13F-Parser Phase 1 — Top-100-Manager je Quartal (PIT-fair) + deren Holdings.

Je Quartals-ZIP: SUBMISSION (Filer-CIK, Filing-Datum = PIT-Anker) + INFOTABLE
(CUSIP/VALUE/Shares). Ranking je Quartal nach Portfolio-Summe AUS DEN DATEN
(kein Rueckschau-Star-Picking). Output: data/13f_top100.parquet (alle Quartale).

Bekannte Grenzen (Phase 2): CUSIP->Ticker-Mapping fehlt noch (openfigi/Referenz);
VALUE-Einheit wechselt ~2023 von Tausend-USD auf USD — fuer RANKINGS je Quartal
irrelevant, fuer absolute Gewichte spaeter normalisieren. KEIN Trial.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
ZIPS = sorted((DATA / "13f").glob("*.zip"))
OUT = DATA / "13f_top100.parquet"
TOP_N = 100


def parse_zip(zp: Path) -> pd.DataFrame | None:
    try:
        z = zipfile.ZipFile(zp)
        sub = pd.read_csv(
            z.open("SUBMISSION.tsv"), sep="\t", dtype=str, on_bad_lines="skip"
        )
        info = pd.read_csv(
            z.open("INFOTABLE.tsv"),
            sep="\t",
            dtype=str,
            on_bad_lines="skip",
            usecols=[
                "ACCESSION_NUMBER",
                "NAMEOFISSUER",
                "CUSIP",
                "VALUE",
                "SSHPRNAMT",
                "PUTCALL",
            ],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {zp.name}: {exc}", flush=True)
        return None
    sub.columns = [c.upper() for c in sub.columns]
    info["VALUE"] = pd.to_numeric(info["VALUE"], errors="coerce")
    info = info[info["PUTCALL"].isna()]  # long equity only, no options overlays
    totals = info.groupby("ACCESSION_NUMBER")["VALUE"].sum()
    # restrict to original 13F-HR (no amendments) where the column exists
    if "SUBMISSIONTYPE" in sub.columns:
        sub = sub[sub["SUBMISSIONTYPE"] == "13F-HR"]
    sub = sub.set_index("ACCESSION_NUMBER")
    sub["total"] = totals
    sub = sub.dropna(subset=["total"]).sort_values("total", ascending=False)
    top = sub.head(TOP_N)
    hold = info[info["ACCESSION_NUMBER"].isin(top.index)].copy()
    hold = hold.merge(
        top[["CIK", "PERIODOFREPORT", "FILING_DATE"]]
        .rename_axis("ACCESSION_NUMBER")
        .reset_index()
        if "FILING_DATE" in top.columns
        else top[["CIK", "PERIODOFREPORT"]]
        .rename_axis("ACCESSION_NUMBER")
        .reset_index(),
        on="ACCESSION_NUMBER",
        how="left",
    )
    hold["quarter_zip"] = zp.name
    return hold


def main() -> int:
    frames = []
    for zp in ZIPS:
        df = parse_zip(zp)
        if df is not None and len(df):
            frames.append(df)
            print(
                f"[OK] {zp.name}: {df['ACCESSION_NUMBER'].nunique()} managers, {len(df)} holdings",
                flush=True,
            )
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(
        f"[DONE] {len(out)} holding rows, {out['quarter_zip'].nunique()} quarters -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
