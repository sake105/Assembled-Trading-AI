"""Fetch Caldara-Iacoviello Geopolitical Risk Index (GPR).

Downloads the monthly GPR Excel export from matteoiacoviello.com (free,
public, no auth) and writes a tidy parquet ready for the multifactor_v2
geo-risk-composite path. The previously-buggy FRED "GPRC" fetch was
removed in 6be8ce3 because the series does not exist in FRED — this is
the proper canonical source.

Series exported (all monthly, 1900-01-01 → current):
  - GPR    = headline Geopolitical Risk Index (the one the strategy uses)
  - GPRT   = threats sub-component
  - GPRA   = acts sub-component
  - GPRH   = historical version (more conservative)

Usage::

    python -m scripts.ops.fetch_caldara_iacoviello_gpr
    python -m scripts.ops.fetch_caldara_iacoviello_gpr \\
        --out output/macro_gpr.parquet
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import urllib.request
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Canonical URL and UA
_GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Columns we extract for the trading pipeline
_WANTED_COLS = ["GPR", "GPRT", "GPRA", "GPRH"]


def fetch_gpr_excel(url: str = _GPR_URL, timeout: int = 30) -> pd.DataFrame:
    """Download and parse the Caldara-Iacoviello GPR Excel file."""
    log.info("[START] fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    log.info("  downloaded %d bytes", len(body))
    df = pd.read_excel(io.BytesIO(body), engine="xlrd")
    return df


def normalize(df: pd.DataFrame, wanted: list[str] = _WANTED_COLS) -> pd.DataFrame:
    """Reduce the wide 115-column export to a tidy GPR panel.

    Output schema:
      - timestamp (UTC, normalized to month-start)
      - gpr_index (= GPR)
      - gpr_threats (= GPRT)
      - gpr_acts (= GPRA)
      - gpr_historical (= GPRH)
    """
    if "month" not in df.columns:
        raise ValueError(
            f"month column missing in source; got {list(df.columns)[:5]}..."
        )
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise ValueError(f"GPR columns missing in source: {missing}")
    out = df[["month"] + wanted].dropna(subset=["month"]).copy()
    out["timestamp"] = pd.to_datetime(out["month"], utc=True)
    out = out.drop(columns=["month"])
    out = out.rename(
        columns={
            "GPR": "gpr_index",
            "GPRT": "gpr_threats",
            "GPRA": "gpr_acts",
            "GPRH": "gpr_historical",
        }
    )
    out = out[["timestamp", "gpr_index", "gpr_threats", "gpr_acts", "gpr_historical"]]
    return out.sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "output" / "macro_gpr.parquet",
        help="Output parquet path (default: output/macro_gpr.parquet).",
    )
    parser.add_argument("--url", default=_GPR_URL, help="Override source URL.")
    args = parser.parse_args()

    raw = fetch_gpr_excel(url=args.url)
    log.info("  raw shape: %s", raw.shape)
    tidy = normalize(raw)
    log.info(
        "  tidy: rows=%d range=%s..%s latest_gpr=%.2f",
        len(tidy),
        tidy["timestamp"].min().date(),
        tidy["timestamp"].max().date(),
        tidy["gpr_index"].iloc[-1],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(args.out, index=False)
    log.info("[OK] wrote -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
