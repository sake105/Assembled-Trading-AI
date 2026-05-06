"""Master universe loader for full_us_universe.yaml (200-symbol US universe)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_DEFAULT_YAML = Path("configs/universes/full_us_universe.yaml")


def load_master_universe(
    yaml_path: str | Path = _DEFAULT_YAML,
) -> tuple[list[str], pd.DataFrame]:
    """Load master universe YAML.

    Returns
    -------
    symbols : list[str]
        Deduplicated ordered list of tickers.
    meta : pd.DataFrame
        Columns: symbol, sector, cap_class, mcap_b, type
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)

    rows: list[dict] = []
    seen: set[str] = set()

    for sector, sector_data in data.get("sectors", {}).items():
        for sym_entry in sector_data.get("symbols", []):
            ticker: str = sym_entry["ticker"].upper()
            if ticker in seen:
                continue
            seen.add(ticker)
            rows.append(
                {
                    "symbol": ticker,
                    "sector": sector,
                    "cap_class": sym_entry.get("cap", "unknown"),
                    "mcap_b": float(sym_entry.get("mcap_b", 0.0)),
                    "type": sym_entry.get("type", "equity"),
                }
            )

    meta = pd.DataFrame(rows)
    symbols = meta["symbol"].tolist()
    return symbols, meta


def write_flat_watchlist(
    yaml_path: str | Path = _DEFAULT_YAML,
    output_path: str | Path = "configs/universes/full_us_universe.txt",
) -> int:
    """Write flat ticker list for backward-compat file-based loaders.

    Returns number of symbols written.
    """
    symbols, _ = load_master_universe(yaml_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    return len(symbols)
