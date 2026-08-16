"""Lightweight security master for symbol metadata."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

_REQUIRED_COLS = {"symbol", "sector", "region", "currency", "asset_type"}
_STR_COLS = ["symbol", "sector", "region", "currency", "asset_type"]


#: Candidate locations for the security master, most authoritative first.
#: configs/ is the real, git-tracked location; data/ is kept as a fallback
#: because it was the documented default before 2026-08-15.
_SECURITY_MASTER_CANDIDATES = (
    Path("configs") / "security_master.csv",
    Path("data") / "security_master.csv",
)


def get_default_security_master_path() -> Path:
    """Return the default security master CSV path.

    BUGFIX 2026-08-15: this returned ``data/security_master.csv``, which does
    not exist and never has — the file is git-tracked at
    ``configs/security_master.csv``. All three callers
    (``scripts/run_backtest_strategy.py``, ``scripts/run_daily.py``,
    ``pipeline/_tc_execution.py``) treat a missing file as "skip", each with a
    log line along the lines of "group exposure limits will be skipped". The
    net effect was that sector/region/FX group caps silently did not run in the
    default configuration — a risk control that was present in code, absent in
    behaviour (the E-135 family).

    COVERAGE WARNING: the file currently maps 32 symbols against a 195-symbol
    trading universe, and partial coverage here is RESTRICTIVE, not permissive.
    ``trading_cycle_shared.py::_apply_group_exposure_caps`` does
    ``out[dim].fillna("UNKNOWN")``, so every unmapped symbol joins one shared
    "UNKNOWN" bucket that is capped as a single group. Extend this file before
    enabling ``risk.group_limits``, or the cap will bite hardest exactly where
    the metadata is missing.

    Returns the first candidate that exists, falling back to the canonical
    ``configs/`` path so the error message names the right file.
    """
    for candidate in _SECURITY_MASTER_CANDIDATES:
        if candidate.exists():
            return candidate
    return _SECURITY_MASTER_CANDIDATES[0]


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _STR_COLS:
        if col in out.columns and hasattr(out[col], "str"):
            out[col] = out[col].str.strip()
    return out


def _validate_required(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")


def store_security_master(df: pd.DataFrame, path: Path | str) -> None:
    """Store security master atomically (.parquet or .csv).

    Raises:
        ValueError: If required columns are missing.
    """
    _validate_required(df)
    path = Path(path)
    out = _strip_strings(df).sort_values("symbol").reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp" + path.suffix)
    try:
        os.close(fd)
        if path.suffix == ".csv":
            out.to_csv(tmp, index=False)
        else:
            out.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_security_master(path: Path | str | None = None) -> pd.DataFrame:
    """Load security master from parquet or CSV.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required columns are missing or format unsupported.
    """
    if path is None:
        path = get_default_security_master_path()
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Security master not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(
            path,
            dtype={
                "symbol": "string",
                "name": "string",
                "exchange": "string",
                "sector": "string",
            },
        )
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    _validate_required(df)
    return _strip_strings(df).sort_values("symbol").reset_index(drop=True)


def resolve_security_meta(
    symbols: list[str],
    master_df: pd.DataFrame,
    missing_policy: str = "raise",
    default_sector: str = "UNKNOWN",
    default_region: str = "UNKNOWN",
    default_currency: str = "UNKNOWN",
    default_asset_type: str = "UNKNOWN",
) -> pd.DataFrame:
    """Resolve security metadata for a list of symbols.

    Args:
        symbols: Symbols to look up.
        master_df: Security master DataFrame.
        missing_policy: 'raise' raises ValueError for unknown symbols;
                        'default' fills them with defaults.

    Returns:
        DataFrame sorted by symbol.

    Raises:
        ValueError: If missing_policy='raise' and symbols are absent from master.
    """
    master = master_df.set_index("symbol")
    requested = sorted(set(symbols))
    missing = [s for s in requested if s not in master.index]

    if missing:
        if missing_policy == "raise":
            raise ValueError(f"Missing symbols in security master: {missing}")
        defaults = pd.DataFrame(
            [
                {
                    "symbol": s,
                    "sector": default_sector,
                    "region": default_region,
                    "currency": default_currency,
                    "asset_type": default_asset_type,
                }
                for s in missing
            ]
        ).set_index("symbol")
        master = pd.concat([master, defaults])

    result = master.loc[requested].reset_index()
    return result.sort_values("symbol").reset_index(drop=True)
