"""Factor store for caching computed factors (Parquet-based).

Provides compute_universe_key, load_factors, store_factors for the factor
cache layer used by features.factor_store_integration and pipeline.trading_cycle.

Storage layout: <root>/<factor_group>/<freq>/<universe_key>/year=YYYY.parquet
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _default_factors_root() -> Path:
    """Return default factor store root anchored to project base dir."""
    try:
        from src.assembled_core.config import get_base_dir  # noqa: PLC0415

        return get_base_dir() / "output" / "factors"
    except Exception:
        return Path("output/factors")


def get_factor_store_root(factors_root: Path | None = None) -> Path:
    """Return the root directory for the factor store, creating it if needed."""
    root = factors_root or _default_factors_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def compute_universe_key(
    symbols: list[str] | None = None,
    universe_file: Path | None = None,
) -> str:
    """Compute a deterministic hash key for a universe of symbols.

    Args:
        symbols: List of ticker symbols (order-independent).
        universe_file: Optional path to a file with one symbol per line.

    Returns:
        Short identifier prefixed with 'universe_'.
    """
    if universe_file is not None:
        text = Path(universe_file).read_text(encoding="utf-8")
        symbols = sorted(s.strip() for s in text.splitlines() if s.strip())
    if symbols is None:
        symbols = []
    joined = ",".join(sorted(symbols))
    digest = hashlib.sha256(joined.encode()).hexdigest()[:12]
    return f"universe_{digest}"


def panel_path(
    factor_group: str,
    freq: str,
    universe_key: str,
    year: int | None = None,
    factors_root: Path | None = None,
) -> Path:
    """Return path to a panel directory or a specific year partition file.

    Args:
        factor_group: Factor group name.
        freq: Trading frequency (e.g. '1d').
        universe_key: Universe identifier.
        year: If given, returns path to year=YYYY.parquet; else returns the directory.
        factors_root: Optional root directory.
    """
    root = factors_root or _default_factors_root()
    panel_dir = root / factor_group / freq / universe_key
    if year is None:
        return panel_dir
    return panel_dir / f"year={year}.parquet"


def factor_partition_path(
    group: str,
    freq: str,
    universe_key: str,
    year: int,
    root: Path | None = None,
) -> Path:
    """Return path to a specific year partition file.

    Args:
        group: Factor group name.
        freq: Trading frequency.
        universe_key: Universe identifier.
        year: Year number.
        root: Optional root directory.
    """
    return panel_path(group, freq, universe_key, year=year, factors_root=root)


def _write_parquet_atomic(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to Parquet atomically via temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp.parquet")
    try:
        os.close(fd)
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_all_partitions(panel_dir: Path) -> pd.DataFrame | None:
    """Load and concatenate all year=*.parquet files from a panel directory.

    Corrupt partitions were previously skipped at WARNING level, which meant a
    silently truncated panel (e.g. years 2020, 2022, 2024 loading but 2021/2023
    dropped due to a bad parquet footer) looked identical to a legitimately
    partial store. Backtests then computed Sharpe/IC on an incomplete time
    series without any trace. We now elevate the log level to ERROR and, when
    a ``_metadata.json`` manifest is present, verify that the years actually
    loaded match the manifest's declared year list — a mismatch is still
    permissive (returns partial data) because ``store_factors`` calls this on
    append-mode rebuilds where tolerance is expected, but the ERROR log makes
    the corruption visible to operators and CI log scrapers.
    """
    parts = sorted(panel_dir.glob("year=*.parquet"))
    if not parts:
        return None
    dfs = []
    failed_parts: list[str] = []
    for p in parts:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception:
            failed_parts.append(str(p))
            logger.error(
                "[factor_store] Failed to read partition %s", p, exc_info=True
            )
    if failed_parts:
        logger.error(
            "[factor_store] %d of %d partitions unreadable in %s: %s",
            len(failed_parts),
            len(parts),
            panel_dir,
            failed_parts,
        )
        manifest_path = panel_dir / "_metadata.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                declared = set(int(y) for y in manifest.get("years", []))
                loaded_years = set()
                for df_part in dfs:
                    if "timestamp" in df_part.columns and not df_part.empty:
                        ts = pd.to_datetime(df_part["timestamp"], utc=True)
                        loaded_years.update(int(y) for y in ts.dt.year.unique())
                missing = sorted(declared - loaded_years)
                if missing:
                    logger.error(
                        "[factor_store] Manifest/data mismatch in %s: "
                        "manifest declares years %s, loaded years %s, "
                        "missing %s",
                        panel_dir,
                        sorted(declared),
                        sorted(loaded_years),
                        missing,
                    )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.error(
                    "[factor_store] Failed to read manifest %s for "
                    "consistency check: %s",
                    manifest_path,
                    exc,
                )
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def _write_manifest(
    panel_dir: Path,
    df: pd.DataFrame,
    factor_group: str,
    freq: str,
    universe_key: str,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Write _metadata.json manifest to the panel directory."""
    ts_col = df["timestamp"] if "timestamp" in df.columns else pd.Series(dtype="object")
    years: list[int] = (
        sorted(ts_col.dt.year.unique().tolist()) if len(ts_col) > 0 else []
    )
    factor_cols = [c for c in df.columns if c not in ("timestamp", "symbol", "date")]
    date_range: dict[str, str] = {}
    if len(ts_col) > 0:
        date_range = {
            "start": ts_col.min().isoformat(),
            "end": ts_col.max().isoformat(),
        }
    schema = {col: str(df[col].dtype) for col in df.columns}
    config_input = f"{factor_group}:{freq}:{universe_key}:{sorted(factor_cols)}"
    config_hash = hashlib.sha256(config_input.encode()).hexdigest()[:16]

    manifest: dict[str, Any] = {
        "factor_group": factor_group,
        "freq": freq,
        "universe_key": universe_key,
        "computed_at": datetime.now(tz=timezone.utc).isoformat(),
        "years": years,
        "date_range": date_range,
        "factor_columns": factor_cols,
        "schema": schema,
        "config_hash": config_hash,
    }
    if extra_metadata:
        manifest.update(extra_metadata)

    (panel_dir / "_metadata.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def store_factors(
    df: pd.DataFrame,
    factor_group: str,
    freq: str,
    universe_key: str,
    mode: str = "overwrite",
    factors_root: Path | None = None,
    metadata: dict[str, Any] | None = None,
    write_manifest: bool = True,
) -> Path:
    """Store computed factors using year-based partitioning.

    Args:
        df: DataFrame with factor data.
        factor_group: Factor group name.
        freq: Trading frequency.
        universe_key: Universe hash key.
        mode: 'overwrite' replaces all existing partitions; 'append' merges.
        factors_root: Optional root directory.
        metadata: Optional extra fields for the manifest.
        write_manifest: Whether to write _metadata.json.

    Returns:
        Path to the panel directory.
    """
    missing = {"timestamp", "symbol"} - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    panel_dir = panel_path(factor_group, freq, universe_key, factors_root=factors_root)
    panel_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if "date" not in df.columns:
            df["date"] = df["timestamp"].dt.date.astype(str)

    if mode == "overwrite":
        for old in panel_dir.glob("year=*.parquet"):
            old.unlink()
    elif mode == "append":
        existing = _load_all_partitions(panel_dir)
        if existing is not None:
            if "timestamp" in existing.columns:
                existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
                subset=["timestamp", "symbol"], keep="last"
            )

    if "timestamp" in df.columns and "symbol" in df.columns:
        df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    # Write year partitions
    if "timestamp" in df.columns:
        for year, year_df in df.groupby(df["timestamp"].dt.year):
            year_file = panel_dir / f"year={int(year)}.parquet"
            _write_parquet_atomic(year_file, year_df.reset_index(drop=True))
    else:
        _write_parquet_atomic(panel_dir / "data.parquet", df)

    if write_manifest:
        _write_manifest(panel_dir, df, factor_group, freq, universe_key, metadata)

    logger.info("[factor_store] Stored %d rows to %s", len(df), panel_dir)
    return panel_dir


def load_factors(
    factor_group: str,
    freq: str,
    universe_key: str,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    as_of: pd.Timestamp | str | None = None,
    factors_root: Path | None = None,
) -> pd.DataFrame | None:
    """Load cached factors from the year-partitioned store.

    Returns None if no cache exists or the filtered result is empty.
    """
    panel_dir = panel_path(factor_group, freq, universe_key, factors_root=factors_root)
    if not panel_dir.exists():
        return None

    df = _load_all_partitions(panel_dir)
    if df is None:
        return None

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        def _to_ts(val: pd.Timestamp | str) -> pd.Timestamp:
            if isinstance(val, pd.Timestamp):
                return val if val.tzinfo is not None else val.tz_localize("UTC")
            return pd.Timestamp(val, tz="UTC")

        if start_date is not None:
            df = df[df["timestamp"] >= _to_ts(start_date)]
        if end_date is not None:
            df = df[df["timestamp"] <= _to_ts(end_date)]
        if as_of is not None:
            df = df[df["timestamp"] <= _to_ts(as_of)]

    if df.empty:
        return None

    if "timestamp" in df.columns and "symbol" in df.columns:
        df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    return df


def store_factors_parquet(
    df: pd.DataFrame,
    group: str,
    freq: str,
    universe: str,
    mode: str = "replace",
    root: Path | None = None,
) -> None:
    """Store factors via the year-partitioned API (validates required columns).

    Args:
        df: DataFrame — must contain 'timestamp' and 'symbol'.
        group: Factor group name.
        freq: Trading frequency.
        universe: Universe key.
        mode: 'replace' (overwrite) or 'append'.
        root: Optional root directory.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = {"timestamp", "symbol"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    internal_mode = "overwrite" if mode == "replace" else "append"
    store_factors(
        df=df,
        factor_group=group,
        freq=freq,
        universe_key=universe,
        mode=internal_mode,
        factors_root=root,
        write_manifest=False,
    )


def load_factors_parquet(
    group: str,
    freq: str,
    universe: str,
    root: Path | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Load factors from the year-partitioned store.

    Args:
        group: Factor group name.
        freq: Trading frequency.
        universe: Universe key.
        root: Optional root directory.
        start_date: Optional start filter.
        end_date: Optional end filter.

    Returns:
        DataFrame sorted by (timestamp, symbol), or None if not found.
    """
    return load_factors(
        factor_group=group,
        freq=freq,
        universe_key=universe,
        start_date=start_date or start,
        end_date=end_date or end,
        factors_root=root,
    )


def list_factor_partitions(
    group: str,
    freq: str,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """List all available factor partitions for a group and frequency.

    Returns:
        List of dicts with keys: factor_group, freq, universe_key, years.
    """
    base = (root or _default_factors_root()) / group / freq
    if not base.exists():
        return []

    result = []
    for universe_dir in sorted(base.iterdir()):
        if not universe_dir.is_dir() or universe_dir.name.startswith("_"):
            continue
        years = []
        for p in sorted(universe_dir.glob("year=*.parquet")):
            try:
                years.append(int(p.stem.split("=")[1]))
            except (ValueError, IndexError) as exc:
                logger.warning("[FactorStore] failed to parse year from %s: %s", p, exc)
        entry: dict[str, Any] = {
            "factor_group": group,
            "freq": freq,
            "universe_key": universe_dir.name,
            "years": sorted(years),
        }
        manifest_file = universe_dir / "_metadata.json"
        if manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                for key in ("factor_columns", "schema", "date_range", "config_hash"):
                    if key in manifest:
                        entry[key] = manifest[key]
            except Exception as exc:
                logger.warning("[FactorStore] failed to read metadata from %s: %s", manifest_file, exc)
        result.append(entry)

    return result


def list_available_panels(
    factors_root: Path | None = None,
    factor_group: str | None = None,
    freq: str | None = None,
) -> list[dict[str, Any]]:
    """List all available factor panels across all groups and frequencies.

    Args:
        factors_root: Optional root directory.
        factor_group: If given, only return panels for this group.
        freq: If given, only return panels for this frequency.

    Returns:
        List of dicts with keys: factor_group, freq, universe_key, years.
    """
    root = factors_root or _default_factors_root()
    if not root.exists():
        return []

    result = []
    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir() or group_dir.name.startswith("_"):
            continue
        if factor_group is not None and group_dir.name != factor_group:
            continue
        for freq_dir in sorted(group_dir.iterdir()):
            if not freq_dir.is_dir() or freq_dir.name.startswith("_"):
                continue
            if freq is not None and freq_dir.name != freq:
                continue
            result.extend(
                list_factor_partitions(
                    group=group_dir.name, freq=freq_dir.name, root=root
                )
            )

    return result
