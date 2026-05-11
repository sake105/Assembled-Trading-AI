"""Persistent & In-Memory Data-Cache für Live-Pipeline.

Zwei Cache-Layer:
1. **LRU In-Memory**: schnelle Lookups, basiert auf mtime der Quelldatei.
2. **Parquet-Persistence**: precomputed Features (Mom-12/1, EMA, Vol-Trail).

Verwendung
----------
>>> cache = DataCache()
>>> panel = cache.load_yfinance_long_panel(["SPY", "TLT"])
>>> # Second call is ~0ms (in-memory)
>>> panel2 = cache.load_yfinance_long_panel(["SPY", "TLT"])
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class DataCache:
    """In-Memory-Cache für DataFrames mit mtime-Invalidation."""

    _memo: dict = field(default_factory=dict)
    """Map (cache_key, mtime) -> DataFrame."""

    cache_dir: Path = Path("data/cache/erweiterung_features")

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, *args) -> str:
        """Compute stable hash key."""
        s = "|".join(str(a) for a in args)
        return hashlib.md5(s.encode()).hexdigest()[:16]

    def _file_mtime(self, path: Path) -> float:
        try:
            return path.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    # =====================================================================
    # Yfinance-Long-Cache (11 ETFs)
    # =====================================================================
    def load_yfinance_long_panel(
        self, symbols: list[str], cache_dir: str | Path = "data/cache/yfinance_long"
    ) -> pd.DataFrame:
        """Lade Panel von 11 ETFs (oder Subset). In-Memory-cached."""
        cache_dir = Path(cache_dir)
        key = self._key("yf_long", tuple(sorted(symbols)), cache_dir)
        # mtime-check
        mtimes = tuple(
            self._file_mtime(cache_dir / f"{s}.parquet") for s in sorted(symbols)
        )
        cache_entry = self._memo.get(key)
        if cache_entry is not None and cache_entry["mtimes"] == mtimes:
            return cache_entry["df"]

        frames = []
        for sym in symbols:
            p = cache_dir / f"{sym}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p).reset_index()
            df["symbol"] = sym
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        panel = pd.concat(frames, ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"], utc=True)
        panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
        self._memo[key] = {"mtimes": mtimes, "df": panel}
        return panel

    # =====================================================================
    # Equity-Panel mit precomputed Mom-12/1
    # =====================================================================
    def load_equity_panel_with_mom(
        self,
        source_parquet: str | Path = "data/sample/watchlist_2007_2026.parquet",
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """Lade Equity-Panel + precomputed Mom-12/1-Spalte.

        Erste Run: rechnet + speichert in cache_dir/.
        Nächste Runs: lädt aus precomputed cache (ms statt 80ms).
        """
        source_path = Path(source_parquet)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source_mtime = self._file_mtime(source_path)
        cache_path = (
            self.cache_dir / f"equity_panel_with_mom_{self._key(source_path)}.parquet"
        )

        # In-memory check first
        mem_key = self._key("eq_mom", source_path)
        cache_entry = self._memo.get(mem_key)
        if (
            cache_entry is not None
            and cache_entry["mtime"] == source_mtime
            and not force_recompute
        ):
            return cache_entry["df"]

        # Persistent file check
        if cache_path.exists() and not force_recompute:
            if cache_path.stat().st_mtime >= source_mtime:
                df = pd.read_parquet(cache_path)
                self._memo[mem_key] = {"mtime": source_mtime, "df": df}
                return df

        # Compute fresh
        from erweiterung.factors.fama_french import momentum_12_1

        df = pd.read_parquet(source_path)
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "date"})
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        df["return"] = df.groupby("symbol")["close"].pct_change()
        mom = momentum_12_1(df[["date", "symbol", "close"]])
        df = df.set_index(["date", "symbol"])
        df["mom_12_1"] = mom.reindex(df.index)
        df = df.reset_index()

        # Persist
        df.to_parquet(cache_path)
        self._memo[mem_key] = {"mtime": source_mtime, "df": df}
        return df

    # =====================================================================
    # Generic features-cache helpers
    # =====================================================================
    def save_feature(self, name: str, df: pd.DataFrame, key: str | None = None) -> Path:
        """Save a computed feature DataFrame."""
        k = key or self._key(name)
        path = self.cache_dir / f"{name}_{k}.parquet"
        df.to_parquet(path)
        return path

    def load_feature(self, name: str, key: str | None = None) -> pd.DataFrame | None:
        """Load a previously-saved feature DataFrame."""
        k = key or self._key(name)
        path = self.cache_dir / f"{name}_{k}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def clear(self) -> None:
        """Clear in-memory cache (persistent files stay)."""
        self._memo.clear()

    def stats(self) -> dict:
        """In-memory cache stats."""
        return {
            "n_memo_entries": len(self._memo),
            "memo_keys": list(self._memo.keys()),
            "persistent_files": [p.name for p in self.cache_dir.glob("*.parquet")],
        }


_GLOBAL_CACHE: DataCache | None = None


def get_global_cache() -> DataCache:
    """Singleton-Access für global cache."""
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = DataCache()
    return _GLOBAL_CACHE


__all__ = ["DataCache", "get_global_cache"]
