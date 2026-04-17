"""C2 — Offline factor-store pre-warm.

Populates the factor store with ``core_ta`` features for a given universe /
date range so subsequent backtests hit the warm path. Designed for:

* nightly / weekly cron on CI (separate workflow, see
  ``.github/workflows/prewarm-factor-store.yml``)
* manual invocation before a WF-grid sweep.

Usage
-----

    python scripts/prewarm_factor_store.py \
        --universe SPX100 \
        --start 2022-01-01 --end 2025-12-31 \
        --freq 1d

Loads prices via the project's unified data loader (if available) or falls
back to a deterministic synthetic panel so the script is usable inside CI
without real data.

PIT-safety: the builder runs against prices; the cache stores rows with their
original timestamps. Readers enforce PIT via ``load_factors(as_of=...)`` —
this script does *not* itself introduce leakage.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assembled_core.data.factor_store import compute_universe_key  # noqa: E402
from src.assembled_core.features.factor_store_integration import (  # noqa: E402
    build_or_load_factors,
)

logger = logging.getLogger("prewarm_factor_store")


# ---------------------------------------------------------------------------
# Universe resolution
# ---------------------------------------------------------------------------


_BUILTIN_UNIVERSES: dict[str, list[str]] = {
    "SPX100_MINI": [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "JPM",
        "V",
        "WMT",
        "UNH",
        "XOM",
        "PG",
        "MA",
        "HD",
        "JNJ",
        "ABBV",
        "KO",
        "PEP",
        "AVGO",
        "MRK",
        "CVX",
        "LLY",
        "BAC",
        "PFE",
    ],
    "DEMO": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
}


def _resolve_universe(name_or_path: str) -> list[str]:
    if name_or_path in _BUILTIN_UNIVERSES:
        return list(_BUILTIN_UNIVERSES[name_or_path])
    path = Path(name_or_path)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        return [s.strip() for s in text.splitlines() if s.strip() and not s.startswith("#")]
    raise ValueError(
        f"Universe '{name_or_path}' is neither a known preset "
        f"{sorted(_BUILTIN_UNIVERSES)} nor an existing file path"
    )


# ---------------------------------------------------------------------------
# Price loader — real if available, synthetic fallback
# ---------------------------------------------------------------------------


def _load_real_prices(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str,
) -> pd.DataFrame | None:
    """Best-effort load via project data-layer. Returns None on any import
    or data-shape failure so the caller can fall back to synthetic prices."""
    try:
        from src.assembled_core.data.loaders import load_prices  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        df = load_prices(symbols=symbols, start=start, end=end, freq=freq)
    except Exception as exc:
        logger.warning("[prewarm] real loader failed: %s", exc)
        return None
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    if df is None or df.empty or not required.issubset(df.columns):
        return None
    return df


def _synthetic_prices(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="B", tz="UTC")
    rows: list[dict[str, Any]] = []
    for s_i, sym in enumerate(symbols):
        base = 100.0 + 10.0 * s_i
        rets = rng.normal(loc=0.0003, scale=0.012, size=len(dates))
        close = base * np.exp(np.cumsum(rets))
        for i, ts in enumerate(dates):
            c = float(close[i])
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "open": c * 0.999,
                    "high": c * 1.003,
                    "low": c * 0.997,
                    "close": c,
                    "volume": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def prewarm(
    *,
    universe_name: str,
    start: str,
    end: str,
    freq: str = "1d",
    factor_group: str = "core_ta",
    factors_root: Path | None = None,
    force_rebuild: bool = False,
    allow_synthetic: bool = True,
) -> dict[str, Any]:
    """Populate the factor cache. Returns a manifest dict."""
    t0 = datetime.now(tz=timezone.utc)

    symbols = _resolve_universe(universe_name)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    prices = _load_real_prices(symbols, start_ts, end_ts, freq)
    source = "real"
    if prices is None:
        if not allow_synthetic:
            raise RuntimeError(
                "real price loader unavailable and --no-synthetic was set"
            )
        prices = _synthetic_prices(symbols, start_ts, end_ts)
        source = "synthetic"

    universe_key = compute_universe_key(symbols=symbols)
    logger.info(
        "[prewarm] universe=%s symbols=%d start=%s end=%s source=%s "
        "factor_group=%s freq=%s force_rebuild=%s",
        universe_name,
        len(symbols),
        start_ts.date(),
        end_ts.date(),
        source,
        factor_group,
        freq,
        force_rebuild,
    )

    factors = build_or_load_factors(
        prices=prices,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        start_date=start_ts,
        end_date=end_ts,
        force_rebuild=force_rebuild,
        factors_root=factors_root,
    )
    t1 = datetime.now(tz=timezone.utc)

    return {
        "universe": universe_name,
        "universe_key": universe_key,
        "n_symbols": len(symbols),
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "factor_group": factor_group,
        "freq": freq,
        "source": source,
        "rows": int(len(factors)),
        "elapsed_sec": (t1 - t0).total_seconds(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="DEMO")
    parser.add_argument("--start", default="2022-01-03")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--freq", default="1d")
    parser.add_argument("--factor-group", default="core_ta")
    parser.add_argument("--factors-root", default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Fail if real price loader is unavailable (CI-ish mode).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    try:
        manifest = prewarm(
            universe_name=args.universe,
            start=args.start,
            end=args.end,
            freq=args.freq,
            factor_group=args.factor_group,
            factors_root=Path(args.factors_root) if args.factors_root else None,
            force_rebuild=args.force_rebuild,
            allow_synthetic=not args.no_synthetic,
        )
    except Exception as exc:
        logger.error("[prewarm] failed: %s", exc, exc_info=True)
        return 1

    logger.info("[prewarm] done: %s", manifest)
    print(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
