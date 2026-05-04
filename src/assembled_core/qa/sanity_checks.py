"""Pre-trade sanity checks — catch obviously wrong trades before submission (Plan 11/10 §5.2).

These are NOT hard risk gates (those reject). They are anomaly detectors
that flag for review and optionally hold the trade. Checks run only in
paper/live mode; backtest mode skips to avoid false positives.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class SanityChecker:
    """Run pre-trade sanity checks against recent trade history."""

    def __init__(
        self, history_dir: Path | str | None = None, lookback_days: int = 30
    ) -> None:
        self.history_dir = (
            Path(history_dir) if history_dir else Path("output/trade_journal")
        )
        self.lookback_days = lookback_days
        self._recent_trades: pd.DataFrame | None = None

    @property
    def recent_trades(self) -> pd.DataFrame:
        if self._recent_trades is None:
            self._recent_trades = self._load_recent_trades(self.lookback_days)
        return self._recent_trades

    def _load_recent_trades(self, days: int) -> pd.DataFrame:
        if not self.history_dir.exists():
            return pd.DataFrame(columns=["symbol", "side", "qty", "timestamp"])
        files = sorted(self.history_dir.rglob("*.jsonl"))[-days:]
        if not files:
            return pd.DataFrame(columns=["symbol", "side", "qty", "timestamp"])
        import json

        rows = []
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        if not rows:
            return pd.DataFrame(columns=["symbol", "side", "qty", "timestamp"])
        df = pd.DataFrame(rows)
        for col in ("symbol", "side", "qty", "timestamp"):
            if col not in df.columns:
                df[col] = None
        return df

    # ------------------------------------------------------------------
    def check_order(self, order: dict, market_state: dict | None = None) -> dict:
        """Run all sanity checks for one order.

        Args:
            order: Dict with keys: symbol, side, qty, limit_price, as_of (optional).
            market_state: Optional real-time data: last_quote, last_quote_time,
                daily_news_sentiment, symbol_in_pit_universe.

        Returns:
            Dict: n_flags, flags (list of dicts), halt_recommendation, max_severity.
        """
        market_state = market_state or {}
        flags: list[dict] = []
        halt = False

        sym = str(order.get("symbol", ""))
        side = str(order.get("side", "")).lower()
        qty = float(order.get("qty", 0) or 0)
        limit_price = float(order.get("limit_price", 0) or 0)

        # Check 1: Position size vs historical average
        sym_hist = (
            self.recent_trades[self.recent_trades["symbol"] == sym]
            if "symbol" in self.recent_trades.columns and not self.recent_trades.empty
            else pd.DataFrame()
        )
        if not sym_hist.empty and "qty" in sym_hist.columns:
            avg_size = sym_hist["qty"].abs().mean()
            if avg_size > 0 and qty > avg_size * 5:
                flags.append(
                    {
                        "rule": "position_5x_typical",
                        "severity": "high",
                        "detail": f"qty {qty:.0f} vs 30d avg {avg_size:.0f}",
                    }
                )
                halt = True

        # Check 2: Whipsaw — last 3 trades all opposite direction
        if not sym_hist.empty and "side" in sym_hist.columns:
            last_3 = (
                sym_hist.sort_values("timestamp").tail(3)
                if "timestamp" in sym_hist.columns
                else sym_hist.tail(3)
            )
            if len(last_3) == 3:
                all_opposite = (last_3["side"].str.lower() != side).all()
                if all_opposite:
                    flags.append(
                        {
                            "rule": "whipsaw_3x_reverse",
                            "severity": "medium",
                            "detail": "All last 3 trades on this symbol opposite direction",
                        }
                    )

        # Check 3: Quote staleness + limit far from market
        if (
            "last_quote" in market_state
            and market_state["last_quote"]
            and limit_price > 0
        ):
            quote = float(market_state["last_quote"])
            if quote > 0:
                quote_drift = abs(limit_price - quote) / quote
                if quote_drift > 0.02:
                    flags.append(
                        {
                            "rule": "limit_far_from_quote",
                            "severity": "high",
                            "detail": f"Limit {limit_price:.2f} vs quote {quote:.2f} ({quote_drift:.1%} drift)",
                        }
                    )
                    halt = True

            if "last_quote_time" in market_state:
                try:
                    age_s = (
                        pd.Timestamp.now(tz="UTC")
                        - pd.Timestamp(market_state["last_quote_time"])
                    ).total_seconds()
                    if age_s > 60:
                        flags.append(
                            {
                                "rule": "quote_stale",
                                "severity": "medium",
                                "detail": f"Last quote was {age_s:.0f}s ago",
                            }
                        )
                except Exception:
                    pass

        # Check 4: Buying despite strongly negative daily news sentiment
        if side == "buy":
            sentiment = float(market_state.get("daily_news_sentiment", 0) or 0)
            if sentiment < -0.5:
                flags.append(
                    {
                        "rule": "buy_against_news",
                        "severity": "low",
                        "detail": f"Daily sentiment: {sentiment:.2f}",
                    }
                )

        # Check 5: PIT universe membership
        if not market_state.get("symbol_in_pit_universe", True):
            flags.append(
                {
                    "rule": "symbol_not_in_pit_universe",
                    "severity": "critical",
                    "detail": f"{sym} not in PIT universe at {order.get('as_of', 'unknown')}",
                }
            )
            halt = True

        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
        max_sev = (
            max(
                flags,
                key=lambda f: severity_order.get(f["severity"], 0),
                default={"severity": "none"},
            )["severity"]
            if flags
            else "none"
        )

        if flags:
            logger.warning(
                "[sanity] %s %s: %d flag(s), max_severity=%s halt=%s",
                side.upper(),
                sym,
                len(flags),
                max_sev,
                halt,
            )

        return {
            "n_flags": len(flags),
            "flags": flags,
            "halt_recommendation": halt,
            "max_severity": max_sev,
        }

    def invalidate_cache(self) -> None:
        """Force reload of recent trades on next access."""
        self._recent_trades = None
