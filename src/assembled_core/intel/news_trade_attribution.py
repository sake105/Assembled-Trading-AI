"""News-zu-Trade-Attribution.

Verknüpft realisierte Trades mit News-Events im Zeitfenster um den Trade.
Gewichtet Events nach:
- zeitlicher Distanz (exponential decay)
- event impact_bps (vom Enricher geliefert)
- source_tier-Confidence

Resultat: welche News-Events haben vermutlich welchen Trade ausgelöst / gedrückt.

Verwendet bestehende News-Event-Struktur (`intel/news_models.py` oder kompatibel).

PIT-Invariante: Events müssen vor / während des Trade-Fensters published_at sein.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NewsLink:
    event_id: str
    symbol: str
    distance_hours: float
    weight: float
    estimated_contribution: float
    impact_bps: float = 0.0


@dataclass
class TradeAttribution:
    trade_id: str
    symbol: str
    closed_return: float
    news_links: list[NewsLink] = field(default_factory=list)
    residual_return: float = 0.0
    """Nicht-durch-News-erklärter Return."""


class NewsTradeAttributor:
    """Verknüpft Trades mit News-Events im Zeitfenster."""

    def __init__(
        self,
        pre_window_hours: float = 24.0,
        post_window_hours: float = 24.0,
        decay_halflife_hours: float = 12.0,
    ) -> None:
        """Args:
            pre_window_hours: Wie weit vor Trade-Open News relevant sind.
            post_window_hours: Wie weit nach Trade-Open (z.B. wenn innerhalb Position gehalten).
            decay_halflife_hours: Exponential-Decay für zeitlichen Abstand.
        """
        self.pre = pre_window_hours
        self.post = post_window_hours
        self.halflife = decay_halflife_hours

    def link_trade_to_events(
        self,
        trade: dict,
        news_events: pd.DataFrame,
    ) -> list[NewsLink]:
        """Für einen einzelnen Trade: finde zeitnahe Events.

        Trade muss haben: symbol, opened_at (ISO-timestamp).
        news_events DataFrame braucht: event_id, symbol (oder tickers liste),
                                       published_at, impact_bps (optional).
        """
        try:
            symbol = trade.get("symbol")
            opened_raw = trade.get("opened_at") or trade.get("closed_at")
            if not symbol or not opened_raw:
                return []
            opened = pd.Timestamp(opened_raw)
            if opened.tz is None:
                opened = opened.tz_localize("UTC")
        except Exception:
            return []

        if news_events is None or news_events.empty:
            return []

        # Zeitfenster
        window_start = opened - pd.Timedelta(hours=self.pre)
        window_end = opened + pd.Timedelta(hours=self.post)

        events = news_events.copy()
        if "published_at" not in events.columns:
            return []

        events["_pub_ts"] = pd.to_datetime(events["published_at"], errors="coerce", utc=True)
        events = events.dropna(subset=["_pub_ts"])

        # Symbol-Match (entweder direkte 'symbol' oder 'tickers' Liste)
        if "symbol" in events.columns:
            sym_mask = events["symbol"] == symbol
        elif "tickers" in events.columns:
            sym_mask = events["tickers"].apply(
                lambda t: symbol in (t if isinstance(t, list) else [])
            )
        else:
            return []

        time_mask = (events["_pub_ts"] >= window_start) & (events["_pub_ts"] <= window_end)
        relevant = events[sym_mask & time_mask]
        if relevant.empty:
            return []

        links: list[NewsLink] = []
        for _, row in relevant.iterrows():
            try:
                pub_ts = row["_pub_ts"]
                dist_hours = float(abs((opened - pub_ts).total_seconds()) / 3600.0)
                decay = np.exp(-np.log(2) * dist_hours / max(1e-6, self.halflife))
                impact_bps = float(row.get("impact_bps", 0.0))
                # Base weight: decay × normalized-impact
                weight = float(decay * min(1.0, abs(impact_bps) / 100.0 if impact_bps else 0.5))
                # Estimated Contribution to closed return
                closed_ret = float(trade.get("closed_return", 0.0) or trade.get("pnl", 0.0))
                est_contrib = weight * closed_ret

                links.append(NewsLink(
                    event_id=str(row.get("event_id", "")),
                    symbol=symbol,
                    distance_hours=round(dist_hours, 2),
                    weight=round(weight, 4),
                    estimated_contribution=round(est_contrib, 6),
                    impact_bps=round(impact_bps, 2),
                ))
            except Exception:
                continue

        return links

    def attribute_trades(
        self,
        trades: list[dict],
        news_events: pd.DataFrame,
    ) -> list[TradeAttribution]:
        """Batch-Attribution für alle Trades."""
        attributions: list[TradeAttribution] = []
        for trade in trades:
            try:
                links = self.link_trade_to_events(trade, news_events)
                closed_return = float(trade.get("closed_return", 0.0) or trade.get("pnl", 0.0))
                total_attributed = sum(link.estimated_contribution for link in links)
                residual = closed_return - total_attributed

                attributions.append(TradeAttribution(
                    trade_id=str(trade.get("trade_id", trade.get("id", ""))),
                    symbol=str(trade.get("symbol", "")),
                    closed_return=round(closed_return, 6),
                    news_links=links,
                    residual_return=round(residual, 6),
                ))
            except Exception as exc:
                logger.debug("[NewsTradeAttr] trade failed: %s", exc)

        return attributions

    def enrich_learning_store(
        self,
        learning_store_path: Path,
        news_events_path: Path,
        output_path: Path | None = None,
    ) -> int:
        """Reichert learning_store.jsonl mit news_links Feld an.

        Liest alle Records, verknüpft mit News-Events, schreibt entweder
        in-place (output_path=None) oder zu neuem File.

        Returns:
            Anzahl enriched Records.
        """
        if not learning_store_path.exists() or not news_events_path.exists():
            return 0

        try:
            # Load news events — flexibles Format (JSONL oder parquet)
            if news_events_path.suffix == ".parquet":
                news_df = pd.read_parquet(news_events_path)
            elif news_events_path.suffix == ".jsonl":
                news_df = pd.read_json(news_events_path, lines=True)
            else:
                news_df = pd.read_json(news_events_path)
        except Exception as exc:
            logger.debug("[NewsTradeAttr] news load failed: %s", exc)
            return 0

        enriched_lines: list[str] = []
        n_enriched = 0

        with learning_store_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("closed_at"):
                        links = self.link_trade_to_events(rec, news_df)
                        if links:
                            rec["news_links"] = [
                                {
                                    "event_id": l.event_id,
                                    "distance_hours": l.distance_hours,
                                    "weight": l.weight,
                                    "estimated_contribution": l.estimated_contribution,
                                    "impact_bps": l.impact_bps,
                                }
                                for l in links
                            ]
                            n_enriched += 1
                    enriched_lines.append(json.dumps(rec))
                except Exception:
                    enriched_lines.append(line)

        target = output_path or learning_store_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(enriched_lines) + "\n", encoding="utf-8")
        logger.info(
            "[NewsTradeAttr] enriched %d Records in %s",
            n_enriched, target,
        )
        return n_enriched


__all__ = [
    "NewsLink",
    "TradeAttribution",
    "NewsTradeAttributor",
]
