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

        # Avoid underscore-prefix column (_pub_ts) which itertuples silently renames
        events["pub_ts_"] = pd.to_datetime(events["published_at"], errors="coerce", utc=True)
        events = events.dropna(subset=["pub_ts_"])

        # Symbol-Match (entweder direkte 'symbol' oder 'tickers' Liste)
        if "symbol" in events.columns:
            sym_mask = events["symbol"] == symbol
        elif "tickers" in events.columns:
            sym_mask = events["tickers"].apply(
                lambda t: symbol in (t if isinstance(t, list) else [])
            )
        else:
            return []

        time_mask = (events["pub_ts_"] >= window_start) & (events["pub_ts_"] <= window_end)
        relevant = events[sym_mask & time_mask]
        if relevant.empty:
            return []

        # Vectorize decay computation — avoids itertuples underscore-rename bug
        pub_ts_series = relevant["pub_ts_"]
        dist_hours = (opened - pub_ts_series).abs().dt.total_seconds() / 3600.0
        decay = np.exp(-np.log(2) * dist_hours / max(1e-6, self.halflife))
        if "impact_bps" in relevant.columns:
            impact_bps_s = relevant["impact_bps"].fillna(0.0)
        else:
            impact_bps_s = pd.Series(0.0, index=relevant.index)
        weight = decay * (impact_bps_s.abs() / 100.0).clip(upper=1.0)
        weight = weight.where(impact_bps_s != 0, decay * 0.5)
        closed_ret = float(trade.get("closed_return", 0.0) or trade.get("pnl", 0.0))
        est_contrib = weight * closed_ret
        event_id_s = (
            relevant["event_id"].fillna("").astype(str)
            if "event_id" in relevant.columns
            else pd.Series("", index=relevant.index)
        )

        links: list[NewsLink] = []
        for idx in relevant.index:
            links.append(NewsLink(
                event_id=event_id_s[idx],
                symbol=symbol,
                distance_hours=round(float(dist_hours[idx]), 2),
                weight=round(float(weight[idx]), 4),
                estimated_contribution=round(float(est_contrib[idx]), 6),
                impact_bps=round(float(impact_bps_s[idx]), 2),
            ))
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
        reenrich: bool = False,
    ) -> int:
        """Reichert learning_store.jsonl mit news_links Feld an.

        Liest alle Records, verknüpft mit News-Events, schreibt entweder
        in-place (output_path=None) oder zu neuem File.

        Idempotent: Records mit bestehendem ``news_links`` werden übersprungen,
        es sei denn ``reenrich=True``. Writes erfolgen via tmp-file + rename,
        damit der Store bei Abbruch nicht korrupt zurückbleibt.

        Args:
            learning_store_path: JSONL-Pfad mit Trade-Records.
            news_events_path: Pfad zu News-Events (JSONL/JSON/Parquet).
            output_path: Ziel-Pfad. None = in-place.
            reenrich: True überschreibt vorhandene news_links. Default False.

        Returns:
            Anzahl neu enriched Records (ohne bereits-enriched, wenn
            ``reenrich=False``).
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
        n_skipped = 0

        with learning_store_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("closed_at"):
                        if rec.get("news_links") and not reenrich:
                            n_skipped += 1
                        else:
                            links = self.link_trade_to_events(rec, news_df)
                            if links:
                                rec["news_links"] = [
                                    {
                                        "event_id": lnk.event_id,
                                        "distance_hours": lnk.distance_hours,
                                        "weight": lnk.weight,
                                        "estimated_contribution": lnk.estimated_contribution,
                                        "impact_bps": lnk.impact_bps,
                                    }
                                    for lnk in links
                                ]
                                n_enriched += 1
                    enriched_lines.append(json.dumps(rec))
                except Exception as _exc:
                    logger.debug("[NewsTradeAttr] enrich_learning_store line skipped: %s", _exc)
                    enriched_lines.append(line)

        target = output_path or learning_store_path
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp + rename
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            tmp.write_text("\n".join(enriched_lines) + "\n", encoding="utf-8")
            tmp.replace(target)
        except Exception as exc:
            logger.warning("[NewsTradeAttr] atomic write failed: %s", exc)
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return 0
        logger.info(
            "[NewsTradeAttr] enriched %d neu, %d bereits-enriched übersprungen → %s",
            n_enriched, n_skipped, target,
        )
        return n_enriched


__all__ = [
    "NewsLink",
    "TradeAttribution",
    "NewsTradeAttributor",
]
