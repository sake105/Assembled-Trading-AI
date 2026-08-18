"""Geschuetzter Merge fuer den operativen Preis-Cache (E-165/E-166-Lehren).

STRUKTUR (2026-08-17, Audit-Follow-up "sector_etf-Naht-Guard"):
Die drei Schutzschichten aus scripts/ops/prewarm_price_cache.merge_and_save
wurden hierher extrahiert als EINE Wahrheit fuer die Schreiber von
``output/aggregates/daily.parquet``. ANGESCHLOSSEN (Stand 2026-08-17):
prewarm_price_cache.merge_and_save und refresh_sector_etf_cache.refresh —
die beiden LIVE-Schreiber. NICHT angeschlossen (dormant: Panel-Builder tot,
EODHD 401): refresh_daily_cache_from_panel, refresh_daily_cache_from_eodhd,
backfill_adj_close. Ein Helper macht eine Invariante VERFUEGBAR, nicht
durchgesetzt — wer einen dormanten Schreiber reaktiviert, MUSS ihn hier
anschliessen (E-166/E-173 — der
17.08.-Vorfall entstand, weil Guards je Schreiber einzeln nachgeruestet
wurden und der naechste Schreiber sie wieder nicht hatte, E-166):

1. OVERLAP-RE-ADJUSTIERUNG: auch adjusted-zu-adjusted ist nicht nahtfrei,
   wenn zwischen den Adjustierungs-ANKERN der Quellen eine Corporate Action
   lag — die neue Reihe ist dann RUECKWIRKEND anders skaliert. Bei
   konstanter Overlap-Ratio != 1 wird der BESTAND des Symbols auf den neuen
   Anker reskaliert; bei inkonstanter Ratio wird das Symbol verworfen und
   gemeldet (Quellen erzaehlen verschiedene Split-Geschichten).
2. NAHT-GUARD (fail-closed): fuer Symbole OHNE per Overlap bewiesene
   Semantik wird die QUELLEN-NAHT geprueft (letzter Alt-Bar -> erster
   Neu-Bar); |Move| > seam_threshold bricht ab, OHNE zu schreiben. Echte
   historische Extremtage im Bestand (AAPL -51,8 % Sep-2000) blocken nicht.
3. ADJ_CLOSE-INVARIANTE, UNBEDINGT am Schreibpunkt: Spalte wird angelegt,
   falls sie fehlt, und NaN werden aus close gefuellt (E-170: eine bedingte
   Reparatur ist keine Invariante).

Der Helper ist PURE (kein I/O): Aufrufer laden/schreiben selbst und
entscheiden, wie sie ``dropped_symbols`` protokollieren.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

#: Preisspalten, die bei einer Anker-Reskalierung mitskaliert werden.
#: volume bewusst NICHT: die Split-Historie bleibt stueckseitig alt; ADV ueber
#: die Grenze ist dann unscharf (dokumentierter Tradeoff, F-auditor-7).
PRICE_COLS = ("open", "high", "low", "close", "adj_close")


@dataclass
class MergeResult:
    combined: pd.DataFrame
    dropped_symbols: list[str] = field(default_factory=list)
    rescaled: dict[str, float] = field(default_factory=dict)
    #: Zeilen ohne close, die am Schreibpunkt verworfen wurden (E-184).
    dropped_priceless_rows: int = 0


class SeamGuardError(RuntimeError):
    """Merge abgebrochen: Adjustierungs-Naht / korrupter Feed erkannt."""


def guarded_merge(
    existing: pd.DataFrame,
    new_df: pd.DataFrame,
    *,
    seam_threshold: float = 0.5,
    overlap_min: int = 5,
    overlap_spread_max: float = 0.02,
    rescale_warn: float = 0.25,
) -> MergeResult:
    """Merge new price rows into an existing panel with all three guards.

    Args:
        existing: Bestand (long format: timestamp, symbol, ohlcv[, adj_close]).
        new_df: Neue Zeilen derselben Struktur; Overlap mit dem Bestand ist
            ERWUENSCHT (er traegt die Semantik-Pruefung). adj_close darf fehlen.
        seam_threshold: |Tagesmove| an der Quellen-Naht, ab dem abgebrochen wird.
        overlap_min: Mindestzahl gemeinsamer Timestamps fuer die Ratio-Pruefung.
        overlap_spread_max: max. relative Abweichung der Ratio vom Median,
            damit sie als "konstant" gilt.
        rescale_warn: |Faktor-1|, ab dem ein LARGE-rescale-WARN geloggt wird
            (echte Splits sind legitim — laut, nicht blockend).

    Returns:
        MergeResult mit dedupliziertem ``combined`` (last-write-wins),
        verworfenen Symbolen und angewandten Reskalierungsfaktoren.

    Raises:
        SeamGuardError: fail-closed, wenn eine Quellen-Naht > seam_threshold
            fuer ein nicht-Overlap-verifiziertes Symbol erkannt wird.
    """
    existing = existing.copy() if existing is not None else pd.DataFrame()
    new_df = new_df.copy() if new_df is not None else pd.DataFrame()

    # E-184 (F-senior-2, 2026-08-18): Preislose Zeilen MUESSEN am INGRESS
    # fallen — vor Overlap-Ratio, Naht-Guard und Dedup. Stand der Filter am
    # Ende (erste Fassung dieses Fixes), gewann die NaN-Zeile das
    # last-write-wins des drop_duplicates und riss die GUTE Bestandszeile
    # desselben (symbol,timestamp) mit; zusaetzlich vergiftete NaN-close die
    # Ratio-Pruefung (median=NaN, "NaN > threshold" ist False), wodurch ein
    # Symbol ohne Beweis als "verified" galt und den fail-closed Naht-Guard
    # uebersprang — genau die E-165-Klasse, gegen die dieser Helper existiert.
    new_df, n_priceless = _drop_priceless_rows(new_df, source="new")
    result = MergeResult(combined=existing, dropped_priceless_rows=n_priceless)
    if new_df.empty and n_priceless:
        result.combined = _enforce_adj_close(existing)
        return result
    if new_df.empty:
        result.combined = _enforce_adj_close(existing)
        return result

    price_cols = [c for c in PRICE_COLS if c in existing.columns]
    verified: set[str] = set()

    if not existing.empty:
        for sym in new_df["symbol"].unique():
            old_s = existing[existing["symbol"] == sym].set_index("timestamp")
            new_s = new_df[new_df["symbol"] == sym].set_index("timestamp")
            common = old_s.index.intersection(new_s.index)
            if len(common) < overlap_min:
                continue  # kein belastbarer Overlap -> Naht-Guard entscheidet
            ratio = new_s.loc[common, "close"].astype(float) / old_s.loc[
                common, "close"
            ].astype(float)
            med = float(ratio.median())
            spread = float((ratio / med - 1).abs().max())
            if spread > overlap_spread_max:
                logger.warning(
                    "[price-merge] %s: overlap ratio NOT constant "
                    "(spread %.1f%%) — dropping its new rows",
                    sym,
                    spread * 100,
                )
                result.dropped_symbols.append(sym)
                continue
            verified.add(sym)
            if abs(med - 1.0) > 0.001:
                logger.info(
                    "[price-merge] %s: corporate action between anchors — "
                    "rescaling existing history by %.6f",
                    sym,
                    med,
                )
                if abs(med - 1.0) > rescale_warn:
                    logger.warning(
                        "[price-merge] %s: LARGE rescale factor %.4f over "
                        "%d rows — verify real split, not ticker recycling",
                        sym,
                        med,
                        int((existing["symbol"] == sym).sum()),
                    )
                mask = existing["symbol"] == sym
                for c in price_cols:
                    existing.loc[mask, c] = existing.loc[mask, c].astype(float) * med
                result.rescaled[sym] = med

    if result.dropped_symbols:
        new_df = new_df[~new_df["symbol"].isin(result.dropped_symbols)]

    # Naht-Guard VOR dem Combine (fail-closed; E-164-Lektion: Guard auf den
    # Inhalt, und er darf keine Nebenwirkung hinterlassen).
    bad: list[str] = []
    if not existing.empty:
        for sym in set(new_df["symbol"].unique()) - verified:
            old_ts = existing.loc[existing["symbol"] == sym, "timestamp"].max()
            if pd.isna(old_ts):
                continue  # neues Symbol: keine Naht
            after = new_df[
                (new_df["symbol"] == sym) & (new_df["timestamp"] > old_ts)
            ].sort_values("timestamp")
            if after.empty:
                continue
            old_close = float(
                existing.loc[
                    (existing["symbol"] == sym) & (existing["timestamp"] == old_ts),
                    "close",
                ].iloc[0]
            )
            first_new = float(after["close"].iloc[0])
            if old_close > 0 and abs(first_new / old_close - 1.0) > seam_threshold:
                bad.append(sym)
    if bad:
        raise SeamGuardError(
            f"[price-merge] MERGE ABORTED (seam guard): {len(bad)} symbol(s) "
            f"with |seam move| > {seam_threshold:.0%} — adjustment-basis "
            f"mismatch or corrupt feed. Cache unchanged. "
            f"Symbols: {sorted(bad)[:15]}"
        )

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["symbol", "timestamp"], keep="last"
    ).sort_values(["symbol", "timestamp"])

    result.combined = _enforce_adj_close(combined)
    return result


def _drop_priceless_rows(
    df: pd.DataFrame, *, source: str = "new"
) -> tuple[pd.DataFrame, int]:
    """Zeilen OHNE close verwerfen (E-184, Pilot-Diagnose 2026-08-18).

    Ein Vendor lieferte fuer neu aufgenommene Symbole (SH/SHY/VIXY/XLU/TDG)
    einen Bar MIT volume, aber mit NaN in OHLC. Diese Zeile ist wertlos und
    zugleich gefaehrlich: der Frische-Check sieht einen Bar vom letzten
    Handelstag (= "aktuell"), waehrend KEIN Preis existiert — die
    Order-Generierung kann nichts umrechnen und das Symbol faellt still
    aus (im Pilot: die Krisen-Hedges). Frische ohne Inhalt ist die
    E-180-Klasse; hier am INGRESS des gemeinsamen Schreibpunkts entfernt,
    damit KEIN Schreiber sie je wieder einschleust.

    Wirkt bewusst NUR auf die eingehenden Zeilen (``source="new"``), nicht
    auf den Bestand (F-senior-14): eine Bestandsbereinigung ist ein eigener,
    protokollierter Ops-Schritt und darf nicht als Nebenwirkung eines
    Merges passieren.
    """
    if df.empty or "close" not in df.columns:
        return df, 0
    bad = df["close"].isna()
    n = int(bad.sum())
    if n:
        syms = (
            sorted(set(df.loc[bad, "symbol"].astype(str)))
            if "symbol" in df.columns
            else []
        )
        logger.warning(
            "[price-merge] dropped %d incoming row(s) without close "
            "(volume-only bars, source=%s) for %d symbol(s): %s — a bar "
            "without price is not a bar",
            n,
            source,
            len(syms),
            syms[:10],
        )
        df = df.loc[~bad].copy()
    return df, n


def _enforce_adj_close(df: pd.DataFrame) -> pd.DataFrame:
    """adj_close == close-Invariante UNBEDINGT herstellen (E-170)."""
    if df.empty or "close" not in df.columns:
        return df
    if "adj_close" not in df.columns:
        df = df.copy()
        df["adj_close"] = df["close"]
        return df
    na = df["adj_close"].isna()
    if bool(na.any()):
        df = df.copy()
        df.loc[na, "adj_close"] = df.loc[na, "close"]
    return df
