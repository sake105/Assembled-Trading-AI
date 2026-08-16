# -*- coding: utf-8 -*-
"""Taeglicher Attribution- und Signal-Score-Report (Audit-Plan 5.3, 2026-08-16).

WOZU: Das attribution/-Paket (Brinson/Composite-Attribution + SQLite-Store +
Zeitreihen-Diagnostik) war fertig und getestet, hatte aber KEINEN Producer im
aktiven Pfad — und der API-Endpoint /monitoring/signals suchte
signal_scores_*.json, die niemand schrieb (Nutzungsaudit
docs/DATEN_UND_NUTZUNGSAUDIT.md §3, Kategorie (b)). Dieses Script schliesst
beide Luecken in einem Lauf:

  1. Operatives Panel laden (output/aggregates/daily.parquet), TA-Features
     rechnen, 9-Dimensionen-Composite-Score je Watchlist-Symbol.
  2. Attributionen in den AttributionStore persistieren
     (output/attribution/attributions.db, append-only, WAL).
  3. output/signals/signal_scores_<ts>.json schreiben — der Producer fuer
     /monitoring/signals (Default-Dir des Endpoints zeigt seit 5.2 hierauf).
  4. output/attribution/attribution_report_<date>.json mit Top-Beitraegen und
     — sobald genug Historie im Store liegt — Dead-Feature-Diagnostik aus
     attribution.time_series.

Aufrufer: scripts/daily_paper_trading.bat Step 3 (non-fatal, nach dem
Pilot-Lauf) oder manuell. Read-only gegenueber dem Preis-Cache.

EHRLICHE GRENZEN: --regime ist fix 'normal' (der taegliche Lauf liest NICHT
das reale Pipeline-Regime — Scores koennen von den Pipeline-Scores abweichen;
im Artefakt via 'regime'-Feld sichtbar). Dimensionen ohne Panel-Daten
(IV, Intraday, News) stehen
neutral auf 0.0; chart_pattern ist ein dokumentierter 0.0-Stub. Die
Attribution misst also den REAL wirksamen Teil des Composite-Scores — genau
das ist ihr Zweck.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd

SIGNALS_DIR = _REPO / "output" / "signals"
ATTR_DIR = _REPO / "output" / "attribution"
PANEL_PATH = _REPO / "output" / "aggregates" / "daily.parquet"
WATCHLIST = _REPO / "watchlist.txt"  # Repo-Root, wie prewarm_price_cache

#: Mindest-Tage im Store, bevor die Zeitreihen-Diagnostik gerechnet wird.
MIN_DAYS_FOR_TIMESERIES = 10


def _load_watchlist() -> list[str]:
    if not WATCHLIST.exists():
        return []
    return [
        line.strip()
        for line in WATCHLIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lookback-days", type=int, default=120)
    ap.add_argument("--regime", default="normal")
    args = ap.parse_args()

    if not PANEL_PATH.exists():
        print(f"[ERROR] Panel fehlt: {PANEL_PATH}")
        return 1

    symbols = _load_watchlist()
    if not symbols:
        print(f"[ERROR] Watchlist leer/fehlt: {WATCHLIST}")
        return 1

    df = pd.read_parquet(PANEL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=args.lookback_days)
    df = df[(df["timestamp"] >= cutoff) & (df["symbol"].isin(symbols))].copy()
    if df.empty:
        print("[ERROR] Panel-Slice leer (Watchlist vs. Panel-Symbole pruefen)")
        return 1

    from src.assembled_core.features.ta_features import add_all_features
    from src.assembled_core.signals.composite_score import (
        generate_composite_score_signals,
    )

    feat = add_all_features(df)

    attributions: list = []
    signals = generate_composite_score_signals(
        feat, regime=args.regime, collect_attributions=attributions
    )
    if signals.empty:
        print("[ERROR] Keine Signale erzeugt (Historie zu kurz?)")
        return 1

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    # F-senior-4: das As-of der DATEN gehoert in jedes Artefakt — Scores aus
    # einem eingefrorenen Panel duerfen nie als aktuell durchgereicht werden.
    data_as_of = str(df["timestamp"].max())

    # --- 1) Persistenz in den AttributionStore ---
    # Stagnations-Guard (F-senior-9): Attributionen tragen jetzt das
    # Bar-Datum. Hat der letzte Lauf dasselbe data_as_of gesehen (Panel
    # eingefroren / Wochenende), wuerde jeder weitere Save nur Duplikate
    # erzeugen und die IC-Diagnostik verzerren -> Save ueberspringen.
    from src.assembled_core.attribution.storage import AttributionStore

    store = AttributionStore(db_path=str(ATTR_DIR / "attributions.db"))
    marker_path = ATTR_DIR / "last_data_as_of.txt"
    prev_as_of = (
        marker_path.read_text(encoding="utf-8").strip()
        if marker_path.exists()
        else None
    )
    saved = 0
    if prev_as_of == data_as_of:
        print(f"[SKIP] Store-Save: data_as_of unveraendert ({data_as_of})")
    else:
        for attr in attributions:
            store.save(attr)
        saved = len(attributions)
        marker_path.write_text(data_as_of, encoding="utf-8")

    # --- 2) signal_scores-Producer fuer /monitoring/signals ---
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    scores = {row["symbol"]: float(row["score"]) for _, row in signals.iterrows()}
    scores_path = SIGNALS_DIR / f"signal_scores_{stamp}.json"
    scores_path.write_text(
        json.dumps(
            {
                "generated_at": now.isoformat(),
                "data_as_of": data_as_of,
                "regime": args.regime,
                "producer": "scripts/generate_attribution_report.py",
                "n_panel_symbols": int(df["symbol"].nunique()),
                "scores": scores,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    # --- 3) Attribution-Report ---
    from src.assembled_core.attribution.composite import attribution_to_dict

    # F-senior-12: quantifizieren, welcher Gewichtsanteil an diesem Tag
    # inert war (Dimensionen, die ueber ALLE Symbole exakt 0.0 lieferten).
    inert_dims: list[str] = []
    inert_weight = 0.0
    if attributions:
        dims = attributions[0].dimension_raw_scores.keys()
        for dim in dims:
            if all(a.dimension_raw_scores.get(dim, 0.0) == 0.0 for a in attributions):
                inert_dims.append(dim)
                inert_weight += attributions[0].dimension_weights.get(dim, 0.0)

    report: dict = {
        "generated_at": now.isoformat(),
        "data_as_of": data_as_of,
        "store_rows_saved": saved,
        "inert_dimensions": sorted(inert_dims),
        "inert_weight_share": round(inert_weight, 4),
        "regime": args.regime,
        "n_symbols": len(attributions),
        "top_positive": sorted(
            (attribution_to_dict(a) for a in attributions),
            key=lambda d: -d["composite_score"],
        )[:10],
        "top_negative": sorted(
            (attribution_to_dict(a) for a in attributions),
            key=lambda d: d["composite_score"],
        )[:10],
    }

    # Zeitreihen-Diagnostik erst mit genug Historie (ehrlich: vorher waere
    # jede IC-/Dead-Feature-Aussage Rauschen). Kette: Store-Historie je
    # Watchlist-Symbol -> attributions_to_df -> rolling IC gegen 1d-Forward-
    # Returns aus dem Panel -> Dead-Feature-Report.
    try:
        history: list = []
        for sym in symbols:
            history.extend(store.load_for_ticker(sym))
        n_days = (
            pd.to_datetime(pd.Series([a.timestamp for a in history])).dt.date.nunique()
            if history
            else 0
        )
        report["timeseries_days"] = int(n_days)
        if n_days >= MIN_DAYS_FOR_TIMESERIES:
            from src.assembled_core.attribution.time_series import (
                attributions_to_df,
                dead_feature_report,
                detect_dead_features,
                rolling_dimension_ic,
            )

            adf = attributions_to_df(history)
            px = df.sort_values("timestamp").copy()
            px["date"] = px["timestamp"].dt.normalize()
            # pct_change UND shift muessen beide GRUPPIERT laufen — ein
            # ungruppiertes shift(-1) zoege den ersten Return des naechsten
            # Symbols ueber die Gruppengrenze.
            px["fwd"] = px.groupby("symbol")["close"].transform(
                lambda s: s.pct_change().shift(-1)
            )
            fwd = (
                px.dropna(subset=["fwd"])
                .rename(columns={"symbol": "ticker"})
                .set_index(["date", "ticker"])["fwd"]
            )
            ic_df = rolling_dimension_ic(adf, fwd)
            dead = detect_dead_features(ic_df)
            report["dead_features"] = dead_feature_report(dead)
        else:
            report["timeseries_note"] = (
                f"diagnostics start at {MIN_DAYS_FOR_TIMESERIES} distinct days"
            )
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        # Diagnostik ist Zusatz — ihr Ausfall darf den Report nicht kosten,
        # aber er muss SICHTBAR im Artefakt stehen (kein stilles except).
        report["timeseries_error"] = str(exc)

    report_path = ATTR_DIR / f"attribution_report_{now.date().isoformat()}.json"
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(
        f"[OK] {len(attributions)} Attributionen ({saved} gespeichert) -> {store.db_path.name} | "
        f"scores -> {scores_path.name} | report -> {report_path.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
