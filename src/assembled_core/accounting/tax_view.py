# -*- coding: utf-8 -*-
"""Steuer-Sicht als PARALLELE Reporting-Schicht (Audit-Plan 5.5, 2026-08-17).

GRUNDENTSCHEIDUNG (Plan-Verdikt): Steuer ist eine rein LESENDE Schatten-
Sicht. ``position_engine`` (Average-Cost) bleibt autoritativ fuer
operatives PnL/Cash/Reconcile; FIFO ist ausschliesslich die STEUER-Sicht
(deutsches Recht: FIFO, §20 EStG-Reihenfolge via ``tax_regime_de``).
Da ``trades_df`` im EOD-Pfad jeden Lauf vollstaendig neu entsteht
(stateless Voll-Replay), wird auch diese Sicht pro Lauf STATELESS aus der
vollen Trade-Historie berechnet — bewusst KEIN ``TaxLotStore``-SQLite im
EOD-Pfad (Doppel-Ingest-Risiko), sondern In-Memory-Replay mit
``tax_lots.match_fifo``.

INVARIANTEN (Rule 30, testgepinnt in tests/test_tax_view_integration.py):
- Kein Schreibzugriff auf Orders/Fills/Ledger/State — Input wird kopiert.
- Ein Fehler hier darf NIE einen Pipeline-Lauf scheitern lassen (der
  Orchestrator-Hook ist best-effort und default-aus).

EHRLICHE GRENZEN (im Report als ``notes`` sichtbar, nie still):
- Dividenden: es gibt keine DIVIDEND-Ledger-Events — die Anlage-KAP-
  Dividendenzeile bleibt leer und wird als Luecke ausgewiesen.
- Shorts: ``match_fifo`` ist long-only; ein Sell ohne ausreichende Lots
  landet in ``over_close_qty`` (Sichtbarkeitspflicht, kein stilles Clipping).
- FX: Trades sind USD (US-Universum via Alpaca/yfinance); ohne injizierte
  Kurse gilt der Fallback laut ``fx_source`` — nie stillschweigend.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.assembled_core.accounting.tax_lots import TaxLot, match_fifo
from src.assembled_core.accounting.tax_regime_de import AssetClass, PrivatDE

logger = logging.getLogger(__name__)

#: Dokumentierter Konservativ-Fallback (identisch zu tax_lots-Modul); jede
#: Nutzung wird via fx_source="fallback" im Report ausgewiesen (E-180-Klasse:
#: stille Naeherungen sind verboten, sichtbare erlaubt).
FALLBACK_USD_EUR = 0.93


@dataclass
class TaxYearSummary:
    year: int
    gains_eur: float = 0.0  # Summe realisierter Gewinne (brutto, > 0)
    losses_eur: float = 0.0  # Summe realisierter Verluste (als positiver Betrag)
    loss_pot_start: float = 0.0
    loss_pot_end: float = 0.0
    pauschbetrag_used: float = 0.0
    tax_eur: float = 0.0
    n_closes: int = 0


@dataclass
class TaxViewResult:
    years: dict[int, TaxYearSummary] = field(default_factory=dict)
    over_close_qty: dict[str, float] = field(default_factory=dict)
    fx_source: str = "injected"
    open_lots_end: int = 0
    n_fills_replayed: int = 0
    notes: list[str] = field(default_factory=list)


def _fx_rate_for(
    d: date,
    fx_rates: dict[date, float] | None,
    fallback_rate: float,
) -> tuple[float, bool]:
    """Rate fuer ein Datum: injiziert (exakt oder juengster Vortag) sonst Fallback."""
    if fx_rates:
        if d in fx_rates:
            return float(fx_rates[d]), False
        prior = [k for k in fx_rates if k <= d]
        if prior:
            return float(fx_rates[max(prior)]), False
    return float(fallback_rate), True


def build_tax_view_from_trades(
    trades_df: pd.DataFrame,
    *,
    fx_rates: dict[date, float] | None = None,
    fallback_rate: float = FALLBACK_USD_EUR,
    regime: PrivatDE | None = None,
    asset: AssetClass = AssetClass.AKTIE,
) -> TaxViewResult:
    """Replay filled trades chronologically into a FIFO tax view (EUR).

    Args:
        trades_df: Fills im Ledger-Schema (timestamp, symbol, side, fill_qty,
            fill_price, status[, commission_cash]). Nur ``status == "filled"``
            wird beruecksichtigt. Input bleibt unveraendert.
        fx_rates: date -> USD/EUR-Kurs (1 USD = x EUR). Fehlt ein Datum, gilt
            der juengste fruehere injizierte Kurs, sonst ``fallback_rate``.
        fallback_rate: Konservativ-Fallback; Nutzung wird ausgewiesen.
        regime: Steuerregime (Default: frisches ``PrivatDE``).
        asset: AssetClass fuer den Satz (Pilot: Einzelaktien = AKTIE).

    Returns:
        TaxViewResult mit Jahres-Summen, Verlusttopf-Staenden und Anomalien.
    """
    result = TaxViewResult()
    regime = regime or PrivatDE()
    if trades_df is None or trades_df.empty:
        result.notes.append("no trades supplied — empty tax view")
        return result

    df = trades_df.copy()
    required = {"timestamp", "symbol", "side", "fill_qty", "fill_price"}
    missing = required - set(df.columns)
    if missing:
        result.notes.append(f"missing columns {sorted(missing)} — empty tax view")
        return result
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "filled"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")

    used_fallback = False
    open_lots: dict[str, list[TaxLot]] = {}
    cur_year: int | None = None

    def _year(y: int) -> TaxYearSummary:
        nonlocal cur_year
        if y not in result.years:
            result.years[y] = TaxYearSummary(
                year=y, loss_pot_start=round(regime.loss_pot, 4)
            )
        if cur_year != y:
            cur_year = y
            regime.new_year(y)
        return result.years[y]

    for row in df.itertuples(index=False):
        ts: pd.Timestamp = row.timestamp
        d = ts.date()
        sym = str(row.symbol).upper()
        qty = float(row.fill_qty or 0.0)
        px = float(row.fill_price or 0.0)
        if qty <= 0 or px <= 0:
            continue
        fee = float(getattr(row, "commission_cash", 0.0) or 0.0)
        rate, was_fb = _fx_rate_for(d, fx_rates, fallback_rate)
        used_fallback = used_fallback or was_fb
        ysum = _year(d.year)
        side = str(row.side).lower()
        result.n_fills_replayed += 1

        if side == "buy":
            open_lots.setdefault(sym, []).append(
                TaxLot.open_lot(
                    symbol=sym,
                    qty=qty,
                    price_usd=px,
                    usd_eur_rate=rate,
                    trade_date=d,
                    trade_timestamp=ts.to_pydatetime(),
                    fees_usd=fee,
                )
            )
            continue

        # SELL: FIFO gegen die offenen In-Memory-Lots.
        lots = open_lots.get(sym, [])
        close = match_fifo(lots, qty, px, rate, d)
        # match_fifo mutiert die Lots NICHT — hier fortschreiben.
        matched = {c["lot_id"]: float(c["qty"]) for c in close.lots_closed}
        still_open: list[TaxLot] = []
        for lot in lots:
            m = matched.get(lot.id, 0.0)
            if m <= 0:
                still_open.append(lot)
                continue
            if m < lot.qty:
                # Anteilige Fee-Reduktion analog TaxLotStore.partial-close.
                frac_left = (lot.qty - m) / lot.qty
                lot.fees_usd *= frac_left
                lot.fees_eur *= frac_left
                lot.qty -= m
                still_open.append(lot)
        open_lots[sym] = still_open

        if close.qty_remaining > 0:
            result.over_close_qty[sym] = (
                result.over_close_qty.get(sym, 0.0) + close.qty_remaining
            )

        # Exit-Fee mindert den Gewinn (Regel aus mandat2/portfolio.py:141) —
        # match_fifo zieht nur die ENTRY-Fees des Lots ab.
        pnl_eur = close.total_pnl_eur - fee * rate
        if close.lots_closed:
            ysum.n_closes += 1
            if pnl_eur >= 0:
                ysum.gains_eur += pnl_eur
            else:
                ysum.losses_eur += -pnl_eur
            pb_before = regime.pauschbetrag_left
            ysum.tax_eur += regime.on_realized_gain(pnl_eur, asset)
            ysum.pauschbetrag_used += pb_before - regime.pauschbetrag_left
            ysum.loss_pot_end = round(regime.loss_pot, 4)

    # F-senior-10: loss_pot_end wurde nur in Jahren mit Close gesetzt — ein
    # Jahr ohne Verkauf meldete 0.0 und widersprach dem Vortrag des
    # Folgejahres. Chronologisch fortschreiben.
    _carry = 0.0
    for _y in sorted(result.years):
        _s = result.years[_y]
        if _s.n_closes == 0:
            _s.loss_pot_start = round(_carry, 4)
            _s.loss_pot_end = round(_carry, 4)
        _carry = _s.loss_pot_end
    for ysum in result.years.values():
        ysum.gains_eur = round(ysum.gains_eur, 4)
        ysum.losses_eur = round(ysum.losses_eur, 4)
        ysum.tax_eur = round(ysum.tax_eur, 4)
        ysum.pauschbetrag_used = round(ysum.pauschbetrag_used, 4)
        ysum.loss_pot_end = (
            round(regime.loss_pot, 4) if ysum.year == cur_year else ysum.loss_pot_end
        )

    result.open_lots_end = sum(len(v) for v in open_lots.values())
    result.fx_source = (
        "fallback"
        if (used_fallback and not fx_rates)
        else ("mixed" if used_fallback else "injected")
    )
    result.notes.append(
        "dividends: no DIVIDEND ledger events exist — Anlage-KAP dividend "
        "line intentionally empty (documented gap, not silently missing)"
    )
    if result.over_close_qty:
        result.notes.append(
            f"over_close (sell > open lots, match_fifo is long-only): "
            f"{result.over_close_qty} — tax view incomplete for these symbols"
        )
    return result


def write_tax_view_json(
    result: TaxViewResult,
    output_dir: Path,
    run_id: str,
    *,
    trades_source: str = "unspecified",
) -> Path:
    """Persist the tax view deterministically as one JSON per run.

    ``trades_source`` (S1-N3a): woher die Fills stammen — eine Sicht aus
    SIMULATIONS-Trades (Orchestrator Step 4c) ist keine Anlage-KAP-Basis;
    dafuer muss build_tax_report.py mit echten Ledger-Fills laufen. Das
    Feld macht die Verwechslung im Artefakt selbst unmoeglich.
    """
    out_dir = Path(output_dir) / f"tax_report_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "tax.view.v1",
        "trades_source": trades_source,
        "regime": "PRIVAT_DE (26.375 %, Pauschbetrag 1000/J, Verlustvortrag)",
        # F-senior-4 (E-187): NIE literal. Ein Qualitaetssiegel muss aus den
        # Bedingungen folgen, die es behauptet: echte Ledger-Fills UND echte
        # FX-Kurse UND ueberhaupt verarbeitete Fills. Eine Sicht aus
        # Simulations-Trades oder mit Platzhalter-FX ist keine
        # Anlage-KAP-Basis.
        "authoritative": bool(
            trades_source.startswith("ledger")
            and result.fx_source != "fallback"
            and result.n_fills_replayed > 0
        ),
        "fx_source": result.fx_source,
        "n_fills_replayed": result.n_fills_replayed,
        "open_lots_end": result.open_lots_end,
        "over_close_qty": result.over_close_qty,
        "notes": result.notes,
        "years": {
            str(y): {
                "gains_eur": s.gains_eur,
                "losses_eur": s.losses_eur,
                "loss_pot_start": s.loss_pot_start,
                "loss_pot_end": s.loss_pot_end,
                "pauschbetrag_used": s.pauschbetrag_used,
                "tax_eur": s.tax_eur,
                "n_closes": s.n_closes,
            }
            for y, s in sorted(result.years.items())
        },
    }
    out = out_dir / "tax_view.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("[tax-view] wrote %s (%d years)", out, len(result.years))
    return out
