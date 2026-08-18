# -*- coding: utf-8 -*-
"""Regressionstests der Pilot-Diagnose 2026-08-18.

Der Paper-Pilot erzeugte trotz gueltiger Signale NULL Orders. Zwei
unabhaengige Defekte, beide hier gepinnt:

1. E-185 — ``trend_baseline.compute_target_positions`` lieferte
   ``target_qty=0.0`` und verwies auf einen nachgelagerten Konverter, den es
   nie gab (``generate_orders_from_targets`` verlangt target_qty als NOTIONAL
   und liest target_weight nie). Jede Kernposition hatte damit Delta 0.
2. E-184 — der Preis-Cache nahm Bars MIT volume, aber mit NaN-OHLC auf. Der
   Frische-Check sah einen aktuellen Bar, es gab aber keinen Preis: die
   betroffenen Symbole (die Krisen-Hedges SH/SHY/VIXY/XLU) fielen still aus.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.data.price_cache_merge import guarded_merge
from src.assembled_core.strategies.trend_baseline import compute_target_positions

pytestmark = pytest.mark.fast


def _signals(n: int = 5) -> pd.DataFrame:
    syms = [f"S{i}" for i in range(n)]
    return pd.DataFrame(
        {"symbol": syms, "direction": ["LONG"] * n, "score": list(range(n, 0, -1))}
    )


def test_trend_baseline_derives_notional_target_qty():
    """E-185-Pin: target_qty MUSS das Notional tragen (weight * capital).

    Mit target_qty=0 erzeugt generate_orders_from_targets Delta 0 -> keine
    Order. Genau so stand der Pilot: 10 gueltige Ziele, 0 Orders.
    """
    out = compute_target_positions(
        _signals(5), 100_000.0, max_positions=5, target_invested_pct=0.8
    )
    assert len(out) == 5
    assert out["target_weight"].tolist() == pytest.approx([0.16] * 5)
    assert out["target_qty"].tolist() == pytest.approx([16_000.0] * 5)
    assert out["target_qty"].sum() == pytest.approx(80_000.0)  # = 80 % des Kapitals


def test_trend_baseline_qty_scales_with_capital():
    small = compute_target_positions(_signals(4), 10_000.0, target_invested_pct=1.0)
    big = compute_target_positions(_signals(4), 20_000.0, target_invested_pct=1.0)
    assert small["target_qty"].sum() == pytest.approx(10_000.0)
    assert big["target_qty"].sum() == pytest.approx(20_000.0)
    # Gewichte bleiben kapitalunabhaengig — nur das Notional skaliert.
    assert small["target_weight"].tolist() == big["target_weight"].tolist()


def test_trend_baseline_zero_capital_is_zero_notional():
    """Kapital 0 -> Notional 0 (kein Division-Artefakt, kein NaN)."""
    out = compute_target_positions(_signals(3), 0.0, target_invested_pct=0.8)
    assert (out["target_qty"] == 0.0).all()
    assert out["target_weight"].sum() == pytest.approx(0.8)


def _bar(ts: str, sym: str, close: float | None, volume: float = 1000.0) -> dict:
    return {
        "timestamp": pd.Timestamp(ts, tz="UTC"),
        "symbol": sym,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": volume,
    }


def test_merge_drops_volume_only_bars_without_price():
    """E-184-Pin: ein Bar MIT volume aber OHNE close darf nie in den Cache.

    Er macht den Frische-Check gruen (Bar vom letzten Handelstag), waehrend
    die Order-Generierung mangels Preis nichts umrechnen kann — das Symbol
    faellt still aus.
    """
    existing = pd.DataFrame([_bar("2026-08-13", "SH", 32.0)])
    new = pd.DataFrame(
        [_bar("2026-08-14", "SH", 31.9), _bar("2026-08-17", "SH", None, 3_411_195.0)]
    )
    res = guarded_merge(existing, new)
    assert res.dropped_priceless_rows == 1
    assert res.combined["close"].isna().sum() == 0
    assert res.combined["timestamp"].max() == pd.Timestamp("2026-08-14", tz="UTC")


def test_merge_keeps_all_priced_bars():
    existing = pd.DataFrame([_bar("2026-08-13", "AAA", 10.0)])
    new = pd.DataFrame([_bar("2026-08-14", "AAA", 10.5)])
    res = guarded_merge(existing, new)
    assert res.dropped_priceless_rows == 0
    assert len(res.combined) == 2
    assert (res.combined["adj_close"] == res.combined["close"]).all()


def test_priceless_row_does_not_delete_good_row_on_key_collision():
    """F-senior-2-Pin (E-184): DER Fall, der die erste Fassung des Guards
    entlarvte — NaN-Zeile und gute Zeile auf demselben (symbol,timestamp).

    Stand der Filter hinter drop_duplicates(keep='last'), gewann die
    NaN-Zeile das last-write-wins und riss beim Verwerfen die valide
    Bestandszeile mit: der Cache verlor echte Historie.
    """
    existing = pd.DataFrame(
        [_bar("2026-08-16", "SH", 31.5), _bar("2026-08-17", "SH", 32.0)]
    )
    new = pd.DataFrame([_bar("2026-08-17", "SH", None, 3_411_195.0)])
    res = guarded_merge(existing, new)
    assert res.dropped_priceless_rows == 1
    kept = res.combined[
        res.combined["timestamp"] == pd.Timestamp("2026-08-17", tz="UTC")
    ]
    assert len(kept) == 1
    assert kept["close"].iloc[0] == pytest.approx(32.0)  # gute Zeile UEBERLEBT
    assert len(res.combined) == 2


def test_nan_close_in_overlap_cannot_bypass_seam_guard():
    """F-senior-2-Pin, zweite Haelfte: NaN im Overlap darf die
    Ratio-Pruefung nicht vergiften (median=NaN -> 'NaN > threshold' ist
    False -> Symbol galt als 'verified' und uebersprang den fail-closed
    Naht-Guard). Ein 10x-Sprung MUSS weiterhin abbrechen."""
    import pytest as _pytest

    from src.assembled_core.data.price_cache_merge import SeamGuardError

    existing = pd.DataFrame(
        [_bar(f"2026-08-{d:02d}", "XX", 10.0) for d in range(3, 15)]
    )
    # Neue Reihe: Overlap-Tage NUR mit NaN (wertlos) + ein 10x-Sprung danach.
    new = pd.DataFrame(
        [_bar(f"2026-08-{d:02d}", "XX", None, 500.0) for d in range(3, 15)]
        + [_bar("2026-08-17", "XX", 100.0)]
    )
    with _pytest.raises(SeamGuardError):
        guarded_merge(existing, new)


def test_crisis_hedges_are_excluded_from_core_signals():
    """F-senior-8-Pin: die Hedges liegen im PREIS-Frame (sonst werden ihre
    Overlay-Ziele nie zu Orders), duerfen aber keine Core-Signale erzeugen —
    sonst geht der Core genau im Krisenfall LONG in dieselben Instrumente,
    die das Overlay kauft (Doppelallokation).
    """
    from src.assembled_core.events.crisis_alpha.baskets import get_basket_symbols
    from src.assembled_core.ops.paper_runner import _prd_make_strategy_fns

    hedges = set(get_basket_symbols())
    assert hedges, "crisis baskets liefern keine Symbole — Test waere blind"

    signal_fn, _sizing = _prd_make_strategy_fns("trend_baseline", {}, None)

    # Preis-Frame MIT Hedges: steigender Trend fuer alle -> jedes Symbol
    # waere ohne den Ausschluss ein LONG-Kandidat.
    days = pd.bdate_range("2026-01-02", periods=90, tz="UTC")
    rows = []
    for sym in ["AAA", "BBB", *sorted(hedges)]:
        for i, t in enumerate(days):
            rows.append(
                {"timestamp": t, "symbol": sym, "close": 100.0 + i * 0.7, "volume": 1e6}
            )
    out = signal_fn(pd.DataFrame(rows))
    if not out.empty and "symbol" in out.columns:
        leaked = hedges & set(out["symbol"].astype(str))
        assert not leaked, f"Hedge-Symbole im Core-Signal: {sorted(leaked)}"
