"""H-089 Forward-Shadow — taeglicher Dial-Rechner (rechnet, handelt NICHTS).

Zweck: Ab heute sammelt jeder Lauf den PIT-Beweis fuer den H-089-Kandidaten
(mom12-Dial + 3er-Chor auf SPY). Der Shadow schreibt NUR den heutigen
Zustand (Signale, Risiko, Exposure) ins Log — er berechnet KEINE
historische Kurve (das Fenster 2017+ ist versiegelter Holdout; die
Trailing-Daten dienen ausschliesslich der Signalberechnung, nicht der
Bewertung). Bewertet wird der Shadow erst prospektiv aus dem Log selbst.

Betrieb: 1x taeglich nach Boersenschluss (manuell oder Task-Scheduler —
Operator-Entscheidung). Idempotent je Handelstag. Quelle: Haus-EOD-Store
(load_eod_prices, SPY ab 2018).
Log: research/strategie_n1/h089_shadow_log.jsonl (append-only, committbar).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.prices_ingest import load_eod_prices  # noqa: E402

LOG = HIER / "h089_shadow_log.jsonl"
DIAL = 0.6


def main() -> int:
    df = load_eod_prices(["SPY"])
    px = (
        df[df["symbol"] == "SPY"]
        .set_index("timestamp")["close"]
        .sort_index()
        .astype(float)
    )
    if len(px) < 260:
        raise SystemExit(f"[ERROR] zu wenig Historie fuer mom12: {len(px)} Zeilen")
    letzter_tag = str(px.index[-1].date())

    bekannt = set()
    if LOG.exists():
        for zeile in LOG.read_text(encoding="utf-8").splitlines():
            try:
                bekannt.add(json.loads(zeile)["datentag"])
            except Exception:
                continue
    if letzter_tag in bekannt:
        print(f"[SKIP] {letzter_tag} bereits geloggt ({len(bekannt)} Eintraege)")
        return 0

    sma50 = float(px.rolling(50).mean().iloc[-1])
    sma200 = float(px.rolling(200).mean().iloc[-1])
    close = float(px.iloc[-1])
    mom12 = float(px.iloc[-1] / px.iloc[-253] - 1) if len(px) >= 253 else float("nan")
    s_trend = float(close < sma200)
    s_cross = float(sma50 < sma200)
    s_mom = float(mom12 < 0)
    risiko_solo = s_mom
    risiko_chor = (s_trend + s_cross + s_mom) / 3
    eintrag = {
        "geloggt_utc": datetime.now(timezone.utc).isoformat(),
        "datentag": letzter_tag,
        "spy_close": round(close, 2),
        "signale": {"trend200": s_trend, "cross": s_cross, "mom12": s_mom},
        "mom12_wert": round(mom12, 4),
        "exposure_mom12_dial": round(1 - DIAL * risiko_solo, 4),
        "exposure_chor_dial": round(1 - DIAL * risiko_chor, 4),
    }
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(eintrag, ensure_ascii=False) + "\n")
    print(
        f"[OK] {letzter_tag}: mom12 {mom12:+.1%} -> Exposure solo {eintrag['exposure_mom12_dial']:.0%} / Chor {eintrag['exposure_chor_dial']:.0%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
