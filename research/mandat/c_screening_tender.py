"""Kandidat C — Odd-Lot-Tender-Screening (Mandat §3.3). KEIN Autotrade.

Quelle: statische EDGAR full-index Quartalsdateien (form.idx) — erreichbar, im
Gegensatz zur EFTS-Volltextsuche (Fable Round 3: blockiert). Filtert Tender-Offer-
Formulare (SC TO-I = Issuer-Tender, häufigste Odd-Lot-Quelle; SC 13E4 legacy) und
schreibt eine Watchlist. Odd-Lot-Klausel + Konditionen erfordern Lektüre des
Filings selbst — dieses Screening liefert die KANDIDATEN-Liste; Bewertung und
JEDE Entscheidung macht Hans manuell (§3.3, Guardrail 1).

Output: research/mandat/results/tender_watchlist.md
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTD = Path(__file__).resolve().parent / "results"
OUTD.mkdir(parents=True, exist_ok=True)

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass
UA = os.environ.get("SEC_USER_AGENT", "Assembled-Trading-AI hans.oertel2@gmail.com")

FORMS = ("SC TO-I", "SC TO-I/A", "SC 13E4", "13E4")


def quarter_urls(n_quarters: int = 2) -> list[str]:
    today = dt.date.today()
    urls = []
    y, q = today.year, (today.month - 1) // 3 + 1
    for _ in range(n_quarters):
        urls.append(
            f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx"
        )
        q -= 1
        if q == 0:
            y, q = y - 1, 4
    return urls


def main() -> int:
    rows = []
    for url in quarter_urls():
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            text = urllib.request.urlopen(req, timeout=60).read().decode("latin-1")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {url}: {exc}", flush=True)
            continue
        for line in text.splitlines():
            form = line[:12].strip()
            if form in FORMS:
                company = line[12:74].strip()
                cik = line[74:86].strip()
                date_filed = line[86:98].strip()
                fname = line[98:].strip()
                rows.append((date_filed, form, company, cik, fname))
        print(f"[OK] {url}: cumulative {len(rows)} tender filings", flush=True)

    rows.sort(reverse=True)
    out = OUTD / "tender_watchlist.md"
    lines = [
        "# Tender-Offer-Watchlist (Kandidat C — Screening, KEIN Autotrade)",
        "",
        f"Stand: {dt.date.today().isoformat()} · Quelle: EDGAR form.idx (statisch)",
        "",
        "Jede Zeile ist ein KANDIDAT. Odd-Lot-Klausel, Preisspanne, Fristen und",
        "Proration stehen NUR im Filing (Link) — manuelle Prüfung durch Hans",
        "erforderlich (Mandat §3.3). Faustregel: Issuer-Self-Tender (SC TO-I) mit",
        "Odd-Lot-Priorität = ≤99 Aktien werden ohne Proration angenommen.",
        "",
        "| Filed | Form | Company | CIK | Filing |",
        "|---|---|---|---|---|",
    ]
    for date_filed, form, company, cik, fname in rows:
        link = f"https://www.sec.gov/Archives/{fname}"
        lines.append(f"| {date_filed} | {form} | {company} | {cik} | [idx]({link}) |")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] {len(rows)} filings -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
