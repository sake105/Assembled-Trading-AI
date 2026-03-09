# PAT-0004 — Fetch-Cache mit stale-on-error

## Kontext

Intel-Fetches (z.B. EDGAR Form 4, News-RSS) können wegen Netzwerk, Rate-Limits (403/429) oder temporären API-Ausfällen fehlschlagen. Ohne Cache führt jeder Fehler zu ERROR-Health und ggf. zu unterbrochenen Pipelines oder flackernden Alerts.

## Problem

- Kein Cache: Jeder Fetch-Fehler → sofort ERROR; keine Resilienz bei kurzen Outages.
- Nur „hartes“ Cache-Fenster: Bei Fehler wird nichts ausgeliefert; Tests und Staging sind schwer deterministisch (abhängig von Live-Netzwerk).

## Lösung (Pattern)

**Zwei-Zeitfenster-Cache:**

- **`cache_minutes`**: Normales Cache-Fenster. Innerhalb dieser Zeit werden gecachte Daten wiederverwendet, ohne erneuten Netzwerk-Call.
- **`stale_on_error_minutes`**: Nach Ablauf von `cache_minutes` wird ein Fetch versucht. Schlägt der Fetch fehl (HTTP-Fehler, Timeout, Parse-Error), dürfen **stale** (ältere) gecachte Daten noch bis zu `stale_on_error_minutes` nach dem letzten erfolgreichen Fetch ausgeliefert werden. Der Run gilt dann typischerweise als DEGRADED (nicht ERROR), und die Pipeline bleibt lauffähig.

**Implementierung:**

- **Fetch-State**: Persistenter State in z.B. `output/intel/…/cache/fetch_state.json`, keyed by `source_id`. Pro Quelle: z.B. `cached_entries`, `cached_utc` (Zeitpunkt des letzten erfolgreichen Fetches).
- **Fetch-Report**: Ein `cached`-Flag (oder Äquivalent) pro Quelle im Fetch-Report signalisiert, ob die gelieferten Daten aus dem Cache kamen (und ggf. ob „stale-on-error“ genutzt wurde). So können Operatoren und Tests das Verhalten nachvollziehen.

## Warum

- **Resilienz**: Kurze SEC-/Netzwerk-Ausfälle führen nicht sofort zu ERROR.
- **Deterministische Tests**: Mit Mock und festem Cache-State lassen sich Runs ohne echte Netzwerk-Calls reproduzieren.
- **Operatoren**: Klare Unterscheidung „keine neuen Daten“ vs. „Fehler, aber alte Daten genutzt“ über Health-Status und Fetch-Report.

## Checklist

- [ ] Fetch-State wird pro `source_id` persistiert (z.B. `fetch_state.json`).
- [ ] `cache_minutes` und `stale_on_error_minutes` sind konfigurierbar und werden beim Lesen des States berücksichtigt.
- [ ] Bei Nutzung von Cache (normal oder stale-on-error) wird im Fetch-Report ein `cached`-Flag (oder vergleichbar) gesetzt.
- [ ] Health-Logik: Bei ausschließlich stale-on-error-Lieferung → DEGRADED; bei völligem Fetch-Fehler ohne gültigen Cache → ERROR.

## Referenzen

- Disclosures: `docs/disclosures/DISCLOSURES_SPEC.md`, `docs/disclosures/ARTIFACTS.md`
- Runbook: `docs/learning/checklists/RUNBOOK_CHECKLIST.md` (DISCL v1 Runbook)
