Runbook Checklist
=================

Diese Checkliste unterstützt beim Umgang mit **degradierten Systemzuständen** und Incidents.

## If system degraded

- **Status prüfen**
  - Logs der letzten Runs betrachten (Fehler, Warnungen, Zeitouts).
  - Health-/Status-Artefakte prüfen (falls vorhanden).

- **Letzten Run prüfen**
  - Letzte Backtests/Live-Runs: Zeit, Dauer, Errors, ungewöhnliche Metriken.
  - Vergleiche mit „normalem“ Verhalten (z.B. Run-Dauer, Anzahl Trades).

- **Akute Maßnahmen**
  - Falls nötig: Live-Actions pausieren/deaktivieren (Placeholder für spätere Live-Integration).
  - Rollback auf letzte stabile Version/Config (falls Release-bedingt).

## Data Freshness / Health Score (Placeholder)

- **Data Freshness**
  - Prüfen, ob Daten der letzten Tage vorhanden und konsistent sind.
  - Ggf. Data-Pipeline neu anstoßen oder fehlende Daten nachladen.

- **Health Score (Placeholder)**
  - Falls Health-Score implementiert: Score prüfen, Thresholds respektieren.

## Execution Reconcile Check (falls vorhanden)

- Abgleichen, ob erwartete Orders/Fills mit tatsächlichen Fills/Positions übereinstimmen.
- Offene Differenzen dokumentieren (Incident) und nicht ignorieren.

## Emergency Stop / Pause Mode (Placeholder)

## NEWS v1 Runbook

- **Wenn `NEWS health.status == ERROR`:**
  - `output/intel/news/health_latest.json` prüfen (Fehlermeldungen, `failures`).
  - `fetch_report_latest.json` öffnen: HTTP-Status, Timeouts, Cache-Hits ansehen.
  - `configs/news/sources.yaml`: aktive Quellen/Tier prüfen (z.B. versehentlich deaktiviert?).'
  - Output-Verzeichnisse (`output/intel/news/`, `output/intel/news/baseline/`) auf Schreibfehler/Platz prüfen.

- **Wenn `NEWS health.status == DEGRADED`:**
  - `failures` im Health-Artifact lesen: welche Quellen sind ausgefallen?
  - `not_modified` vs. echte Fehler unterscheiden (ETag/304 ist **kein** Ausfall).
  - `stale-on-error`-Verhalten bei GDELT verstehen: gecachte Daten sind erlaubt, aber markieren DEGRADED.

- **Wenn alle Trigger severity==0:**
  - `cluster.evidence.evidence_ok` prüfen (Tier-A/B-Policy erfüllt?).
  - QC-Caps: Health-Status (DEGRADED/ERROR) und `trigger_qc_cap:*` Notes prüfen.
  - TTL/Decay: `ttl_hours` und `decay.age_hours/factor` in `triggers_latest.json` kontrollieren.

- **Wenn `bursts_latest.json` leer wirkt:**
  - `burst.min_doc_count` in `configs/news/news.yaml` prüfen.
  - Baseline: `baseline_latest.json` und `version_hash` / `baseline_days` prüfen.
  - Sicherstellen, dass Clustering/Baseline aktiviert ist (`clustering.enabled`, `burst.enabled`).

- **Wenn keine Cluster erzeugt werden:**
  - `clustering.enabled` und `min_cluster_size` kontrollieren.
  - Overlap-Filter (`require_overlap`, `same_day_only`) ggf. temporär lockern.
  - Prüfen, ob `topics`/`candidate_triggers` und `evidence` wie erwartet gesetzt werden.

- Mechanismus definieren, wie das System in einen **PAUSE**-Modus versetzt wird:
  - Keine neuen Live-Trades.
  - Backtests/Research weiter erlaubt.

## Postmortem Pflicht (Incident)

- Nach jedem relevanten Incident:
  - Incident-File in `docs/learning/incidents` anlegen (Template nutzen).
  - Root Cause + Fix + Tests + Prevention dokumentieren.

