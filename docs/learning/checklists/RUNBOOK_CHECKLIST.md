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

## Risk Overlays / Troubleshooting

- **Intel orchestration (paper runner)**
  - **`intel_orchestration.news.status == ERROR` oder `disclosures.status == ERROR`:**
    - In `run_kpis.json` steht unter `intel_orchestration` der Status von NEWS- und DISCLOSURES-Pipeline.
    - Bei ERROR: Health-Artefakte prüfen (`output/intel/news/health_latest.json`, `output/intel/disclosures/health_latest.json`) sowie Fetch-Reports (`fetch_report_latest.json`); Fehlermeldungen und Quellen-Status auswerten.
  - **Runner nutzt unerwartet Sim statt echte Artefakte:**
    - Config prüfen: `paper_runner.intel.mode` in `configs/app.yaml`. Nur bei `mode=real` laufen die echten Pipelines; bei `sim` wird der BENCH-Harness verwendet, bei `none` kein Intel.

- **„Warum bin ich underinvested / so viel Cash?“**
  - **GeoRisk Overlay** prüfen:
    - Policy: `georisk_overlay.*` in `configs/policy.yaml` (State-Mapping, `by_geo_score`, `confidence_floor`).
    - Laufzeit: Logs nach Einträgen wie „GeoRisk overlay applied“ bzw. nach dem finalen Exposure-Multiplikator durchsuchen.
  - **Profit Lock** prüfen:
    - Policy: `profit_lock.*` (`lookback_days`, `trigger_return`, `multiplier_on_trigger`, `floor`, `cooldown_days`).
    - Laufzeit: Metriken/Meta-Daten (z.B. `profit_lock.multiplier`, `profit_lock_state`) ansehen; ist `multiplier < 1.0`, reduziert das System bewusst Exposure nach Gewinnen.
  - **Turnover Budget** prüfen:
    - Policy: `turnover_budget.*` (`mode`, `cap`, `behavior`, `qc`).
    - Laufzeit: `estimated_turnover`, `scale_factor`, `behavior`; ein `scale_factor < 1.0` skaliert Trades herunter, auch wenn Signale „voll investiert“ wären.

- **„Warum werden Trades blockiert oder stark skaliert?“**
  - **Turnover QC / fehlende Preise**:
    - Wenn Preise für Teile des Universums fehlen oder inkonsistent sind, kann das Turnover-Overlay konservativ auf „block“ oder „scale_to_zero“ gehen.
    - Artefakte/Logs prüfen, ob `estimated_turnover == inf` oder QC-Pfade ausgelöst wurden.
  - **Risk State / PAUSE**:
    - Wenn Risk-State `PAUSE` oder ein Health-Gate (**MARKETDATA = DEGRADED/ERROR**) aktiv ist, können Overlays faktisch alle neuen Orders unterbinden.

- **„Warum bleibt Profit Lock aktiv, obwohl die Equity seit Tagen seitwärts läuft?“**
  - **Cooldown-Logik**:
    - `cooldown_days` definiert eine Mindestdauer, in der der reduzierte Multiplikator aktiv bleibt, selbst wenn die kurzfristige Performance wieder flacher wird.
  - **State-Roundtrip**:
    - Profit-Lock-State (z.B. `trigger_idx`) wird zwischen Runs persistiert und beim nächsten Trading-Cycle wieder eingelesen.
    - Erst nach Ablauf des Cooldowns oder wenn die Lookback-Performance den Trigger nicht mehr erfüllt, kehrt der Multiplikator zu 1.0 zurück.

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

## DISCL v1 Runbook

- **Wenn `health.status == ERROR`:**
  - `output/intel/disclosures/health_latest.json` prüfen (Fehlermeldungen, `notes`, `sources_ok/failed`).
  - `configs/disclosures/sources.yaml`: aktive Quellen prüfen (type `edgar_form4`, nicht versehentlich deaktiviert?).
  - SEC User-Agent und Netzwerk: korrekter `User-Agent` in `configs/disclosures/disclosures.yaml` (edgar.form4.user_agent); Netzwerk/Firewall zu SEC.
  - `output/intel/disclosures/cache/fetch_state.json`: Cache-State je source_id prüfen (cached_utc, cached_entries); bei Fehlern ggf. stale-on-error-Logik prüfen.

- **Wenn `fetch_report` 403/429 anzeigt:**
  - User-Agent rotieren (SEC verlangt identifizierbaren Kontakt).
  - Cadence reduzieren (weniger häufige Fetches); SEC Fair-Access-Policy respektieren.
  - Keine aggressiven Retries ohne Backoff.

- **Wenn `items == 0` aber `sources_ok >= 1`:**
  - In der Regel kein Fehler: keine neuen Filings im Zeitfenster (OK).
  - `fetch_report_latest.json` und `notes` in `health_latest.json` prüfen, um temporäre Leere von echten Fehlern zu unterscheiden.

## Postmortem Pflicht (Incident)

- Nach jedem relevanten Incident:
  - Incident-File in `docs/learning/incidents` anlegen (Template nutzen).
  - Root Cause + Fix + Tests + Prevention dokumentieren.

