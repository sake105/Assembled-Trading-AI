# NEWS Triggers → TradingContext Integration (read-only)

## 1. Zweck

- **Was:** Integration von `output/intel/news/triggers_latest.json` als **read-only Snapshot** in den `TradingContext`.
- **Nicht-Ziele (v1):**
  - Keine Trading-/Portfolio-/Execution-Änderungen.
  - NEWS-Trigger werden nur im Context verfügbar gemacht (z.B. für spätere Regeln / Logging / UI).

## 2. Config

Beispielkonfiguration (YAML):

```yaml
intel:
  news_triggers:
    enabled: true
    path: "output/intel/news/triggers_latest.json"
```

- **Default-Verhalten:**
  - `enabled: false` → keine Integration, `TradingContext.news_triggers` bleibt `None`.
  - Wenn `enabled: true` aber Datei fehlt/invalid ist:
    - `ctx.news_triggers = None`
    - `ctx.intel_health_flags["intel_news_triggers"] = "DEGRADED"`
  - Loader prüft:
    - `schema_version == "news.triggers.v1"`
    - `items` ist eine Liste
    - JSON ist parsebar

## 3. Datenmodell im Context

- **TradingContext Felder:**
  - `news_triggers: NewsTriggerSnapshot | None`
  - `intel_health_flags: dict[str, str] | None`

- **`NewsTriggerSnapshot` Felder:**
  - `generated_utc: str`
  - `triggers: list[dict]` (direkter Inhalt von `items` aus `triggers_latest.json`)
  - `summary: dict` mit:
    - `max_severity`: maximale `severity` über alle Trigger
    - `active_count_sev2plus`: Anzahl Trigger mit `severity >= 2`
    - `watch_count_sev1plus`: Anzahl Trigger mit `severity >= 1`

## 4. QC / Troubleshooting

- **Flag `intel_news_triggers=DEGRADED`:**
  - Prüfe, ob die Datei `output/intel/news/triggers_latest.json` existiert.
  - Prüfe, ob `schema_version` im JSON `news.triggers.v1` ist.
  - Prüfe, ob die Datei gültiges JSON ist (`json.loads`).

- **Wenn alle `severity == 0`:**
  - Hinweis: das kann durch folgende Mechanismen kommen:
    - NEWS Health/QC-Caps (DEGRADED/ERROR → Severity gedeckelt).
    - Evidence-Gating (`evidence_ok == False`).
    - TTL/Decay (stale Trigger werden auf Severity 0 gesetzt).
  - Siehe Details in:
    - `docs/news/NEWS_SPEC.md`
    - `docs/news/ARTIFACTS.md` (Beschreibung von `triggers_latest.json` + Health-Metriken).

## 5. Test-Hinweis

- **`tests/test_news_triggers_loader.py` (oder äquivalent):**
  - Test **valides JSON**:
    - `schema_version == "news.triggers.v1"`, `generated_utc` gesetzt.
    - `summary.max_severity`, `watch_count_sev1plus`, `active_count_sev2plus` korrekt berechnet.
  - Test **missing/invalid JSON**:
    - Missing Datei oder invalides JSON → leerer `NewsTriggerSnapshot`.
  - Test **Context-QC-Flag**:
    - `intel.news_triggers.enabled=true`, Pfad auf fehlende Datei → `ctx.news_triggers is None` und `ctx.intel_health_flags["intel_news_triggers"] == "DEGRADED"`.

Diese Integration erlaubt es, NEWS-Trigger sicher in den Trading-Kontext zu bringen, ohne das eigentliche Trading-Verhalten zu verändern. Die QC-Flags sorgen dafür, dass schlechte/fehlende Inputs nicht zu Hard-Failures führen.

