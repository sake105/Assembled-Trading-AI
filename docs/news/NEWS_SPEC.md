NEWS v1 Specification (Phase 0–1)
=================================

Dieses Dokument beschreibt die **NEWS v1 Pipeline (free & robust)** auf Policy-/Contract-Ebene.
Implementierung befindet sich unter `src/assembled_core/events/news/`.

Konfiguration:
- Quellen-Registry: `configs/news/sources.yaml`
- Parameter-Config: `configs/news/news.yaml`

---

## 1. Ziele / Nicht-Ziele

**Ziele:**
- Robuste News-Pipeline auf **kostenfreien Quellen**:
  - RSS-Feeds (Tier A/B)
  - Optional GDELT Doc API (leicht integrierbar, aber deaktivierbar)
- Einheitliches **NewsEvent-Schema** + **Health-Status** (OK/DEGRADED/ERROR).
- **Atomare JSON-Artefakte** unter `output/intel/news/`.
- Klare **Schema-Versionierung** pro Output-Datei.

**Nicht-Ziele (v1):**
- Kein Trading-Impact / keine direkten Orders.
- Kein „fancy“ NLP (keine komplexen Transformer/Sentiment-Modelle).
- Keine Security-/Secrets-Implementierung (nur TODO markiert).

---

## 2. Pipeline-Phasen (Phase 0–14, high-level)

- **Phase 0 – Contract & Schema-Versionierung**
  - Dieses Dokument (`docs/news/NEWS_SPEC.md`).
  - Alle Output-JSONs enthalten `schema_version`.
- **Phase 1 – Source Registry & Config**
  - `configs/news/sources.yaml` als Single Source of Truth für Quellen.
  - `configs/news/news.yaml` für Timeout/Dedupe/Health-Parameter.
- **Phase 2 – Fetch-Robustheit**
  - Retries, Backoff, ETag/If-Modified-Since.
- **Phase 3 – Clustering/Bursts**
  - Clustering verwandter News, Burst-Erkennung.
- **Phase 4 – Trigger Scoring**
  - Einfaches Scoring (0–3) für potenzielle Trading-/Risk-Trigger.
- **Phase 5 – TTL/Decay**
  - Zeitliche Verfallslogik für News-Signale.
- **Phase 6 – Integration in Health-Gates**
  - Verknüpfung mit globalen Health/QC-Gates (News/Disclosures/MarketData).
- **Phase 7–14**
  - Erweiterungen (mehr Quellen, smartere Dedupe/Clustering, Ausfall-Szenarien).

---

## 3. Schema-Versionierung (JSON Outputs)

Roadmap-Regel: **Jedes Output-JSON** der News-Pipeline trägt eine **`schema_version`**.

### 3.1 Events (`events_latest.json`)

```json
{
  "schema_version": "news.v1",
  "generated_utc": "2025-01-15T12:00:00Z",
  "count": 123,
  "items": [
    {
      "event_id": "news_abc123...",
      "title": "Example Title",
      "url": "https://example.com/article",
      "canonical_url": "https://example.com/article",
      "source_id": "rss_example_world",
      "source_name": "Example World",
      "source_domain": "example.com",
      "published_utc": "...",
      "fetched_utc": "...",
      "summary": "...",
      "language": null,
      "raw": { "...": "..." },
      "fingerprint": "..."
    }
  ]
}
```

### 3.2 Health (`health_latest.json`)

```json
{
  "schema_version": "news.health.v1",
  "generated_utc": "2025-01-15T12:00:00Z",
  "health": {
    "status": "OK",
    "fetched_utc": "...",
    "sources_total": 3,
    "sources_ok": 2,
    "sources_failed": 1,
    "items_raw": 200,
    "items_after_dedupe": 150,
    "failures": [
      {"source": "rss_foo", "reason": "rss_fetch_error: 500"}
    ],
    "notes": [
      "No items after dedupe despite at least one source OK."
    ]
  }
}
```

### 3.3 Clusters & Triggers (Stubs, optional)

```json
{
  "schema_version": "news.clusters.v1",
  "generated_utc": "2025-01-15T12:00:00Z",
  "count": 0,
  "items": []
}
```

```json
{
  "schema_version": "news.triggers.v1",
  "generated_utc": "2025-01-15T12:00:00Z",
  "count": 0,
  "items": []
}
```

---

## 4. Health-Gates & Operator-Verhalten

**Health-Status:**

- `OK`: mindestens eine Quelle erfolgreich, dedupte Items vorhanden.
- `DEGRADED`: mindestens eine Quelle ok, aber Fehler/0 Items → News nutzbar, aber nur im WATCH-Mode.
- `ERROR`: keine Quelle ok oder harter Fehler → News nicht vertrauenswürdig.

**Trigger-QC-Gates:**

- Bei `DEGRADED` werden Trigger-Severities per Config gedeckelt (z.B. `≤1`).
- Bei `ERROR` werden Trigger-Severities auf `0` gesetzt.
- Es wird eine Audit-Note wie `trigger_qc_cap:<STATUS>:<CAP>` in `health.notes` abgelegt.

---

## 5. Upgrade-Pfade & Backward-Kompatibilität

**Backward-Kompatibilität (v1):**

- `bursts_latest.json` enthält für v1:
  - Legacy-Felder: `window_hours`, `count`, `items` (Primary-Window-View).
  - Neu: `windows[]` mit je `window_hours`, `top_entities_burst`, `top_phrases_burst`, `top_clusters_burst`.
- Reader, die nur `count/items` kennen, bleiben funktionsfähig.

Bei Schema-Inkompatibilitäten:
- `schema_version` wird erhöht (z.B. `news.v1.1`, `news.v2`).
- Alte Reader können je nach Strategie:
  - nur bekannte Felder nutzen, oder
  - Policy-gesteuert auf neue Schema-Versionen migrieren.

---

## 6. Referenzen auf Configs

- `configs/news/news.yaml`: Fetch/GDELT/Dedupe/Clustering/Baseline/Burst/Trigger-Scoring-Parameter.
- `configs/news/sources.yaml`: Quellen-Registry inkl. Tier/Domain.
- `configs/news/taxonomy.yaml`: Rule-based Topics & Trigger-Typen.

## 7. Deferred / Known Issues

- Security/Secrets (API-Keys, OAuth, `.env`-Handling) sind **bewusst deferred** und in `KNOWN_ISSUES.md` dokumentiert.
- NEWS v1 arbeitet ausschließlich mit Free/Unauthenticated Sources.

