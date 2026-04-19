# Data Models — Assembled-Trading-AI (T6.11)

Schema reference for core data models. All models should use `frozen=True` Pydantic v2
where immutability is required (PIT snapshots, audit artifacts).

---

## Intel / News

### `NewsEvent`

| Field | Type | Notes |
|-------|------|-------|
| `event_id` | `str` | `ne_<sha256[:16]>` — derived from url + date_str |
| `source_id` | `str` | e.g. `"gdelt"`, `"reuters"`, `"newsapi"` |
| `source_tier` | `SourceTier` | T0–T3 (T0 = highest trust) |
| `title` | `str` | Article title or synthetic from source+url_tail |
| `url` | `str` | Canonical URL |
| `published_at` | `datetime` | UTC-aware |
| `ingested_at` | `datetime` | UTC-aware; set at ingest time |
| `geo_tags` | `list[str]` | 2-letter ISO country codes |
| `entities` | `list[str]` | Free-text org names (up to 5) |
| `keywords` | `list[str]` | GDELT themes or keywords (up to 10) |
| `content_hash` | `str` | `sha256(url)[:16]` — for deduplication |

PIT invariant: `published_at` is the disclosure timestamp; never the fetch timestamp.

---

### `EvidenceCluster`

| Field | Type | Notes |
|-------|------|-------|
| `cluster_id` | `str` | `ec_<uuid4_short>` |
| `trigger_type` | `str` | e.g. `"GEO_CONFLICT"` |
| `geo_tags` | `list[str]` | Union of member event geo_tags |
| `confidence` | `float` | 0–1; Bayesian (T2.8 flipped to production) |
| `max_tier` | `SourceTier` | Best tier among supporting events |
| `supporting_events` | `list[NewsEvent]` | Contributing events |
| `created_at` | `datetime` | UTC-aware |
| `expires_at` | `datetime` | UTC-aware; TTL-based |

Confidence formula (T2.8 Bayesian): see `src/assembled_core/intel/bayesian_confidence.py`.

---

### `GeoTrigger`

| Field | Type | Notes |
|-------|------|-------|
| `trigger_id` | `str` | `gt_<uuid4_short>` |
| `trigger_type` | `str` | CAMEO-derived category |
| `geo_tags` | `list[str]` | Affected countries |
| `severity` | `str` | `"HIGH"`, `"MEDIUM"`, `"LOW"` |
| `confidence` | `float` | 0–1 |
| `source_tier` | `SourceTier` | Best tier |
| `as_of` | `datetime` | PIT timestamp |
| `evidence_ids` | `list[str]` | Linked cluster or event IDs |

---

## Disclosure / Insider

### Disclosure Event (CSV/DataFrame schema)

| Column | Type | Notes |
|--------|------|-------|
| `event_id` | `str` | Unique ID |
| `symbol` | `str` | Ticker, uppercase |
| `event_date` | `date` | Filing/transaction date |
| `disclosure_date` | `date` | When publicly available (Form-4: event_date+2d) |
| `event_type` | `str` | `"insider_buy"`, `"insider_sell"`, `"form4"`, `"8k"`, … |
| `severity` | `str` | `"HIGH"`, `"MEDIUM"`, `"LOW"` (optional) |
| `source_tier` | `str` | `"T0"`, `"T1"`, `"T2"`, `"T3"` |
| `value_usd` | `float` | Transaction value (optional) |

PIT rule: use `disclosure_date` as the gate, not `event_date`.
Form-4 insider: `disclosure_date = event_date + 2 business days` (T2.3 flipped).

---

## Evidence Grade Artifact

Schema: `evidence_grade.v1`

| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | `str` | `"evidence_grade.v1"` |
| `run_id` | `str` | Pipeline run ID |
| `generated_utc` | `str` | ISO 8601 UTC |
| `evidence_grade` | `str` | `"A"`, `"B"`, `"C"`, `"D"` |
| `grade_description` | `str` | Human-readable |
| `sources` | `list[str]` | Contributing source IDs |
| `geo_score` | `int \| null` | Number of independent geo signals |
| `crisis_mode` | `str \| null` | Active crisis mode state |
| `misinfo_score` | `float \| null` | 0–1; higher = more misinfo risk |

File: `output/intel/evidence/evidence_grade_{run_id}.json`

---

## IC Feedback Artifact

Schema: `ic_report.v1` (produced by `ICTracker.compute_report()`)

| Field | Type | Notes |
|-------|------|-------|
| `generated_utc` | `str` | ISO 8601 UTC |
| `results` | `dict[str, TriggerIC]` | Per trigger-type IC |

`TriggerIC`:

| Field | Type | Notes |
|-------|------|-------|
| `ic` | `float \| null` | Pearson IC (signal vs realized return) |
| `n_obs` | `int` | Number of observations |
| `flagged_weak` | `bool` | True if IC < threshold |

---

## Trigger Snapshot Artifact

File: `{store_dir}/{source}/{run_id}.json` (produced by `TriggerSnapshotStore`)

| Field | Type | Notes |
|-------|------|-------|
| `source` | `str` | Source system ID |
| `run_id` | `str` | Pipeline run ID |
| `archived_at` | `str` | ISO 8601 UTC |
| `artifact` | `str` | Path to original artifact |
| `data` | `dict` | Full trigger payload at PIT |

---

## Schema Versioning Rules

1. All artifact files must carry a `schema_version` field.
2. Breaking field changes require a new version suffix (`.v2`, etc.).
3. Pydantic v2 models for in-memory objects should use `model_config = ConfigDict(frozen=True)` for PIT snapshots.
4. Never mutate a snapshot after it's been archived.

---

*Last updated: 2026-04-20 (T6.11)*
