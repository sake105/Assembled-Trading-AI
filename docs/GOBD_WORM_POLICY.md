# GoBD WORM Storage Policy

> Audit C4-091 — once the firm is `gewerblich` (commercial), the
> German **Grundsätze zur ordnungsmäßigen Führung und Aufbewahrung von
> Büchern, Aufzeichnungen und Unterlagen in elektronischer Form sowie
> zum Datenzugriff (GoBD)** require that trade-relevant records are
> retained **10 years**, **unmodified**, **machine-readable**, and
> **available to the tax authority on demand**.
>
> **Status today:** private trader, no GoBD applicability. Once UG or
> sole-proprietor-with-Gewerbe is in place, this policy activates
> automatically by operation of law.

## 1. What GoBD requires (short version)

- **Vollständigkeit** — every relevant transaction recorded.
- **Richtigkeit** — recorded as it actually happened (no edits).
- **Zeitgerechtigkeit** — recorded promptly.
- **Ordnung** — recoverable and searchable.
- **Unveränderbarkeit** — once written, cannot be modified
  (Write-Once-Read-Many semantics).
- **Aufbewahrungsfrist** — 10 years for trade records (§ 257 HGB,
  § 147 AO).

## 2. What is in scope

| Record type | Source | GoBD scope? | Today |
|---|---|---|---|
| Filled orders (price, qty, ts) | `output/ops/orders_audit.jsonl` (planned) + broker confirmations | YES | hash-chained planned, broker statement is the canonical record |
| Reconciliation outcomes | `output/ops/reconciliation_audit.jsonl` | YES | append-only with fsync (W5-2) |
| Kill-switch state changes | `output/ops/kill_switch_audit.jsonl` | YES (operational record) | hash-chained (W3-1) |
| Tax-relevant P&L | broker year-end statement | YES | broker provides |
| Strategy code at time of trade | git commit on `main` | YES (signed version of "how the order was generated") | git is immutable |
| Research notebooks | `research/` | NO (drafts, not records) | — |

## 3. How we achieve WORM today

- **Append-only JSONL** for all `output/ops/*.jsonl` (write-only,
  no in-place mutation by application code).
- **fsync per write** so a process kill cannot truncate.
- **Hash-chain** on the kill-switch audit log (`prev_hash` + `hash`)
  — tampering is detectable. Future: same pattern on orders /
  reconciliation audit logs.
- **Off-site replication** weekly to Backblaze B2 with **Object Lock
  in Compliance mode** — once written, the record cannot be deleted
  by any actor (including the operator) until the lock expires.

## 4. Object Lock specifics (Backblaze B2 / S3)

For each audit-log bucket:

- Default retention: 3650 days (10 years).
- Mode: **Compliance** (not Governance — Compliance cannot be
  overridden even by the bucket owner).
- Legal-hold flag: OFF by default; SET when an open
  regulator/court request applies.

Setup script: `scripts/ops/setup_b2_backup.sh` (Wave 10 deliverable).

## 5. Tax-authority access (Z3 / Z2 / Z1 per GoBD)

GoBD recognises three access levels:

- **Z1 — unmittelbarer Datenzugriff**: auditor logs into the system
  and reads. Today: would require the operator to grant a temporary
  read-only account on the live host. Not pre-provisioned.
- **Z2 — mittelbarer Datenzugriff**: operator runs queries on the
  auditor's behalf and shows results. The right level for a solo
  operator.
- **Z3 — Datenträgerüberlassung**: operator hands over a machine-
  readable export (CSV / JSON / parquet). Easy to satisfy:
  `output/ops/*.jsonl` already is the right format.

The operator MUST be able to produce a Z3 export within a
reasonable working day. The export procedure is documented in
`docs/OPERATOR_RUNBOOK.md` (add export step under §5).

## 6. Format & migration

- JSONL is acceptable (GoBD Rz. 134 — machine-readable formats).
- If we migrate to a newer schema, the old records MUST be
  retained in their original form for the full 10 years (no
  silent transformation).
- Migration events themselves are audited:
  `{kind: "schema_migration", from_version, to_version, ts}`
  added to `output/ops/admin_audit.jsonl`.

## 7. What this policy is NOT

- Not a substitute for a Steuerberater. The 10-year clock starts
  at the end of the calendar year in which the record was created,
  not at creation. A Steuerberater confirms which records actually
  fall under § 147 AO for each tax year.
- Not a guarantee of WORM-ness without the off-site Object-Lock
  bucket. Local-only retention is not WORM-compliant.
- Not legal advice.
