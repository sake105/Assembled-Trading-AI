# Audit Log Retention Policy

> Audit C3-074 — operational policy for the JSONL audit logs under
> `output/ops/`. Sets retention, rotation, and the off-site backup
> expectation. Until a regulator imposes specifics, this is the project's
> self-imposed standard.

## Logs in scope

| File | Producer | Purpose |
|---|---|---|
| `output/ops/kill_switch_audit.jsonl` | `execution/kill_switch.py` | every activate/deactivate/guard event, hash-chained |
| `output/ops/pit_guard_audit.jsonl` | `data/pit_guard.py` | every PIT violation observation |
| `output/ops/reconciliation_audit.jsonl` | `accounting/reconciliation.py` | every SLO evaluation |
| `output/ops/api_audit.jsonl` | `api/middleware.api_audit_middleware` | every state-mutating API call |
| `output/ops/orders_audit.jsonl` (planned) | order-lifecycle | every order state transition |

## Retention

- **Online retention**: 7 years on the local filesystem.
- **Cold storage**: identical copies replicated off-site at least weekly.
  Backblaze B2 / S3 / object storage with object-lock recommended.
- **Tamper-evidence**: the kill-switch audit log is hash-chained;
  any future audit logs SHOULD adopt the same `prev_hash` + `hash`
  pattern. Use `kill_switch.verify_audit_chain()` as the reference.

## Rotation

- Files are append-only JSONL. When a file exceeds 100 MB **or** crosses
  a quarter boundary, rotate to `<file>.YYYY-Q<n>.jsonl.gz` (gzip,
  read-only) and start a fresh `<file>.jsonl`.
- After rotation: keep the gzip on disk (cold copy lives off-site), update
  the chain anchor of the new file to the last hash of the rotated file.
- Use logrotate or `scripts/log_rotation.py` (see `src/assembled_core/ops/log_rotation.py`).

## Deletion

- **Never** delete logs younger than 7 years on the live host.
- Logs older than 7 years may be deleted from the live host only after
  the off-site copy is verified.
- Deletion is itself an audit event: log
  `{kind: "audit_log_purge", file, pre_hash, n_records, ts}` to
  `output/ops/admin_audit.jsonl`.

## Access

- Audit logs are **read-only** in normal operation. Operators MUST NOT
  edit them by hand. Any necessary correction is recorded as a new
  record referencing the original by line number — no in-place mutation.
- A `chmod 0444` (POSIX) or ACL-equivalent on the live files is
  recommended once rotation is automated.

## What this policy is NOT

- Not a GDPR retention policy (see C4-090 — News-PII has its own
  shorter retention).
- Not a regulatory document — confirm specifics with counsel if/when
  RTS-6 activates (see `docs/RTS6_SELF_ASSESSMENT.md`).
