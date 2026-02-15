# Ops Evidence Golden Path (Import -> Require -> Pack -> Verify -> Archive)

Single canonical workflow for Ops/Devs. Windows, ASCII-only, one command block.

**See also:** docs/LEDGER_RECONCILIATION.md (system context) | docs/EVIDENCE_PACK.md (schemas) | docs/PROJECT_STRUCTURE.md (presets)

---

## When to use

- You need a **verified** Evidence Pack ZIP for a run/date (audit, compliance, support).
- You may have an external broker snapshot to import and want reconciliation to **require** it.
- You want one place to copy-paste the full path: import (optional) -> run pipeline with require + pack -> verify (gate) -> archive.

---

## 5-step Windows command block (py -3)

Set variables, then run in order. Verify must pass (exit 0) before archive runs.

```batch
set RUN_ID=ledger_eod_1d
set AS_OF=2025-01-15
set OUT=output
set ARCHIVE=archive
mkdir %ARCHIVE% 2>nul

REM 1) Import (optional): external broker snapshot
py -3 scripts/import_broker_snapshot.py --input broker_positions_%AS_OF%.json --run-id ops_%AS_OF% --as-of-date %AS_OF% --output-dir %OUT%

REM 2) Require + Pack: EOD with require policy and evidence pack
py -3 scripts/run_eod_pipeline.py --freq 1d --broker-snapshot-policy require --broker-snapshot-run-id ops_%AS_OF% --write-evidence-pack

REM 3) Pack fallback: if pack was not created by EOD, export standalone
py -3 scripts/export_evidence_pack.py --run-id %RUN_ID% --as-of-date %AS_OF% --output-dir %OUT%

REM 4) Verify gate: must pass; do not archive if verify fails
py -3 scripts/verify_evidence_pack.py --zip %OUT%/evidence_%RUN_ID%/pack_%AS_OF%.zip
if errorlevel 1 echo Verify failed - not archiving & exit /b 1

REM 5) Archive (only after verify succeeded)
copy /Y "%OUT%\evidence_%RUN_ID%\pack_%AS_OF%.zip" "%ARCHIVE%\pack_%RUN_ID%_%AS_OF%.zip"
echo Archived to %ARCHIVE%\pack_%RUN_ID%_%AS_OF%.zip
```

---

## PowerShell block (same workflow)

Same steps as above; use PowerShell variables and exit gate.

```powershell
$RUN_ID="ledger_eod_1d"
$AS_OF="2025-01-15"
$OUT="output"
$ARCHIVE="archive"
if (-not (Test-Path $ARCHIVE)) { New-Item -ItemType Directory -Path $ARCHIVE | Out-Null }

# 1) Import (optional): external broker snapshot
py -3 scripts/import_broker_snapshot.py --input broker_positions_$AS_OF.json --run-id ops_$AS_OF --as-of-date $AS_OF --output-dir $OUT

# 2) Require + Pack: EOD with require policy and evidence pack
py -3 scripts/run_eod_pipeline.py --freq 1d --broker-snapshot-policy require --broker-snapshot-run-id ops_$AS_OF --write-evidence-pack

# 3) Pack fallback: if pack was not created by EOD, export standalone
py -3 scripts/export_evidence_pack.py --run-id $RUN_ID --as-of-date $AS_OF --output-dir $OUT

# 4) Verify gate: must pass; do not archive if verify fails
py -3 scripts/verify_evidence_pack.py --zip "$OUT/evidence_$RUN_ID/pack_$AS_OF.zip"
if ($LASTEXITCODE -ne 0) { Write-Error "Verify failed - not archiving"; exit 1 }

# 5) Archive (only after verify succeeded)
Copy-Item -Force "$OUT\evidence_$RUN_ID\pack_$AS_OF.zip" "$ARCHIVE\pack_${RUN_ID}_$AS_OF.zip"
Write-Host "Archived to $ARCHIVE\pack_${RUN_ID}_$AS_OF.zip"
```

---

## If Evidence Index missing -> Manifest fallback

Export (step 3) builds the pack from the **Evidence Index** when present (`output/evidence_<run_id>/evidence_<YYYY-MM-DD>.json`). If that file is missing, it falls back to an orchestrator **manifest** (e.g. `run_manifest_1d.json`) when available. If both are missing or insufficient, export fails with a clear error. See docs/EVIDENCE_PACK.md (Input Sources, Manifest fallback).

---

## Verify gate must pass before archive

Step 4 is a **gate**: exit code 0 = pack valid (manifest present, schema ok, checksums ok, no illegal paths). If verify returns non-zero, the batch exits and the copy in step 5 is not run. Fix the pack or path and re-run from step 3 or 4. Do not archive unverified ZIPs.

---

## Where to find artifacts

| Artifact | Path (relative to output dir) |
|----------|-------------------------------|
| Evidence Index | `evidence_<run_id>/evidence_<YYYY-MM-DD>.json` |
| Evidence Pack ZIP | `evidence_<run_id>/pack_<YYYY-MM-DD>.zip` |
| Pack manifest | `evidence_<run_id>/pack_manifest_<YYYY-MM-DD>.json` (also inside ZIP) |
| After archive | `%ARCHIVE%\pack_<run_id>_<YYYY-MM-DD>.zip` |

Run_id and as_of_date come from the variables in the block above (or your own RUN_ID/AS_OF).
