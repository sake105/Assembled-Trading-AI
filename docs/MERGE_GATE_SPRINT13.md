# Sprint 13 merge gate

**Primary command: release_sprint13.** Run this once before merge; it runs all blocking checks (release_sprint13 + evidence_pack presets). Secondary / deeper diagnostics: run the evidence_pack preset or verify_evidence_pack only if you need extra verification (no second copy-paste command block here).

Short checklist:
- Run the primary command below (covers CI inventory, verify CLI/schema, ops golden path, paths POSIX).
- CI workflows inventory test is included in the primary preset.
- Release Notes and full ops workflow: [docs/RELEASE_NOTES_SPRINT13.md](RELEASE_NOTES_SPRINT13.md), [docs/OPS_EVIDENCE_GOLDEN_PATH.md](OPS_EVIDENCE_GOLDEN_PATH.md).

## Primary command (Windows)

```powershell
py -3 scripts/dev/release_sprint13.py
```

Optional: add `--ops-evidence` to also run the ops_evidence preset; use `--dry-run` to print commands only. For deeper diagnostics (e.g. evidence_pack preset or offline verify) see the same preset/CLI in docs; no second code block.
