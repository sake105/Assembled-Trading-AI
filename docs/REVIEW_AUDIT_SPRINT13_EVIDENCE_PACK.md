# Code Review / QA Audit: Evidence Pack & Ops (Sprint 13)

**Datum:** 2025-01-23  
**Rolle:** Code-Reviewer, QA/DevEx  
**Scope:** Fehler finden, Risiken reduzieren, Abläufe absichern – ohne Feature- oder Architekturänderungen.

**Harte Constraints eingehalten:** Keine neuen Features, keine Semantikänderungen zentraler Flows, Windows-kompatibel (PowerShell/cmd, py -3), ASCII-only in CLI-Fehlern, Determinismus nicht verschlechtert.

---

## Kurzer Repo-Audit (High-Level)

- **evidence_pack.py:** Path-Guards (.., backslashes, root-only) und Keys-only (missing_required/optional) sind konsistent. Pack-Manifest und ZIP werden atomar geschrieben (temp + replace). Einige `ValueError`-Meldungen nutzen noch nicht durchgängig `_ascii_only()` (verify-Pfad, _normalize_zip_path, _write_zip_deterministic, build missing_required). `read_pack_manifest_from_zip` und verify nutzen dieselbe Root-Only-Regel (`/` not in name).
- **verify_evidence_pack.py:** `--fail-on-warn` und JSON-Output sind umgesetzt. `_result_dict_for_output` enthält kein `missing_entries_count` (Schema-Parität zu anderen Counts).
- **export_evidence_pack.py:** Logging geht auf stderr; bei `--print-pack-path` nur eine Zeile stdout. Keine Vermischung mit JSON.
- **evidence_index.py:** JSON wird per `write_text()` geschrieben – nicht atomar (Partial Write bei Crash).
- **run_checks.py / release_sprint13.py:** Presets hart codiert; exit codes werden korrekt durchgereicht.
- **CI:** release-gate und ops-evidence laden Logs zuverlässig (always/failure). accounting-ci lädt bei Fehlschlag keine Artifacts.

---

## Explizite Checkliste (Stichproben)

| Prüfpunkt | Status |
|-----------|--------|
| stdout/stderr getrennt, JSON sauber | OK (export: stderr für Logs; verify: JSON nur stdout) |
| Path.resolve()/as_posix() Semantik | OK (relativ zu base_dir, POSIX für ZIP/JSON) |
| ValueError in CLI-Kontext ASCII-only | Teilweise – siehe PR 1, 2, 5 |
| Guards: missing_required/optional nur Keys | OK (evidence_pack + export) |
| ZIP-Einträge root-only, keine .., kein Backslash | OK (verify + read_pack_manifest_from_zip) |
| Doppelte Keys in JSON-Outputs | OK (bereits bereinigt: tool_version) |
| Verzeichnis als Datei | OK (is_dir() in collect_evidence_files) |
| CI: Logs bei Fail / Exit-Code | release-gate/ops-evidence OK; accounting-ci siehe PR 6 |

---

## Konkrete PR-Änderungen (max 10)

### 1. ASCII-only: ValueError in verify_evidence_pack_zip (evidence_pack.py)

- **Problem/Risiko:** In `verify_evidence_pack_zip` werden drei `ValueError` ohne `_ascii_only()` geworfen (Zeilen 976, 981, 985). `exc` bzw. `schema_version` könnten in Sonderfällen nicht-ASCII enthalten; die Verify-CLI gibt zwar `_ascii(exc_str)` aus, aber die Library-Exception sollte konsistent ASCII-only sein.
- **Relevanz:** Ops/CLI – einheitlich ASCII-only Fehlermeldungen, bessere Log-/Pipeline-Stabilität.
- **Minimaler Fix:** Alle drei Stellen: `raise ValueError(_ascii_only(f"..."))` (bzw. bei "Invalid pack manifest" bereits reiner ASCII-String; nur die beiden anderen anpassen).
- **Test/Beweis:** Bestehende Tests (z. B. `test_verify_evidence_pack_ok`, `test_verify_detects_*`) bleiben grün; optional kurzer Test, der bei ungültigem Manifest/JSON eine ValueError mit `str(exc).isascii()` oder Encoding-Check abfängt.
- **Risk Level:** low

---

### 2. ASCII-only: _normalize_zip_path und _write_zip_deterministic (evidence_pack.py)

- **Problem/Risiko:** In `_normalize_zip_path` (Zeilen 182, 186, 194) und `_write_zip_deterministic` (232, 234) werden `ValueError` mit `posix_path`, `file_path` oder `zip_entry_path` geworfen. Unter Windows oder bei Nutzerpfaden mit Umlauten können diese nicht-ASCII sein.
- **Relevanz:** Windows/Determinismus – einheitlich ASCII-only Fehler, bessere CI/Ops-Logs.
- **Minimaler Fix:** Message vor dem `raise` mit `_ascii_only(...)` umschließen (z. B. `raise ValueError(_ascii_only(f"Path contains '..' or is absolute: {posix_path}"))`).
- **Test/Beweis:** Bestehende Pfad-/ZIP-Tests; evtl. Test mit nicht-ASCII-Pfad (optional, nur wenn ohne Semantikänderung möglich).
- **Risk Level:** low

---

### 3. Evidence Index: atomares Schreiben (evidence_index.py)

- **Problem/Risiko:** `write_evidence_index_json` schreibt mit `json_path.write_text(payload + "\n", encoding="utf-8")`. Bei Abbruch (Crash/Kill) kann eine teils geschriebene Datei liegen; andere Module (Pack, Verify) könnten kaputtes JSON lesen.
- **Relevanz:** Manifest/Evidence-Index – „keine Partial Writes“, Konsistenz mit evidence_pack (Manifest wird atomar geschrieben).
- **Minimaler Fix:** Wie in evidence_pack: In dasselbe Verzeichnis mit `tempfile.NamedTemporaryFile` (oder `Path` + Suffix `.tmp`) schreiben, dann `Path.replace(tmp_path, json_path)` (Windows-sicher).
- **Test/Beweis:** Bestehende Tests für `write_evidence_index_json` (z. B. evidence_index_written, evidence_pack_written); optional Test, der während Schreiben nicht lesbare Datei vermeidet (z. B. nur Prüfung, dass nach Rückgabe gültiges JSON vorhanden ist).
- **Risk Level:** low

---

### 4. Verify JSON: missing_entries_count ergänzen (verify_evidence_pack.py + Schema-Test)

- **Problem/Risiko:** `_result_dict_for_output` enthält `bad_paths_count`, `paths_not_in_zip_entries_count`, `checksum_mismatches_count`, aber kein `missing_entries_count`. Die Logik von `--fail-on-warn` nutzt `len(result.get("missing_entries"))`; Automatisierung, die nur das JSON auswertet, kann den Count nicht direkt ablesen (nur über `details["missing_entries"]`, das auf 20 begrenzt ist).
- **Relevanz:** Schema/CLI – Parität der Counts, bessere Ops-Parsing-Logik.
- **Minimaler Fix:** In `_result_dict_for_output` ein Feld `"missing_entries_count": len(result.get("missing_entries", []))` hinzufügen (analog zu den anderen Counts). In `tests/test_verify_evidence_pack_json_schema_stable.py` den Key `"missing_entries_count"` zu `REQUIRED_KEYS` hinzufügen und für OK-ZIP `assert out1["missing_entries_count"] == 0`.
- **Test/Beweis:** `test_verify_json_schema_stable_ok_zip` und ggf. Test mit fehlenden Einträgen, der `missing_entries_count` > 0 prüft.
- **Risk Level:** low

---

### 5. ASCII-only: build_evidence_pack missing_required ValueError (evidence_pack.py)

- **Problem/Risiko:** Beim Fehlschlag wegen fehlender required files (Zeilen 501–504) wird `ValueError(f"... {missing_required}")` geworfen. Theoretisch könnten Keys nicht-ASCII sein (aktuell sind es nur feste Key-Namen).
- **Relevanz:** Konsistenz mit restlichen ASCII-only ValueErrors in diesem Modul.
- **Minimaler Fix:** Message mit `_ascii_only(...)` umschließen (z. B. `raise ValueError(_ascii_only(f"Required files missing for run_id=..."))`).
- **Test/Beweis:** Bestehende Tests für strict/required (evidence_pack_written, export CLI).
- **Risk Level:** low

---

### 6. Accounting CI: Logs bei Fehlschlag hochladen (.github/workflows/accounting-ci.yml)

- **Problem/Risiko:** Bei Fehlschlag der accounting- oder broker_snapshot-Checks werden keine Artifacts hochgeladen; Logs sind nur in der Job-Ausgabe sichtbar und gehen bei langen Runs verloren.
- **Relevanz:** DevEx/CI – gleiches Muster wie evidence-pack-ci (Log bei Fail hochladen).
- **Minimaler Fix:** Ausgabe von `run_checks.py` in eine Datei umleiten (z. B. `accounting_log.txt`), Schritt „Upload accounting logs on failure“ mit `if: failure()` (oder `outcome == 'failure'`) und `actions/upload-artifact` für diese Datei ergänzen. Exit-Code des run-Schritts unverändert lassen.
- **Test/Beweis:** Manuell: PR mit absichtlich fehlschlagendem Test; prüfen, ob Artifact erscheint und Log lesbar ist.
- **Risk Level:** low

---

### 7. (Optional) Docs: Verify-JSON-Schema-Tabelle ergänzen (docs/EVIDENCE_PACK.md)

- **Problem/Risiko:** Die Tabelle „verify_evidence_pack --json output schema“ listet `bad_paths_count` und `checksum_mismatches_count`, aber weder `paths_not_in_zip_entries_count` noch `missing_entries_count` (sobald PR 4 umgesetzt ist). Das kann zu Missverständnissen bei Integration führen.
- **Relevanz:** Docs als Single Source of Truth.
- **Minimaler Fix:** In der Tabelle zwei Zeilen ergänzen: `paths_not_in_zip_entries_count` (int), `missing_entries_count` (int), mit kurzer Beschreibung (ASCII-only). Optional in „details“ erwähnen: `missing_entries` (bis 20), `paths_not_in_zip_entries` (bis 20).
- **Test/Beweis:** Kein Code-Change; Review.
- **Risk Level:** low  
- **Hinweis:** Erst nach PR 4 sinnvoll, damit die Tabelle zum tatsächlichen Output passt.

---

### 8. (Optional) evidence-pack-ci / accounting-ci: py -3 statt python

- **Problem/Risiko:** evidence-pack-ci und accounting-ci nutzen `python`; unter Windows wird oft `py -3` empfohlen. release-gate-ci und ops-evidence-ci nutzen bereits `py -3`.
- **Relevanz:** Windows-Kompatibilität in CI einheitlich.
- **Minimaler Fix:** In den genannten Workflows den run-Schritt auf `py -3 scripts/dev/run_checks.py ...` umstellen (analog zu release-gate).
- **Test/Beweis:** CI-Lauf auf Windows; keine Verhaltensänderung außer Aufruf.
- **Risk Level:** low  
- **Hinweis:** Nur als optionale Vereinheitlichung; nur anwenden, wenn im Projekt „py -3“ als Standard für Windows festgelegt ist.

---

## Nicht geändert (bewusst)

- **Exception-Typ FileNotFoundError in verify_evidence_pack_zip:** Bleibt `FileNotFoundError`; die Verify-CLI wandelt bereits mit `_ascii(str(exc))` für die Ausgabe. Keine Änderung der Semantik (kein Wechsel zu ValueError).
- **Determinismus (ZIP/JSON):** Keine Änderungen an sort_keys, indent, newline oder fixed timestamps.
- **Path.resolve()/as_posix():** Verwendung ist konsistent und korrekt für relative Pfade und POSIX-Ausgabe; keine Anpassung nötig.

---

## Zusammenfassung

| # | Thema | Datei(en) | Risk |
|---|--------|-----------|------|
| 1 | ASCII-only ValueError in verify_evidence_pack_zip | evidence_pack.py | low |
| 2 | ASCII-only in _normalize_zip_path / _write_zip_deterministic | evidence_pack.py | low |
| 3 | Atomares Schreiben Evidence Index | evidence_index.py | low |
| 4 | missing_entries_count im Verify-JSON + Schema-Test | verify_evidence_pack.py, test_verify_evidence_pack_json_schema_stable.py | low |
| 5 | ASCII-only build_evidence_pack missing_required | evidence_pack.py | low |
| 6 | Accounting CI: Log-Artifact bei Failure | .github/workflows/accounting-ci.yml | low |
| 7 | (Optional) Docs Verify-JSON-Tabelle | docs/EVIDENCE_PACK.md | low |
| 8 | (Optional) CI: py -3 in evidence-pack/accounting | .github/workflows/*.yml | low |

Alle Vorschläge sind lokale, sichere Anpassungen ohne Konzept- oder Architekturänderung. Optional-Einträge (7, 8) können bei Bedarf umgesetzt werden; 1–6 sind „No-Regrets“-Verbesserungen.
