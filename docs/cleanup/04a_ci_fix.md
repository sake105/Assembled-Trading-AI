# Paket 4a — CI-Reparatur

**Datum:** 2026-05-28  
**Branch:** main  
**Zweck:** GO_LIVE_CHECKLIST A2 — 6 rote Workflows reparieren.

---

## Ursachen und angewandte Fixes

### 1. Backend CI (ubuntu, Py 3.10+3.11) — exit code 127

**Ursache:**  
`backend-ci.yml` installiert via `pip install -r requirements.txt` + `pip install -e . --no-deps`.  
`requirements.txt` enthielt weder `pytest` noch `ruff` noch `pytest-cov`.  
`--no-deps` verhindert die Installation der `[dev]`-Extras aus `pyproject.toml`.  
→ `pytest`-Befehl nicht gefunden → exit 127.

**Fix:**  
`requirements.txt`: `pytest==9.0.1`, `pytest-cov==7.0.0`, `ruff==0.14.7` hinzugefügt.

---

### 2. CI ubuntu+windows — test_causal_ml.py "fit_plr requires scikit-learn"

**Ursache:**  
`ci.yml` installiert `pip install -e ".[dev]"`. Das `[dev]`-Extra in `pyproject.toml`  
enthält kein `scikit-learn` (das liegt im `[ml]`-Extra).  
`test_causal_ml.py` importiert `from src.assembled_core.signals.causal_ml import fit_plr`  
auf Modulebene — wenn `causal_ml.py` intern sklearn benötigt, schlägt bereits die  
Testkollektion fehl.

**Fix:**  
`tests/test_causal_ml.py`: `pytest.importorskip("sklearn")` **vor** dem causal_ml-Import  
eingefügt. Entspricht der Repo-Konvention (`tests/regression/test_deflated_sharpe.py`  
verwendet dasselbe Muster für scipy). Der Test wird übersprungen statt rot.

**Alternativ-Option (nicht gewählt):** sklearn in `[dev]` aufnehmen — wäre invasiver,  
würde sklearn in jedem CI-Install erzwingen.

---

### 3. Accounting CI (Windows) — ModuleNotFoundError scipy

**Ursache:**  
`accounting-ci.yml` installiert manuell: `pip install pandas pyarrow fastparquet pytest ruff pydantic pydantic-settings pyyaml exchange_calendars`.  
Kein scipy, statsmodels oder scikit-learn. Die Preset-Tests (`--preset broker_snapshot`,  
`--preset accounting`) importieren src/-Module, die scipy und statsmodels voraussetzen.

**Fix:**  
`accounting-ci.yml`: `scipy statsmodels scikit-learn` zur pip-install-Zeile hinzugefügt.

---

### 4. Release Gate CI (Windows) — scipy cascade

**Ursache:**  
`release-gate-ci.yml` installiert `pip install pandas pyarrow pytest ruff pydantic pydantic-settings`.  
`release_sprint13.py` und die zugehörigen Tests benötigen scipy/statsmodels.

**Fix:**  
`release-gate-ci.yml`: `scipy statsmodels scikit-learn` hinzugefügt.

---

### 5. Evidence Pack CI (Windows) — scipy cascade

**Ursache:**  
`evidence-pack-ci.yml`: identisches Problem wie Release Gate CI — gleiche minimale  
Install-Liste ohne scipy.

**Fix:**  
`evidence-pack-ci.yml`: `scipy statsmodels scikit-learn` hinzugefügt.

---

### 6. Ops Evidence CI (Windows) — scipy cascade

**Ursache:**  
`ops-evidence-ci.yml`: identisches Problem.

**Fix:**  
`ops-evidence-ci.yml`: `scipy statsmodels scikit-learn` hinzugefügt.

---

## Geänderte Dateien (Diff-Stat)

| Datei | Änderung |
|-------|----------|
| `requirements.txt` | +5 Einträge: pytest==9.0.1, pytest-cov==7.0.0, ruff==0.14.7, scipy>=1.10.0, scikit-learn>=1.3.0 (scipy/sklearn als Range, nicht Exact-Pin — scipy 1.16+ und sklearn 1.8+ benötigen Python >=3.11, backend-ci testet auch 3.10) |
| `tests/test_causal_ml.py` | +3 Zeilen: pytest.importorskip("sklearn") |
| `.github/workflows/accounting-ci.yml` | +3 Packages in pip-install-Zeile |
| `.github/workflows/release-gate-ci.yml` | +3 Packages (windows-job); redundante `pip install scipy`-Zeile (walk-forward-gate Ubuntu-Job) entfernt |
| `.github/workflows/evidence-pack-ci.yml` | +3 Packages in pip-install-Zeile |
| `.github/workflows/ops-evidence-ci.yml` | +3 Packages in pip-install-Zeile |

---

## Warum NICHT requirements.txt für alle Windows-Workflows?

Die 4 Windows-Workflows haben absichtlich minimale Install-Listen (schnelle CI).  
`requirements.txt` enthält Data-Provider-Pakete (`alpaca-py`, `edgartools`,  
`polygon-api-client`, etc.) die für Accounting/Ops-Tests nicht nötig sind und  
die Install-Zeit verlängern würden. Minimaler chirurgischer Fix gewählt.

---

## Nicht adressiert (kein Scope-Creep)

- pyproject.toml `[dev]` enthält kein sklearn → bleibt so (sklearn ist optional)
- requirements.txt pin-Drift gegenüber pyproject.toml ranges → bekanntes Problem  
  (Rule 40 Dependency-Drift), kein Regressionsblocker hier
- `test_causal_ml.py` hat keinen `@pytest.mark.requires_sklearn`-Marker → optional,  
  importorskip ist ausreichend für korrekte Kollektion

---

_Dieses Dokument wurde manuell erstellt._
