# Migrationsanleitung: trading_cycle.py → trading_cycle_v2.py

**Ziel:** Die alte `src/assembled_core/pipeline/trading_cycle.py` (9.141 Zeilen) komplett ablösen, sodass `trading_cycle_v2.py` der einzige Pfad für `run_trading_cycle` bleibt und die alte Datei gelöscht werden kann.

**Voraussetzung:** v2 ist bereits primär (`__init__.py` exportiert v2). Die alte Datei lebt nur noch wegen Helper-Funktionen, dem `TradingContext`-Datentyp und ein paar Step-Funktionen, die in v2 noch fehlen.

**Zeitschätzung:** 4–6 Sprints à ~3–4 Stunden. Nicht in einem Rutsch — sonst kippt die Test-Suite.

---

## Phase 0: Bestandsaufnahme (1 Sprint, ~2h)

Bevor du eine Zeile migrierst, mach den Code-Friedhof in der alten Datei sichtbar. Hunderte Imports zeigen auf Module, die schon im `archive/` liegen — die laufen nur durch, weil sie in `try:/except ImportError`-Blöcken stehen.

### Schritte

1. **Liste der toten Imports erzeugen.** Lege ein neues Skript `scripts/audit_trading_cycle_dead_imports.py` an, das durch `trading_cycle.py` geht und für jeden `from src.assembled_core.X.Y` prüft, ob die Datei noch unter `src/` existiert oder schon im `archive/` liegt. Output als CSV: `module, status, line_number, in_try_block`.

2. **Erwartetes Ergebnis dokumentieren.** Schreibe das Audit-Resultat in einen neuen Abschnitt von `KNOWN_ISSUES.md` unter `## 5. trading_cycle.py Migration Audit`. Ohne das schreibst du später blind.

3. **Test-Coverage der alten Datei messen.** Lauf einmal:
   ```powershell
   pytest --cov=src/assembled_core/pipeline/trading_cycle --cov-report=term-missing -m phase4
   ```
   Notiere: welche Zeilen sind durch Tests abgedeckt? Was nicht abgedeckt ist, ist Kandidat für ersatzlose Streichung — nicht für Migration.

### Definition of Done für Phase 0

- `scripts/audit_trading_cycle_dead_imports.py` läuft und produziert ein CSV.
- `KNOWN_ISSUES.md` enthält den Migration-Audit-Abschnitt mit konkreter Zahl: „X von Y Imports zeigen auf archivierte Module."
- Coverage-Report liegt unter `docs/audit/trading_cycle_coverage_<datum>.txt`.

---

## Phase 1: TradingContext nach `trading_cycle_shared.py` (1 Sprint, ~3h)

**Problem:** `TradingContext` wird von 12+ Stellen aus `trading_cycle.py` importiert (`run_daily.py`, `run_backtest_strategy.py`, `paper_track.py`, `state_machine.py`, viele Tests). Solange `TradingContext` formal in der alten Datei wohnt, kannst du sie nicht löschen — selbst wenn 99% des restlichen Codes weg ist.

`TradingContext` ist aber bereits in `trading_cycle_shared.py` re-exportiert. Du musst nur die Imports umstellen.

### Schritte

1. **Verschiebe `TradingContext` physisch.** Falls die Klasse selbst noch in `trading_cycle.py` liegt, schneide sie raus und füge sie in `trading_cycle_shared.py` ein. Falls sie schon in `shared.py` ist, springe zu Schritt 2.

2. **Globaler Import-Refactor.** Mit ripgrep + sed (oder via deinem IDE):
   ```bash
   rg -l "from src.assembled_core.pipeline.trading_cycle import.*TradingContext" --type py | \
     xargs sed -i 's|from src.assembled_core.pipeline.trading_cycle import|from src.assembled_core.pipeline.trading_cycle_shared import|g'
   ```
   **Wichtig:** Mach das nur für Imports, die ausschließlich `TradingContext` (ggf. plus andere Helper aus `shared`) holen. Imports, die `run_trading_cycle` ziehen, bleiben (vorerst) auf der alten Datei — die behandelst du in Phase 2.

3. **Re-Export in alter Datei.** Damit nichts brutal bricht, lass am Anfang von `trading_cycle.py` ein paar Zeilen stehen:
   ```python
   # DEPRECATED: import from trading_cycle_shared instead
   from src.assembled_core.pipeline.trading_cycle_shared import (
       TradingContext,
       TradingCycleResult,
   )
   ```
   Damit funktionieren Alt-Imports weiter, aber neue Codestellen nutzen die saubere Quelle.

4. **Tests laufen lassen:**
   ```powershell
   pytest -m phase4 -q
   pytest tests/test_pipeline_trading_cycle_contract.py -q
   ```
   Beide grün, sonst zurückrollen.

### Definition of Done für Phase 1

- Ripgrep auf `from src.assembled_core.pipeline.trading_cycle import TradingContext` findet **nur noch** den Re-Export in der alten Datei selbst und maximal die alte Datei + ein/zwei Tests, die explizit die Legacy-Schnittstelle testen.
- Phase-4-Suite grün.
- Commit-Message: `refactor(pipeline): move TradingContext imports to trading_cycle_shared`

---

## Phase 2: Die drei „Hauptaugenmerk"-Migrationen (3 Sprints à ~3–4h)

Das ist der inhaltliche Kern: die News→Signal→Order-Verdrahtung, die du in unserem letzten Gespräch als „Hauptaugenmerk" markiert hast. v2 hat schon den `news_signal_bridge` (Z. 840) und `crisis_alpha.pipeline` (Z. 1515). Was fehlt, sind drei spezifische Integrationen, die nur in der alten Datei leben.

### Phase 2a: Evidence-Engine Action-Gate (1 Sprint)

**Was?** Die alte Datei ruft an Z. 6685–6726 die `evidence_engine`-Kette auf: `grader.grade_evidence` → `action_gate.check_evidence_grade_gate` → `misinfo_risk.compute_misinfo_risk`. Das ist die Logik, die News-Signale **blockiert oder durchlässt**, je nachdem, ob die Evidenzlage stark genug ist. Die fehlt in v2 vollständig.

**Schritte:**

1. **Neue Funktion `_apply_evidence_gate` in `trading_cycle_v2.py`.** Position: zwischen `generate_signals` (Z. 628) und `size_positions` (Z. 1004) — also als neuer Schritt 3.5. Signatur:
   ```python
   def _apply_evidence_gate(
       signals: pd.DataFrame,
       news_events: pd.DataFrame | None,
       policy: dict,
   ) -> tuple[pd.DataFrame, dict]:
       """Filter signals through evidence-grade gate. Returns (filtered_signals, audit_info)."""
   ```

2. **Code aus alter Datei extrahieren.** Z. 6685–6770 in `trading_cycle.py` 1:1 kopieren, in die neue Funktion einbetten. Imports an den Funktionskopf ziehen (kein Lazy-Import mehr — die Module `events.evidence_engine.grader`, `.action_gate`, `.grades`, `.misinfo_risk` sind real und nicht archiviert).

3. **In `run_trading_cycle` einhängen.** Direkt nach `generate_signals`:
   ```python
   if policy.get("evidence_gate", {}).get("enabled", False):
       signals, evidence_audit = _apply_evidence_gate(signals, news_events, policy)
       result.meta["evidence_gate"] = evidence_audit
   ```

4. **Tests schreiben:**
   - `tests/test_evidence_gate_v2.py` mit drei Cases:
     - Evidence-Grade `STRONG` → Signal passiert.
     - Evidence-Grade `WEAK` → Signal wird gefiltert.
     - `evidence_gate.enabled=False` → no-op (Backwards-Compat).

5. **Charakterisierungstest.** In `tests/characterization/` einen Test, der den Output von alter und v2-Pipeline für denselben Input vergleicht. Wenn die Outputs identisch sind, ist die Migration verlustfrei.

### Phase 2b: News-Burst + Fingerprint-Dedupe (1 Sprint)

**Was?** Z. 6726–6770 in der alten Datei: `compute_bursts_for_window`, `simhash64`/`hamming_distance` für Dedupe, `build_tfidf_vectors`/`cosine_sparse` für Cluster-Ähnlichkeit, `score_triggers` für die finale Trigger-Bewertung. Das ist die News-Mikrostruktur — wie aus rohen RSS/GDELT-Items „handlungsfähige Trigger" werden.

**Schritte:**

1. **Neue Funktion `_compute_news_triggers` in `trading_cycle_v2.py`.** Position: in `_load_intel` (Z. 194) erweitern, oder als eigener Helper davor. Das Ergebnis fließt in den `news_events`-DataFrame, den Phase 2a konsumiert.

2. **Code-Block aus alter Datei isolieren.** Z. 6726–6770 sind eine geschlossene Einheit. Kopieren, Imports an den Funktionskopf.

3. **Achtung — Reihenfolge der Operationen:**
   - Erst `simhash64` + Hamming-Dedupe (Duplikate raus).
   - Dann `build_tfidf_vectors` + Cosine-Cluster (semantische Gruppen).
   - Dann `compute_bursts_for_window` (Volumen-Spikes).
   - Dann `score_triggers` (finale Bewertung).
   - Diese Sequenz steht implizit in der alten Datei drin — beim Migrieren explizit als Funktion mit kommentierten Schritten.

4. **Tests:** `tests/test_news_triggers_pipeline.py` mit:
   - Synthetischer News-Stream mit zwei Duplikaten → Dedupe entfernt sie.
   - Burst-Cluster aus 5 Artikeln → wird als ein Trigger gescored.
   - Edge case: leerer News-Stream → keine Exception, leerer Output.

### Phase 2c: News-ML-Bridge mit IC-Weights (1 Sprint)

**Was?** Z. 5963 in der alten Datei: `news_ml_bridge.get_event_type_ic_weights`. Das ist die Stelle, wo News-Eventtypen (z.B. „M&A", „earnings_surprise", „sanctions_announcement") jeweils ein **historisch kalibriertes Information-Coefficient-Gewicht** bekommen — also: welche Eventtypen haben in der Vergangenheit echte Alpha geliefert, welche waren Noise.

**Achtung — kritischer Check vor der Migration:**

```bash
ls src/assembled_core/ml/news_ml_bridge.py 2>/dev/null && echo "EXISTS" || \
  find archive -name "news_ml_bridge.py"
```

Wenn die Datei nur noch im `archive/` liegt: **migrierst du nicht**, sondern markierst die Funktion in v2 als „backlog: ML-IC-Weights". Begründung: das war ein Modul, das du schon bewusst archiviert hast. Es einfach zurückzuholen wäre Rückschritt.

Wenn die Datei noch unter `src/` liegt:

1. **Funktion `_apply_news_ic_weights` in v2.** Direkt nach `news_signal_bridge.load_and_apply_news_signals` (Z. 840) anhängen. Die News-Signale bekommen pro Event-Typ einen Multiplikator.

2. **Policy-Integration.** Konfigurierbar machen via `configs/policy.yaml`:
   ```yaml
   news_ic_weights:
     enabled: true
     min_lookback_days: 90
     fallback_weight: 0.5  # für Eventtypen ohne genug Historie
   ```

3. **Tests:** Bekanntes IC-Weight für „earnings_surprise" → Signal verstärkt. Unbekannter Eventtyp → fällt auf `fallback_weight`.

### Definition of Done für Phase 2 (alle drei)

- Drei neue Funktionen in `trading_cycle_v2.py`: `_apply_evidence_gate`, `_compute_news_triggers`, `_apply_news_ic_weights`.
- Pro Phase mindestens drei Tests (positiv, negativ, edge case).
- Charakterisierungstest pro Phase: alter vs. neuer Output identisch (oder Abweichung dokumentiert und akzeptiert).
- Phase-4-Suite plus die neuen Tests grün.

---

## Phase 3: Die Lange Schwanzwiederkehr (1 Sprint, ~3h)

Nach Phase 2 sind die drei Hauptintegrationen migriert. Was bleibt in der alten Datei:

- ~6.000 Zeilen, die hauptsächlich aus `try:/except ImportError` für **archivierte** Module bestehen.
- Eine Handvoll Helper, die noch nicht in `shared.py` liegen.

### Schritte

1. **Audit-CSV aus Phase 0 wieder vornehmen.** Alle Imports, die auf archivierte Module zeigen, sind tote Branches. Lösche sie samt zugehörigem Code-Block. Das müsste die Datei um 50–70% schrumpfen.

2. **Verbleibende Helper nach `shared.py`.** Alles, was mit `_` beginnt und noch in der alten Datei steht (`_apply_pre_trade_impact`, `_evaluate_circuit_breaker_daily`, `_estimate_symbol_volatilities`, `_filter_prices_for_as_of` etc.) — falls diese Helper in `shared.py` schon liegen: aus alter Datei löschen, falls nicht: nach `shared.py` verschieben und Re-Export in alter Datei für Backward-Compat.

3. **Test-Imports umstellen:**
   ```bash
   rg -l "_apply_pre_trade_impact\|_evaluate_circuit_breaker\|_estimate_symbol_volatilities" tests/ | \
     xargs sed -i 's|from src.assembled_core.pipeline.trading_cycle import|from src.assembled_core.pipeline.trading_cycle_shared import|g'
   ```

4. **Legacy `run_trading_cycle` rauswerfen.** Die Funktion in der alten Datei ist im Header schon als „superseded" markiert. Jetzt: Body löschen, durch eine Compat-Shim ersetzen:
   ```python
   def run_trading_cycle(*args, **kwargs):
       """DEPRECATED: use trading_cycle_v2.run_trading_cycle instead.

       Will be removed in 2026q3.
       """
       import warnings
       warnings.warn(
           "trading_cycle.run_trading_cycle is deprecated, "
           "use trading_cycle_v2.run_trading_cycle",
           DeprecationWarning,
           stacklevel=2,
       )
       from src.assembled_core.pipeline.trading_cycle_v2 import (
           run_trading_cycle as _v2,
       )
       return _v2(*args, **kwargs)
   ```

### Definition of Done für Phase 3

- `wc -l src/assembled_core/pipeline/trading_cycle.py` zeigt **unter 500 Zeilen** (von 9.141).
- Die Datei enthält nur noch: Re-Exports + DeprecationWarning-Shim.
- Phase-4-Suite grün.
- `pytest -W error::DeprecationWarning` zeigt, welche Stellen noch auf die alte API zugreifen — diese wandern als Issues in `KNOWN_ISSUES.md`.

---

## Phase 4: Beerdigung (1 Sprint, ~2h)

**Voraussetzung:** Phase 3 ist abgeschlossen, alle DeprecationWarnings sind gefixt, kein Test und kein Skript ruft mehr `trading_cycle.run_trading_cycle` auf.

### Schritte

1. **Letzter Reality Check:**
   ```bash
   rg "from src.assembled_core.pipeline.trading_cycle import" --type py
   ```
   Die einzigen Treffer dürfen sein: `trading_cycle.py` selbst (Self-Imports im Re-Export-Block) und gegebenenfalls `pipeline/__init__.py`.

2. **`trading_cycle.py` archivieren, nicht löschen.** Verschieben nach `archive/pipeline_legacy_2026q2/trading_cycle.py`. Das ist der Stil, den du im Repo schon etabliert hast (`observability_graveyard_2026q2`, `intel_research_2026q2`). Konsistenz schlägt aufräumen.

3. **`pipeline/__init__.py` säubern.** Der Import-Block:
   ```python
   from src.assembled_core.pipeline.trading_cycle import (
       TradingContext, TradingCycleResult, ...,
   )
   ```
   wird zu:
   ```python
   from src.assembled_core.pipeline.trading_cycle_shared import (
       TradingContext, TradingCycleResult,
   )
   from src.assembled_core.pipeline.trading_cycle_v2 import run_trading_cycle
   ```

4. **`trading_cycle_v2.py` umbenennen?** Optional, aber sauber: `trading_cycle_v2.py` → `trading_cycle.py` (der Name ist dann frei). Mit `git mv`, damit die Historie erhalten bleibt. Vorteil: keine `_v2`-Reminiszenzen mehr im Code, Außenstehende lesen einfach „die Trading-Cycle-Datei". Nachteil: ein großer Rename-Diff. **Empfehlung:** mach es trotzdem — der psychologische Effekt einer aufgeräumten Datei ist viel wert.

5. **`AGENTS.md` und `README.md` updaten.** Alle Stellen, die noch von `trading_cycle.py vs trading_cycle_v2.py` reden, vereinheitlichen.

6. **Letzte Tests:**
   ```powershell
   pytest -m phase4 -q
   pytest tests/test_pipeline_trading_cycle_contract.py -q
   pytest tests/characterization/ -q
   pytest -m "not external" --maxfail=3   # die volle Offline-Suite
   ```

### Definition of Done für Phase 4

- Es existiert nur noch eine Datei namens `trading_cycle.py` (oder `_v2`, je nach Entscheidung in Schritt 4).
- `wc -l` der verbleibenden Datei: unter 2.500 Zeilen.
- Alle CI-Workflows grün auf einem PR-Branch, bevor du auf `main` mergest.
- Tag setzen: `git tag pipeline-unified-2026q2 -m "trading_cycle.py and v2 merged into single canonical module"`.

---

## Globale Regeln für die ganze Migration

1. **Kein Big-Bang-Merge auf main.** Jede Phase ist ein eigener PR, jeder PR ist grün, bevor der nächste anfängt.

2. **Charakterisierungstests vor jeder inhaltlichen Änderung.** Du hast `tests/characterization/` schon — nutze es. Pinne den Output der alten Pipeline für 3–5 repräsentative Inputs als Snapshot, dann migriere, dann vergleiche. Wenn Outputs abweichen: bewusst entscheiden, ob das ein Bugfix oder ein Regressionsproblem ist.

3. **DeprecationWarnings als CI-Failure.** Sobald Phase 3 läuft, in `pytest.ini` ergänzen:
   ```ini
   filterwarnings =
       error::DeprecationWarning:src.assembled_core.pipeline.*
   ```
   Damit bricht jeder Test, der noch auf die alte API zugreift. Das macht „Vergessenes Aufräumen" sichtbar.

4. **Nicht migrieren, was du nicht brauchst.** Wenn du im Phase-0-Audit siehst, dass ein Code-Block in der alten Datei nur archivierte Module ruft, und kein Test schlägt an, wenn er fehlt: **ersatzlos streichen**, nicht migrieren. Du hast schon einmal aufgeräumt (55 ML-Module → 1) — bleib dabei.

5. **Pro Sprint einen Reality-Check schreiben.** Am Ende jedes Sprints ein zwei-Sätze-Eintrag in `PROJEKT_STATUS.md`: was migriert, was übrig, was ausgegraut. Drei Sprints später freust du dich darüber.

---

## Wenn etwas schiefgeht

- **Tests werden plötzlich rot, obwohl du nichts inhaltlich geändert hast:** Wahrscheinlich Import-Reihenfolge. Lazy-Imports in der alten Datei haben oft maskiert, dass Modul A vor Modul B importiert werden muss. Trace mit `python -X importtime scripts/run_daily.py 2>&1 | head -50`.

- **Charakterisierungstest schlägt an, du verstehst nicht warum:** Floating-Point-Rundungen oder NaN-Propagation. Vergleiche nicht mit `==`, sondern mit `pd.testing.assert_frame_equal(rtol=1e-6)`. Wenn das auch failt, ist es ein echter Logik-Drift.

- **Performance regrediert:** v2 hat weniger Try-Imports und damit weniger Cold-Start-Overhead — sollte eigentlich schneller sein. Wenn nicht, profile mit `cProfile`. Häufige Ursache: doppelte Berechnung in einer migrierten Funktion.

- **Du verlierst den Faden:** Geh zurück zur Phase-0-Audit-CSV. Sie ist deine Karte. Streich ab, was migriert ist; markiere, was als nächstes dran ist; ignoriere, was im `archive/` liegt.
