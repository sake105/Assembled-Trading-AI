# Projekt Status - Assembled Trading AI

**Letzte Aktualisierung:** 2026-04-26

## Phase 4: Backend Core - ✅ ABGESCHLOSSEN

### Status
- **110 Phase-4-Tests**: Alle grün ✅
- **Laufzeit**: ~18 Sekunden
- **Performance**: Optimiert und stabil

### Test-Infrastruktur

#### Standard Phase-4-Tests (empfohlen für tägliche Entwicklung)
```powershell
# Schnell (~18s, 110 Tests)
pytest -m phase4 -q

# Oder mit PowerShell-Script
.\scripts\run_phase4_tests.ps1
```

#### Backtest Engine Tests
```powershell
# Schnelle Tests (~1.5s, 7 Tests)
pytest tests/test_qa_backtest_engine.py -m "not slow" -q

# Langsame Tests (~10s, 5 Tests) - nur bei Engine-Änderungen
pytest tests/test_qa_backtest_engine.py -m "slow" -q

# Alle Backtest-Tests (~11s, 12 Tests)
pytest tests/test_qa_backtest_engine.py -q
```

#### Vollständige Offline-Suite (optional, ~10 Minuten)
```powershell
# Nur bei größeren Umbauten (IO/API/Health)
pytest -m "not external" --maxfail=3
```

### Phase-4-Module (alle getestet)
- ✅ **TA-Features**: `add_log_returns`, `add_atr`, `add_rsi`, `add_all_features`
- ✅ **QA-Metriken**: Sharpe, Sortino, Drawdown, Turnover, CAGR, etc.
- ✅ **QA-Gates**: OK/WARNING/BLOCK-Logik
- ✅ **Backtest-Engine**: Vollständige Portfolio-Simulation
- ✅ **Reports**: QA-Report-Generierung
- ✅ **Pipelines**: `run_backtest_strategy.py`, `run_eod_pipeline.py`

### Performance-Optimierungen
- `test_run_backtest_strategy_with_universe`: 95% schneller (31s → 1.3s)
- Langsame Tests mit `@pytest.mark.slow` markiert
- Test-Daten optimiert (3 Jahre → 1.5 Jahre für Multi-Year-Tests)

### Git Status
- **Tag**: `phase4_stable` gesetzt
- **Branch**: `main` (up to date)
- **Dependencies**: Alle sauber dokumentiert (pyarrow, fastparquet)

---

## Nächste Phasen

### Phase 5 & 6
- Bereit für neue Features
- Phase-4-Suite als Sicherheitsnetz
- Regelmäßige Test-Läufe empfohlen

---

## Test-Strategie

### Tägliche Entwicklung
1. **Vor jedem Commit**: `pytest -m phase4 -q`
2. **Bei Backtest-Engine-Änderungen**: `pytest tests/test_qa_backtest_engine.py -m "not slow" -q`
3. **Gelegentlich**: Langsame Tests laufen lassen

### Vor größeren Releases
- Vollständige Offline-Suite: `pytest -m "not external" --maxfail=3`
- Alle Phase-4-Tests: `pytest -m phase4 -q`
- Langsame Tests: `pytest tests/test_qa_backtest_engine.py -m "slow" -q`

---

## Dependencies

### Core Dependencies
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `pyarrow>=10.0.0` (für Parquet)
- `fastparquet>=2023.1.0` (für Parquet)

### Development Dependencies
- `pytest>=7.0.0`
- `pytest-cov>=4.0.0`
- `ruff>=0.14.0`
- `mypy>=1.5.0`

Alle Dependencies sind in `pyproject.toml` und `requirements.txt` dokumentiert.

---

## Strategische Ausrichtung & Constraints

> **Herkunft:** Migriert aus `CLAUDE.md` §1.3–1.5 / §3 / §13 am **2026-05-30**
> im Zuge der CLAUDE.md-Verschlankung. `CLAUDE.md` selbst trägt nur noch Kernregeln;
> die strategische Ziel-/Constraint-Beschreibung lebt jetzt hier als Projektkontext.

### Projektcharakter (vormals §1.3)

Assembled-Trading-AI ist ein **modulares Python-Backend** für: Research, Backtests,
Paper-/Simulation, Risk-Overlays, QA / Evidence / Reporting, API / OMS-light / Paper-Routing
sowie schrittweise Intel-, News-, Disclosure- und GeoRisk-Integration.

Es ist **kein kleines Einzel-Skript**, sondern ein wachsendes System mit mehreren
Teilwelten, historischer Drift, branch-spezifischen Fixes und dokumentationsgetriebener
Weiterentwicklung.

### Nicht nur Rendite (vormals §1.4)

Dieses Projekt ist **nicht nur auf nominale Rendite** ausgerichtet. Wichtige Ziele zusätzlich:
Reproduzierbarkeit, Nachvollziehbarkeit, Qualitäts- und Testdisziplin, kontrollierte
Weiterentwicklung, harte Risk-Grenzen, saubere Zustands- und Kontrolllogik, dokumentierte
Entscheidungen, Vermeidung architektonischer Drift.

### Strategische Leitidee (vormals §1.5)

- EOD-/Daily-zentrierter Kern
- modulare Alpha-Generierung
- später stärkere Intel-/Geo-/Disclosure-Einbindung
- Risk-first statt Rendite-first
- kontrollierte State-Machine-Logik
- kein Leverage im frühen Betriebsmodus
- keine unkontrollierte Tool- oder Agentenautomatisierung

### Projektziele, Constraints und Risiko-Philosophie (vormals §3)

**Primäre Ziele:** robustes quantitatives Backend; Research, Backtests, Paper-Runs und
Risk-Steuerung tragen; schrittweise Intel-/GeoRisk-/Disclosure-Funktionalität aufnehmen;
langfristig produktionsnäher werden, aber nicht durch zu frühe Live-/Prod-Komplexität
destabilisiert werden.

**Qualitätsziele:** deterministische oder weitgehend reproduzierbare Läufe; dokumentierte
Artefakte; branch- und CI-sichere Änderungen; testbare Interfaces; keine stillen
Seiteneffekte; keine unkontrollierte Kopplung; saubere Trennung zwischen Kernlogik und
Hilfsschichten.

**Risikophilosophie:** harte Risiko-Grenzen sind wichtiger als aggressive Zielrendite;
Drawdown-, Volatilitäts- und Turnover-Kontrolle sind zentrale Steuergrößen; Risk-State- und
Overlay-Logik sind zentraler Systembestandteil; lieber kontrolliert konservativ als
unkontrolliert aggressiv.

**Renditeziele:** grobes Zielband **ca. 20–30 % p.a.** als strategischer
Orientierungsrahmen — **kein** simpler Stopp-Schalter. Steuerfokus liegt stärker auf MaxDD,
Ziel-Volatilität, Turnover, Exposure-Steuerung, Risk-State, Soft Profit Lock und
policy-basierter Regulierung.

**Harte Constraints:**
- zunächst **kein Leverage / keine Hebelprodukte**
- keine blinden Merges oder Git-Gewaltaktionen
- keine Produktionserwartung aus synthetischen Daten ableiten
- keine stillen Live-/Prod-Annahmen
- keine unkontrollierte „selbstlernende" Agentenlogik ohne Guardrails
- kein großer Architekturumbau ohne klaren Scope

**Bewusst verschobene Themen:** Leverage; aggressive Live-Selbstoptimierung; große
Plattform-/Monorepo-Schritte vor sauberem Backend-Setup; komplexe Persistenz-/Memory-Automation
ohne klare Guardrails; einige Security-/Secrets-Härtungen wurden zeitweise bewusst als TODO
verschoben, bleiben aber wichtig.

### Bevorzugte Entwicklungsrichtung (vormals §13)

- **Backend bleibt Kern.** Kurz- bis mittelfristig wichtigster Systemkern. Plattform/Frontend
  sind Zukunftsthemen, nicht die operative Leitstruktur.
- **Realismus-Härtung ist echter Schwerpunkt:** Secret-Scanning/Security-Härtung, echtere
  Datenpfade, realistischere Cost-/Impact-Modellierung, Corporate Actions / Kalender /
  Universe-Realismus, per-day Intel-Refresh, branch-/CI-saubere Integrationspfade, weitere
  Reduktion unnötiger Churn-/Rotationseffekte.
- **Roadmap-Denken ja, Roadmap-Fiktion nein.** Roadmaps/Sprints sind wichtig, aber **kein**
  Implementierungsbeweis.
- **Erwartete spätere Ausbaurichtung:** News-/Intel-Pipeline; Disclosures-/Slow-Intel-Pfade;
  Risk-State-Machine-Härtung; Execution-Worker / Reconciliation / Kill-Switch-Härtung;
  Robustness-, Walk-Forward- und Stability-Packs; Observability und Governance-Ausbau.
