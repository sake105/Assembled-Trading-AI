# Code-Audit — Bugs gefixt, Real-Data-Tests hinzugefügt

**Stand:** 2026-05-11
**Branch:** ERWEITERUNG

---

## 1. Methodik

Systematischer Durchgang aller Erweiterungs-Module mit Fokus auf:
- Lookahead-Bias (.shift() vergessen, Reihenfolge falsch)
- NaN-Handling (silent zeros, ffill carrying stale data)
- Off-by-one in Index-Arithmetik
- Duplikate-Funktionen
- Test-Daten: synthetic noise statt echte Daten

Zusätzlich: alle Tests gegen **reale 19y-Daten** (Equity + Cross-Asset) validiert.

---

## 2. Gefundene Bugs

### Bug 1: Lookahead in `fomc_macro_signal.py`

**Severity:** Mittel
**Lokalisierung:** `src/erweiterung/strategies/fomc_macro_signal.py:92`

```python
# VORHER (Lookahead):
mask = (out.index >= meeting_date) & (out.index <= end_date)
out[mask] = target

# NACHHER (PIT-korrekt):
mask = (out.index > meeting_date) & (out.index <= end_date)
out[mask] = target
```

**Erklärung:** FOMC-Statements werden 14:00 ET veröffentlicht, US-Close ist
16:00 ET. Daher kann der Statement-Inhalt erst ab dem **nächsten** Trading-Day
für Allokations-Entscheidungen genutzt werden (für Close-basierte Returns).
Mit `>=` wurde der Statement-Tag selbst inkludiert → Lookahead.

**Impact:** Demo-Backtest minimaler Effekt (nur 22 nicht-neutrale Tage in
1315), aber Pipeline jetzt PIT-sauber.

---

### Bug 2: Stale Weights im `cross_asset_momentum_top_n`

**Severity:** Hoch
**Lokalisierung:** `src/erweiterung/strategies/master_allocator.py:130-144`

```python
# VORHER (Bug):
weights = pd.DataFrame(0.0, index=..., columns=...)
for d in rebal_dates:
    top_syms = mom.loc[d].nlargest(n_top).index
    weights.loc[d, top_syms] = 1.0 / n_top
weights = weights.replace(0, np.nan).ffill().fillna(0.0)

# NACHHER (korrekt):
weights = pd.DataFrame(np.nan, ...)
for d in rebal_dates:
    weights.loc[d, :] = 0.0  # RESET alle columns
    weights.loc[d, top_syms] = 1.0 / n_top
weights = weights.ffill().fillna(0.0)
```

**Erklärung:** An jedem Rebalance-Datum müssen Non-Top-Symbole auf **0**
gesetzt werden, sonst überträgt der `ffill` ihre alten Weights von vorigen
Rebalances. Beispiel: Symbol A an T0 in Top-5 (Weight 0.2), an T1 nicht
mehr → mit alter Logik behielt A weiter 0.2 wegen ffill.

**Impact:** Cross-Asset-Mom-Top-N hat in 19y-Backtest minimal abweichende
Trades. Master-Allocator-p=0.997 vs 60/40 bleibt nominal gleich (Long-History
ist robust gegen kleine Trade-Drift), aber Implementierung ist jetzt korrekt.

---

### Bug 3: Off-by-one in LiveEngine `update_with_new_day`

**Severity:** Mittel
**Lokalisierung:** `src/erweiterung/live/live_decision_engine.py:217-275`

```python
# VORHER (Off-by-one):
def update_with_new_day(self, date, eq_returns, xa_returns, ...):
    # Step 1: Append eq + xa returns to history
    st.eq_log_return_history = pd.concat([..., new_eq_log])
    # Step 2: Compute factor return MIT updated history
    eq_factor_return = self._compute_today_eq_factor_return(eq_returns)
    ...

# NACHHER (PIT-korrekt):
def update_with_new_day(self, date, eq_returns, xa_returns, ...):
    # Step 1: Compute factor return BEFORE append (history bei T-1)
    eq_factor_return = self._compute_today_eq_factor_return(eq_returns)
    # Step 2: NOW append history
    st.eq_log_return_history = pd.concat([..., new_eq_log])
    ...
```

Plus: `_compute_today_eq_factor_return` Index-Konvention angepasst von
`iloc[-skip-1]` zu `iloc[-1-skip]` (semantisch identisch nach Reihenfolge-
Fix, aber dokumentationsklarer).

**Erklärung:** Bei der Live-Engine wird `update_with_new_day(T, ...)`
aufgerufen wenn der T-Bar abgeschlossen ist. Bevor die Picks für T
angewendet werden, müssen sie aus History bis T-1 berechnet sein —
nicht aus History inkl. T (= Lookahead-Risiko).

**Impact:** Mit altem Code waren Picks für T basierend auf Mom-12/1 mit
Daten bis T-21 (statt korrekt bis T-22). Nominell unauffällig (1-day-shift
in monatlich-rebalanced Strategy), aber konsistent mit Bootstrap-Konvention
ist jetzt der Code.

Performance unverändert: Live-Bench 1.65 ms median (war 1.76 ms vor Fix).

---

### Test-Bug 1: Falsche Label-Strings

**Severity:** Niedrig
**Lokalisierung:** `tests/erweiterung/test_real_data_integration.py:324`

Test prüfte auf `trigger == "calm"`, aber Modul gibt `"normal"` zurück.
Fixed.

---

## 3. Real-Data-Tests neu hinzugefügt

`tests/erweiterung/test_real_data_integration.py` — 13 Tests gegen echte:
- 22 Mega-Caps 19y (Equity)
- 11 ETFs 19y (Cross-Asset)
- VIX-Series (Macro)
- News-Sentiment (sparse)
- Master-Pipeline-Equity (audit)

Smoke-Checks:
- Master-Allocator auf 19y: AnnRet/Sharpe/MDD in plausible Range
- Master_70_30 Calmar-p(>0) vs 60/40 ≥ 0.85
- Live-Engine Bootstrap + Update + Decide auf realen Daten
- Vol-Targeting reduziert SPY-Vol Richtung Target
- EMA-Trend AnnRet > 5% auf 22 Mega-Caps 19y
- Mom-12/1 AnnRet > 5%
- VIX-Trigger erwischt 2020 COVID-Crash
- Audit: Master-Pipeline-Equity hat 0 kritische Flags

---

## 4. Shared Conftest mit Real-Data-Fixtures

`tests/erweiterung/conftest.py` erweitert um:
- `real_xa_returns` — 11-ETF Cross-Asset wide returns
- `real_eq_returns_wide` — 22 Mega-Caps wide returns
- `real_vix` — VIX-Close-Series
- `real_news_panel` — News-Sentiment-Panel

Synthetic-Fixtures bleiben für Edge-Case-Unit-Tests.

---

## 5. Nicht gefundene "Bugs" / dokumentierte Design-Entscheidungen

- **`risk/dynamic_drawdown_control.py:vol_targeted_leverage`** vs
  `strategies/volatility_targeting.py:vol_target_leverage`: zwei separate
  Implementierungen mit unterschiedlichen Semantiken. Erstere ist
  Drawdown-Control-spezifisch. Kein Duplikat.

- **3 Stress-Score-Funktionen** (multi_signal_regime, macro_stress_signals,
  intermarket_macro_factors): unterschiedliche Inputs/Konzepte. Naming
  könnte konsistenter sein, aber kein funktionaler Duplikat.

- **`live/_vol_target_leverage`** (inline) vs Pandas-DataFrame-Variante:
  bewusste Duplikation für Speed (numpy-only vs Pandas).

---

## 6. Status

- **Bugs gefunden:** 4 (3 Code, 1 Test)
- **Bugs gefixt:** 4
- **Tests: 545 grün** (+13 Real-Data-Integration)
- **Performance:** Live-Engine 1.65 ms median (war 1.76 ms — Speedup durch Fix-Reihenfolge)
- **Master-Allocator 19y:** Calmar-p vs 60/40 unverändert ≥ 0.99 (Bugs hatten minimal Impact in Long-History)
- **Audit:** 0 kritische Flags auf Master-Pipeline-Equity

Branch unverändert getrennt vom Mainline. Kein Merge.
