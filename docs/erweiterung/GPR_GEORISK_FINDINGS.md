# GPR-GeoRisk-Overlay — Wertvolle Daten, schwacher Trading-Edge

**Stand:** 2026-05-11
**Branch:** ERWEITERUNG
**Module:**
- `src/erweiterung/altdata/caldara_iacoviello_gpr.py`
- `src/erweiterung/risk/gpr_overlay.py`

**Datenquelle:** Caldara & Iacoviello (2018), "Measuring Geopolitical Risk",
FED Working Paper. URL: matteoiacoviello.com/gpr_files/data_gpr_export.xls.

---

## 1. Motivation (Mainline-PR-Pfad)

Mainline `src/assembled_core/features/geopolitical_features.py` enthält:

> "If Caldara-Iacoviello GPR data is available via FRED (GPRH, GPRC), those
> are used directly. Otherwise, a proxy is constructed from the intel
> pipeline outputs."

**Aber:** Mainline implementiert **keinen Loader** für echte GPR-Daten —
nur den Proxy aus GDELT. Diese Erweiterung füllt die Lücke.

PR-kompatible API:
- `compute_exposure_multiplier(ctx, policy)` — identische Signatur zu
  `src/assembled_core/risk/georisk_overlay.compute_exposure_multiplier`
- Output-Schema matches `compute_gpr_proxy`:
  `gpr_level, gpr_zscore, gpr_momentum, gpr_regime`
- State-Hints `{WATCH, ACTIVE, COOLDOWN, PAUSE}` matchen Mainline-State-Machine

---

## 2. Daten-Validierung (was funktioniert)

**1516 Monthly Records, 1900-01 → 2026-04** — echte Multi-Decade-Daten.

| Event | Datum | GPR | Validierung |
|-------|-------|----:|:------------|
| Gulf War I | 1990-08 | 250 | ✓ |
| Operation Desert Storm | 1991-01 | 379 | ✓ |
| 9/11 | 2001-09 | 498 | ✓ |
| 9/11 Aftermath | 2001-10 | **513** | ✓ (Allzeit-Spike) |
| Iraq War | 2003-03 | 358 | ✓ |
| Russia-Ukraine | 2022-02+ | >150 elevated | ✓ |

Historisches Mean: GPR=103.8, GPRH=99.8. State-Hints verteilen sich
auf 19y: PAUSE 16.8%, ACTIVE 20.1%, WATCH 28.1%, COOLDOWN 35.0%.

**Daten sind valide.** Tests prüfen 9/11 und Ukraine-Spikes auf Schwellen.

---

## 3. Trading-Edge-Test: GPR-Overlay auf Master_V1

Master_V1 = SA-VolTarget (Mom-12/1) 70 % + Cross-Asset-Hybrid 30 %.

### Performance (19y, 4589 days)

| Strategy | AnnRet | Sharpe | Sortino | Calmar | MDD |
|----------|-------:|-------:|--------:|-------:|----:|
| 60/40 Classic | +8.02 % | +0.696 | +0.664 | +0.241 | −33.24 % |
| **Master_V1 (no overlay)** | **+14.46 %** | **+1.209** | +1.140 | **+0.741** | −19.52 % |
| Master_V4 (V1 + GPR) | +13.06 % | +1.205 | +1.136 | +0.693 | **−18.85 %** |

### Calmar-Bootstrap

| Test | p(>0) |
|------|------:|
| V1 vs 60/40 | **0.997** ✓ |
| V4 vs 60/40 | 0.996 ✓ |
| V4 vs V1 | **0.209** (nicht signifikant) |

**Befund:** GPR-Overlay auf Master_V1 ist **nicht signifikant besser**.
MDD-Reduktion (−0.67 pp) kostet 1.40 pp AnnRet. Sharpe nahezu identisch.

### Sub-Period

| Periode | V1 AnnRet | V4 AnnRet | V1 MDD | V4 MDD | ΔMDD |
|---------|----------:|----------:|-------:|-------:|-----:|
| GFC 2008 | −9.00 % | **−11.88 %** | −14.76 % | **−16.27 %** | −1.51 pp |
| COVID 2020 | −21.32 % | −21.34 % | −16.99 % | −17.34 % | −0.35 pp |
| Ukraine 2022 | −3.35 % | −5.48 % | −13.07 % | **−12.01 %** | +1.06 pp |

GPR-Overlay verschlechtert GFC- und COVID-Performance leicht.

---

## 4. Trading-Edge-Test: GPR-Overlay auf Pure-Mom (Mainline-Style)

Hypothese: vielleicht hilft GPR auf einer NICHT-Vol-Targeted Strategy
(wie Mainline-Faktor-Strategien).

| Strategy | AnnRet | Sharpe | Calmar | MDD |
|----------|-------:|-------:|-------:|----:|
| Pure-Mom-12/1 LO | +29.05 % | +1.108 | +0.574 | −50.60 % |
| Pure-Mom + GPR | +25.87 % | +1.073 | +0.520 | −49.78 % |

Calmar-Bootstrap p(>0) = **0.109** — auch hier nicht signifikant.

| Periode | PM AnnRet | PM+GPR AnnRet | PM MDD | PM+GPR MDD |
|---------|----------:|--------------:|-------:|-----------:|
| GFC | −29.50 % | **−36.04 %** | −41.50 % | −43.34 % |
| COVID | +8.54 % | +11.78 % | −32.06 % | −32.48 % |
| Ukraine | −0.71 % | −6.70 % | −22.34 % | −20.64 % |

GPR-Overlay reduziert in 2 von 3 Tail-Events sowohl AnnRet als auch
verschlechtert MDD leicht.

---

## 5. Ehrliche Interpretation

### Warum scheitert GPR-Overlay als Trading-Edge?

1. **Monthly-Latency**: GPR-Daten sind Monthly (ffilled daily). Bei einem
   schnellen Crash (Vol-Mageddon, COVID-March) reagiert GPR erst Wochen
   nach Equity-Crash. **De-Risking nach Tief = Rebound verpassen.**

2. **GPR mean-reverts NACH Spike**: 2001-10 = 513 (peak), 2001-11 = 307.
   Strategy würde gerade dann Cash halten, wenn Equity-Bounce kommt.

3. **GPR ist News-getrieben, nicht market-getrieben**: Geopolitik kann
   eskalieren ohne dass Märkte zu fallen beginnen (Beispiel 2022
   Februar: Märkte fielen bereits Januar, GPR-Spike erst Februar).

4. **Master_V1 hat schon Vol-Target**: zusätzliche GPR-Reduktion =
   Doppelhedging (konsistent mit VIX-Tail-Hedge-Befund).

### Warum ist das Modul trotzdem wertvoll?

1. **Echte Multi-Decade-Daten** (1900+) als Mainline-Ergänzung:
   die Mainline-Doc sagt "If GPR available, use directly" — jetzt ist
   es verfügbar.

2. **GPR ist akademisch validiert** für **Risk-Disclosure** und
   **Macro-Reporting**, auch wenn nicht für Trading-Overlay.

3. **Kompatible API**: `compute_exposure_multiplier(ctx, policy)` matches
   Mainline-Signatur — drop-in nutzbar.

4. **State-Hints** matchen Mainline-State-Machine-Labels — kann mit
   `risk_state` in `trading_cycle_v2` integriert werden.

---

## 6. Empfehlung für Mainline-PR

**Was portieren:**

- `src/erweiterung/altdata/caldara_iacoviello_gpr.py` →
  `src/assembled_core/data/altdata/caldara_iacoviello_gpr.py` —
  ergänzender Daten-Provider neben GDELT/Finnhub.

- `compute_gpr_features()` ergänzt
  `src/assembled_core/features/geopolitical_features.compute_gpr_proxy`:
  echte GPR-Werte statt Proxy.

**Was NICHT portieren (mit Begründung):**

- GPR-Overlay als Master-Replacement: dokumentiert als negativ-getestet
  (p=0.21 bzw 0.11 in zwei Setups). Lehre: GPR ist für News-Risk-
  Disclosure besser geeignet als für Direct-Trading-Overlay.

**Möglicher zukünftiger Pfad:**

GPR könnte als 1-von-N-Faktor in `multifactor_v2`'s `geo_risk_composite`
ergänzt werden — dort hat es kleines Gewicht (~5 %) und schadet
weniger durch Mean-Reversion-Effekt.

---

## 7. Test-Coverage

`tests/erweiterung/test_gpr_overlay.py` — 12 Tests:
- Daten-Loader
- expand_to_daily forward-fill
- compute_gpr_features Schema + Range
- state_hint Mapping
- Apply-Overlay korrektheit
- Mainline-kompatible API-Signatur
- **9/11 Historisch-Test: GPR > 300** ✓
- **Ukraine-2022 Historisch-Test: GPR > 150** ✓

---

## 8. Output-Artefakte

- `data/cache/gpr/sheet1.parquet` — 1516 Monthly GPR rows (1900-2026)
- `output/erweiterung_master_v4_gpr_equity.csv` — V1 + V4 + Multiplier
- `output/erweiterung_master_v4_gpr_summary.json` — Metriken + State-Verteilung
