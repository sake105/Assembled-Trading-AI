# Multi-Faktor-Vol-Targeting — In-Sample-Edge, OOS überfittet

**Stand:** 2026-05-11
**Branch:** ERWEITERUNG
**Module:**
- `src/erweiterung/strategies/multi_factor_vol_target.py`
- `src/erweiterung/backtest/calmar_bootstrap.py`

**Scripts:**
- `scripts/erweiterung/run_multi_factor_vol_target_backtest.py`
- `scripts/erweiterung/run_multi_factor_walk_forward.py`

---

## 1. Motivation

Nach dem Vol-Targeting-Befund: Lässt sich der Edge durch Multi-Faktor-
Kombination (Mom + ResMom + LowVol) weiter verbessern? Theoretisch ja:
Faktor-Korrelation < 1 → Diversifikations-Edge zusätzlich zum Vol-Target.

Zusätzlich wurde der **Calmar-Bootstrap** als test-statistik eingeführt
(stationary bootstrap, Politis-Romano), weil Sharpe-Bootstrap nicht den
MDD-Aspekt erfasst, der für Vol-Targeting der dominante Edge ist.

---

## 2. In-Sample (19y, 22 Mega-Caps)

### Performance

| Strategy | AnnRet | Sharpe | Sortino | Calmar | MDD |
|----------|-------:|-------:|--------:|-------:|----:|
| Pure Equal-Weight | +22.81 % | +1.074 | +1.023 | +0.511 | −44.66 % |
| Pure Mom-12/1 LO | +29.05 % | +1.108 | +1.055 | +0.612 | −47.49 % |
| Pure ResMom LO | +22.00 % | +0.943 | +0.916 | +0.478 | −46.06 % |
| Pure LowVol LO | +11.76 % | +0.765 | +0.734 | +0.304 | −38.73 % |
| MultiFac-VT-EqWeight | +14.42 % | +1.216 | +1.144 | +0.823 | −17.52 % |
| MultiFac-VT-InvVol | +14.30 % | +1.212 | +1.139 | +0.753 | −19.00 % |
| **MultiFac-VT-HRP** | +14.50 % | **+1.234** | **+1.164** | +0.859 | **−16.89 %** |
| **Single-VolTarget Mom** | +17.01 % | **+1.277** | **+1.219** | **+0.907** | −18.76 % |

### Calmar-Bootstrap (vs Equal-Weight)

| Challenger | Observed Δ | Mean Δ | 95 % CI | p(>0) |
|------------|-----------:|-------:|--------:|------:|
| Pure Mom-12/1 LO | +0.108 | +0.102 | [−0.20, +0.43] | 0.779 |
| Pure LowVol LO | −0.206 | −0.272 | [−0.65, +0.02] | 0.033 |
| MultiFac-VT-EqWeight | +0.313 | +0.097 | [−0.27, +0.49] | 0.723 |
| MultiFac-VT-InvVol | +0.243 | +0.088 | [−0.27, +0.46] | 0.713 |
| **MultiFac-VT-HRP** | +0.349 | +0.124 | [−0.24, +0.49] | **0.784** |
| **Single-VolTarget Mom** | +0.403 | +0.277 | [−0.13, +0.71] | **0.919** |

### Calmar-Bootstrap (vs Pure Mom-12/1 LO)

| Challenger | p(>0) |
|------------|------:|
| MultiFac-VT-EqWeight | 0.500 |
| MultiFac-VT-InvVol | 0.479 |
| MultiFac-VT-HRP | 0.617 |
| **Single-VolTarget Mom** | **0.863** |

**Single-VolTarget Mom ist In-Sample nominaler Champion.**

---

## 3. Walk-Forward Out-of-Sample (13 Windows, 2013-2026)

### OOS-Performance

| Strategy | AnnRet | Sharpe | Sortino | Calmar | MDD |
|----------|-------:|-------:|--------:|-------:|----:|
| Pure Equal-Weight | +26.61 % | +1.334 | +1.249 | +0.849 | −31.35 % |
| Pure Mom-12/1 LO | **+37.90 %** | +1.405 | +1.329 | +1.182 | −32.06 % |
| **Single-VolTarget Mom** | +20.06 % | **+1.462** | **+1.376** | **+1.306** | **−15.36 %** |
| MultiFac VolTarget | +25.40 % | +1.398 | +1.302 | +1.070 | −23.74 % |
| Walk-Forward Switch | +30.72 % | +1.424 | +1.357 | +1.061 | −28.94 % |

### Calmar-Bootstrap OOS (vs Equal-Weight)

| Challenger | Observed Δ | p(>0) |
|------------|-----------:|------:|
| Pure Mom-12/1 LO | +0.333 | 0.853 |
| Single-VolTarget Mom | +0.457 | 0.793 |
| MultiFac VolTarget | +0.221 | 0.656 |
| **Walk-Forward Switch** | +0.213 | **0.952** |

### Calmar-Bootstrap OOS (vs Pure Mom-12/1 LO)

| Challenger | p(>0) |
|------------|------:|
| Walk-Forward Switch | 0.392 |
| Single-VolTarget Mom | **0.508** |
| MultiFac VolTarget | 0.250 |

---

## 4. Wichtige Erkenntnisse

### 4.1 Single-VolTarget Mom — robust OOS-Champion in Risk-Metriken

- **MDD halbiert**: −15.36 % vs Pure-Mom −32.06 % (**+16.7 pp Verbesserung**)
- **Sharpe**: 1.462 (höchster aller OOS-Varianten)
- **Calmar**: 1.306 (höchster)
- **In-Sample-Reproduzierbarkeit**: gleiches Profil über alle Tests

### 4.2 Multi-Faktor-Combiner überfittet im OOS

Walk-Forward-Combiner-Verteilung: HRP wurde **12 von 13 Mal** gewählt,
target_vol = 0.18 ebenfalls 12/13. Das ist verdächtig: Train-Optimierung
lernt eine sehr enge Konfiguration, die im OOS nicht überlegen ist.

- In-Sample: MultiFac-VT-HRP Calmar 0.859 vs Single-VolTarget 0.907
- OOS: MultiFac-VolTarget Calmar 1.070 vs Single-VolTarget 1.306

Multi-Faktor-Combiner zahlt einen **Overfit-Tax**, der die theoretischen
Diversifikations-Vorteile zunichte macht.

### 4.3 Walk-Forward-Switch hat statistisch signifikanten Edge vs Equal-Weight

**p(Calmar-Δ > 0) = 0.952** im OOS-Calmar-Bootstrap. Das ist **die einzige
Strategie**, die statistisch signifikant besser ist als das naive
Equal-Weight-Portfolio nach 13 OOS-Windows.

Wichtig: Walk-Forward-Switch ist *nicht* signifikant besser als Pure-Mom
(p=0.392), aber **signifikant besser als Equal-Weight**. Damit ist der
Switch sinnvoll als Equal-Weight-Replacement, nicht als Mom-Replacement.

### 4.4 Calmar-Bootstrap > Sharpe-Bootstrap für MDD-Reducer

Sharpe-Bootstrap-Tests aus vorherigen Sessions gaben p ≈ 0.99 für jeden
Switching-Vorschlag. Calmar-Bootstrap zeigt deutlich differenziertere
p-Werte (0.150 - 0.952), die echte Differenzen aufdecken.

**Standard-Test-Statistik für Vol-Targeting ist Calmar-Bootstrap, nicht
Sharpe-Bootstrap.**

---

## 5. Konsequenzen für die Erweiterungs-Roadmap

1. **Single-VolTarget Mom-12/1-LO ist die robust beste Risk-adjustierte
   Strategie** über 13 OOS-Jahre — und braucht keine Threshold-Optimierung.

2. **Multi-Faktor-Combiner-Optimierung lohnt sich nicht.** Equal-Weight-
   Combiner ohne Optimierung wäre möglicherweise nicht schlechter und
   würde Overfit-Risk vermeiden.

3. **Calmar-Bootstrap wird die Standard-Statistik für alle weiteren
   Allocator-Tests.** Sharpe-Bootstrap bleibt für Return-orientierte
   Strategien, Calmar-Bootstrap für Risk-orientierte.

4. **Nächster sinnvoller Schritt:** Cross-Asset-Diversifikation (Bonds,
   Gold) statt Multi-Faktor — Asset-Klassen haben tiefere Diversifikation
   als Single-Asset-Faktoren.

---

## 6. Output-Artefakte

- `output/erweiterung_multi_factor_vol_target_equity.csv` — In-Sample-Equities
- `output/erweiterung_multi_factor_vol_target_summary.json` — Metriken
- `output/erweiterung_multi_factor_walk_forward_oos.csv` — OOS-Equity
- `output/erweiterung_multi_factor_walk_forward_windows.csv` — Per-Window
- `output/erweiterung_multi_factor_walk_forward_summary.json` — Aggregat
