# Meta-Labeling auf Master-Allocator — Negativer Befund

**Stand:** 2026-05-11
**Branch:** ERWEITERUNG
**Module:** `src/erweiterung/ml/meta_labeling_master.py`
**Script:** `scripts/erweiterung/run_meta_labeling_master.py`
**Referenz:** Lopez de Prado (2018), "Advances in Financial Machine Learning", §3.6.

---

## 1. Hypothese

Master-Allocator (Master_70_30) liefert In-Sample Calmar 1.10. Idee: ein
Meta-Klassifikator (Logistic / Random Forest) lernt, **wann** der Master-
Allocator profitabel sein wird, basierend auf Trailing-Features. An "go"-
Tagen wird allokiert, sonst Cash gehalten.

Pipeline:
1. Triple-Barrier-Labels auf Master-Returns (±2.5 %, horizon 21 d).
2. Features: Trailing-Vol, Trailing-Sharpe, Drawdown, VIX, Yield-Curve.
3. Walk-Forward 18-Monat-Train → 3-Monat-Test, beide Modelle parallel.
4. Meta-Gate: trade nur bei `predict=1`.

---

## 2. Ergebnis

### Label-Verteilung

- +1 (TP traf): 562 (43 %)
- −1 (SL traf): 325
- 0 (Time-out): 427
- Base-Rate für binäres y=1: 50 %

### Klassifikator-Performance OOS (315 Samples)

| Modell | Accuracy | Trade-Anteil | Base-Rate |
|--------|---------:|-------------:|----------:|
| Logistic | **0.337** | 45.7 % | 0.498 |
| Random Forest | **0.410** | 52.4 % | 0.498 |

**Beide Modelle sind schlechter als Coin Flip** (0.50 base-rate). Random
Forest ist näher dran, aber Logistic ist deutlich anti-prädiktiv.

### Performance-Vergleich (OOS, 315 Tage)

| Strategy | AnnRet | Sharpe | Sortino | Calmar | MDD |
|----------|-------:|-------:|--------:|-------:|----:|
| 60/40 Classic | +12.44 % | +1.096 | +1.088 | +1.104 | −11.27 % |
| Pure Mom-12/1 LO | +29.17 % | +1.058 | +1.025 | +1.079 | −27.04 % |
| **Master_70_30 (pure)** | **+19.20 %** | **+1.482** | **+1.385** | **+1.632** | −11.77 % |
| Master + Logistic Gate | +4.27 % | +0.513 | +0.322 | +0.363 | −11.77 % |
| Master + RandomForest Gate | +7.14 % | +0.817 | +0.538 | +0.751 | −9.51 % |

### Calmar-Bootstrap vs Pure Master_70_30 (OOS)

| Challenger | Observed Δ | Mean Δ | p(>0) |
|------------|-----------:|-------:|------:|
| 60/40 Classic | −0.528 | −0.904 | 0.191 |
| Pure Mom-12/1 LO | −0.553 | −0.665 | 0.176 |
| **Master + Logistic Gate** | −1.269 | −1.938 | **0.020** ← statistisch signifikant SCHLECHTER |
| **Master + RandomForest Gate** | −0.881 | −1.469 | **0.062** ← knapp signifikant schlechter |

**Beide Meta-Gate-Varianten sind statistisch signifikant *schlechter* als der
ungated Master.** Das ist eine klare Falsifikation der Hypothese.

---

## 3. Ehrliche Lesart

### Warum funktioniert Meta-Labeling hier nicht?

1. **Vol-Targeting glättet die Returns:** Master-Allocator nutzt schon
   Vol-Targeting, was Magnitude-Variation reduziert. Damit fehlt dem
   Meta-Klassifikator das "Asymmetrie-Signal", das er normalerweise braucht.

2. **Trailing-Features sind reaktiv:** Trailing-Vol, Sharpe, Drawdown
   sind alle reaktiv — sie messen, was *war*, nicht was *kommt*. Ohne
   Lead-Indikatoren kann der Klassifikator nicht zwischen "next 21d
   profitable" und "next 21d loss" trennen.

3. **Sample-Größe zu klein:** 1315 Tage gesamt, nach Train-Window-Aufbau
   nur 315 OOS-Samples. Klassische Meta-Labeling-Anwendungen brauchen
   10 000+ Samples (intraday) für robuste Klassifikatoren.

4. **Vol-Targeted Master-Returns sind dichter normal verteilt** als
   typische Pure-Mom-Returns. Damit ist weniger "Signal" in der Verteilung,
   das ein Klassifikator nutzen kann.

### Was diese Negativ-Erkenntnis bedeutet

- **Don't add ML on top of well-designed deterministic strategies** ohne
  vorher zu prüfen, ob die Features Signal haben.
- **Master_70_30 ist als-ist gut genug** — Meta-Layer ist Overengineering.
- **Diese Falsifikation ist wertvoll**: sie verhindert, dass wir ein
  prima-facie-attraktives ML-Konzept blind als "Verbesserung" deklarieren.

---

## 4. Was diese Erkenntnis NICHT bedeutet

- Meta-Labeling ist nicht generell wertlos. Lopez de Prado zeigt es
  funktioniert für *Primary-Signals mit erkennbarer Direction*. Master-
  Allocator hat keine direktionale Primary-Vorhersage, sondern konstante
  long-Exposure. Damit fehlt die Voraussetzung für Meta-Labeling.

- Mit anderen Features (Volume, Options-IV-Skew, Cross-Asset-Stress) und
  längerem Sample (50 000+ Tage über Asset-Klassen) könnte das Setup
  durchaus funktionieren.

- Pure-Mom-12/1 LO (ohne Vol-Target) wäre ein besseres Primary-Signal für
  Meta-Labeling, weil hier die Direction echte Information trägt.

---

## 5. Konsequenzen für die Erweiterungs-Roadmap

1. **Master_70_30 bleibt die finale Allokations-Strategie.** Meta-Layer
   wird nicht aufgenommen.
2. **Meta-Labeling-Modul bleibt im Repo** als Building-Block für andere
   Primary-Signals (z. B. Cross-Asset-Mom-Top-5), die noch nicht Vol-
   Targeted sind.
3. **Nächster sinnvoller ML-Versuch:** Lead-Indikator-Features statt
   Trailing — z. B. Sentiment-Drift, Implied-Vol-Term-Structure, Macro-
   Surprise-Indizes.

---

## 6. Output-Artefakte

- `output/erweiterung_meta_labeling_summary.json` — Metriken + Label-Verteilung
- `output/erweiterung_meta_labeling_predictions_lr.csv` — Logistic-Predictions
- `output/erweiterung_meta_labeling_predictions_rf.csv` — RF-Predictions
