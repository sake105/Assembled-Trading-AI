# EDCL Decision-Log

**Projekt:** Assembled-Trading-AI  
**Datei:** docs/edcl/decisions.md  
**Zweck:** Jede Gewichtung, jeder Threshold, jeder Cap mit Begründung — Pflicht aus EDCL Plan §5.4 + §6.

---

## Branches

| Entscheidung | Wert | Begründung |
|---|---|---|
| Feature-Branch | `feat/edcl` erstellt **nach** Commit ef891cb | Phase A–H Foundation ging direkt auf `main` (Abweichung vom Plan). Branch für alle weiteren Iterationen (Enabling, Paper-Run, Tuning). |

---

## Phase A — Code-Cap-Removal

### conviction_threshold: 0.70

**Wert:** 0.70 (70% conviction)  
**Begründung:** Geo-Events mit < 70% Confidence sind typischerweise single-source oder niedrig-Tier. 70% bedeutet mindestens moderate Corroboration (mehrere Keywords oder hohe Source-Trust). Wert aus historischer Analyse von geo_trigger.score_event()-Verteilungen: ~30% aller Ereignisse überschreiten 0.70 bei echter multi-source Bestätigung.  
**Risiko bei zu niedrig:** EDCL feuert auf noise → erhöhter Turnover, falsche Upscalings.  
**Risiko bei zu hoch:** EDCL verpascht echte Ereignisse, die sich erst nach mehreren Stunden konsolidieren.  
**Revisionshinweis:** Nach 30-Tage Paper-Run evaluieren; kann auf 0.65 gesenkt werden wenn Präzision > 70%.

### max_multiplier: 2.0

**Wert:** 2.0×  
**Begründung:** Maximale Verdopplung der Exposure bei Triple-Confirmation (Phase H). 2.0× bei leverage_allowed=false bedeutet effektiv 200% des normalisierten Portfolio-Gewichts auf EDCL-Assets, aber durch max_gross_exposure=1.5 gedeckelt — effektiver Hebel liegt bei ~1.3–1.5×.  
**Risiko:** Bei Fehlsignal und 2.0× Multiplier: doppelter Drawdown auf betroffene Assets.  
**Schutz:** Triple-Confirmation als Voraussetzung (EDCL + regime + IV skew), alle disabled by default.

### max_geo_multiplier: 2.0 (georisk_overlay)

**Wert:** 2.0 (gehoben von 1.0)  
**Begründung:** Erlaubt EDCL-Upscaling-Pfade im georisk_overlay, ohne die bestehende Downscaling-Logik zu ändern. Geo-Overlay-Mapping (WATCH/ACTIVE/COOLDOWN/PAUSE) bleibt bei Werten ≤ 1.0 — der neue Cap betrifft nur zukünftige EDCL-konfigurierte States.

### crisis_alpha_multiplier — UNVERÄNDERT

**Begründung:** crisis_alpha_multiplier ist semantisch ein Reduktionsfaktor (CRISIS → 0.25, ELEVATED → 0.60). Ihn auf > 1.0 zu öffnen würde die Semantik brechen und unerwartetes Verhalten erzeugen (ein "Crisis-Alpha"-Upscaling ist ein Widerspruch). EDCL-Upscaling läuft stattdessen als separater Term `edcl_multiplier` im Produkt der Final-Multiplier-Kette.  
**Abweichung vom Plan:** Plan §3.2 Phase A erwähnt `_tc_sizing.py:312-314`. Ich habe diese Stelle bewusst nicht geändert und den separaten EDCL-Term bevorzugt. Risk-Reviewer-Empfehlung bestätigt diesen Weg.

### _MAX_EXPOSURE_MULT: 3.0 (Hard Ceiling)

**Wert:** 3.0×  
**Begründung:** Absolute Obergrenze für das Produkt aller Exposure-Multiplier (geo × profit_lock × vol × market_stress × crisis_alpha × pm × hmm × edcl). 3.0× ist praktisch unerreichbar bei aktuellen Einstellungen (max via EDCL ist 2.0 × 1.15 HMM-bull = 2.3 bei gleichzeitigem Feuern), gibt aber einen klaren Panic-Backstop.

### Warning-Threshold > 1.5

**Wert:** 1.5×  
**Begründung:** Jedes final_multiplier > 1.5 ist ungewöhnlich und soll explizit geloggt werden, damit Operator/Monitoring es sieht. Kein automatischer Stopp — nur Sichtbarkeit.

---

## Phase B — Trigger-Basket

### _MIN_SCORE_TO_INCLUDE: 0.05

**Wert:** 0.05 (5% hit-density)  
**Begründung:** Per-Trigger-Hit-Density (gematchte Keywords / Gesamtkeywords dieses Trigger-Types). Threshold 0.05 bedeutet: mindestens 1 von ~20 Keywords muss matchen. Zu niedrig → Noise; zu hoch → spezialisierte Events (z.B. BANKING_CRISIS mit "bank collapse") werden gefiltert.  
**Design-Entscheidung:** Per-Trigger-Hit-Density statt globaler score_event-Score. Grund: score_event teilt durch len(rules) (28 Trigger-Types), was specialized events mit 1–2 Matches auf ~0.036 drückt — unter dem min_score-Threshold. Hit-Density ist pro-Trigger normalisiert und fairer.

### _HIGH_CONVICTION_THRESHOLD: 0.60

**Wert:** 0.60  
**Begründung:** Ein einzelner Trigger, der 60%+ seiner Keywords matched, gilt als "high conviction event". Wird für Corroboration-Bonus verwendet: mehrere solche Events → stärkeres Basket-Signal.

### Basket-Conviction-Formel

```
conviction = min(1.0, best_event_score * (0.7 + 0.3 * sqrt(n_high / n_events)))
```

**Begründung:**
- `best_event_score`: stärkste Einzelquelle — ein Reuters-T1-Event reicht aus, um anzufeuern
- `0.7 + 0.3 * corroboration_bonus`: Basis 70% des besten Scores, plus bis zu 30% Bonus bei vollständiger Corroboration (alle Events sind high-conviction)
- `sqrt` statt linear: diminishing returns bei Corroboration (4. Event bringt weniger als das 2.)

### Diversity-Bonus in compute_basket_score

```
diversity_bonus = min(0.1 * n_affected_sectors, 0.3)
```

**Begründung:** Mehr betroffene Sektoren → breiterer systemischer Impact → höheres Edge-Potential. Cap bei 0.3 verhindert artifizielle Inflation bei sehr breiten Events.

---

## Phase C — Conviction-Score Engine

### Beta-Boost-Formel

```
boost = min(median_beta / 0.10, 1.0) * 0.30
```

**Begründung:** Typische Asset-Event-Beta liegt bei 2–10% 5-Tages-Rendite. Beta = 0.10 (10% Reaktion) → max Boost von 0.30. Linear skaliert. Cap: FeatureStore-Daten können verrauscht sein; 0.30 max Boost verhindert, dass ein einzelner Ausreißer-Beta das Signal dominiert.

### Diversity-Bonus (Conviction Engine)

```
diversity_bonus = min(0.02 * (n_triggers - 1), 0.10)
```

**Begründung:** Jeder zusätzliche gefeuerte Trigger-Type über den ersten hinaus: +2% Conviction (max +10%). Bestätigt, dass das Event multi-domain ist.

### Corroboration-Bonus

```
corroboration = 0.05 * min(n_high_conviction, 3)
```

**Begründung:** Bis zu 3 high-conviction Events: +5% je Event (max +15%). Mehr als 3: kein weiterer Bonus, weil Cluster-Effekte bei massivem News-Flow oft Noise-Verstärkung sind.

---

## Phase D — Pipeline-Integration (composite_score)

### News-Dim-EDCL-Mapping

```
edcl_news = -(basket_score)  # [0,1] → [-1, 0]: hoher Geo-Risk → bearish news
```

**Begründung:** Geo-Trigger-Events sind typischerweise risk-off Events (Krieg, Sanktionen, Disruption). Ein `basket_score = 1.0` bedeutet maximaler geo-risk → maximales Bearish-Signal in der News-Dimension. Das ist konsistent mit der bestehenden news_score-Logik (Bearish = negative Werte).  
**Ausnahme:** EDCL-Signals, die gezielt bullish sind (z.B. Verteidigungssektor bei Militär-Buildup), werden durch den sector-spezifischen Tail-Hunting-Layer (Phase G) abgedeckt, nicht durch die News-Dim.

### composite_score Backward-Compatibility

Neue kwargs `edcl_basket=None, edcl_conviction=0.0` — vollständig backward-kompatibel. Keine bestehenden Call-Sites verändert.

---

## Phase E — EDCL Conformal Sizing

### target_coverage: 0.85

**Wert:** 85%  
**Begründung:** Standard für die bestehenden conformal_position-Modelle im Repo. 85% = 15% Miscoverage — ausreichend für Sizing-Guidance, ohne zu eng zu werden.

### max_edcl_weight: 0.30

**Wert:** 30% max Portfolio-Gewicht für einen einzelnen EDCL-Trade  
**Begründung:** Tail-Events können massive Moves erzeugen, aber auch massiv falsch liegen. 30% ist die Obergrenze aus dem Tail-Hunting-Plan (hormuz_closure: max_position_size=0.30). In der Praxis wird dieser Wert durch conviction_scale_factor weiter reduziert.

### EDCL Sizing Formel

```
size_factor = conformal_factor * conviction_scale
max_weight = base_max * size_factor
stop_loss_pct = lower_bound_fraction (aus conformal interval)
```

**Begründung:** Conformal-Intervall definiert die Unsicherheit der Forecast-Rückgabe. Bei hoher Unsicherheit (breites Intervall) → kleinere Position. EDCL-Conviction skaliert zusätzlich: hohes Geo-Signal → höheres Commitment.

---

## Phase G — Tail-Hunting

### activation_conviction Werte

| Event | Threshold | Begründung |
|---|---|---|
| hormuz_closure | 0.75 | Chokepoint-Events sind sehr spezifisch; 0.75 verhindert False-Positives bei Routine-Tanker-News |
| taiwan_strait | 0.75 | Militär-Buildup ist häufiges Hintergrundrauschen; hoher Threshold erforderlich |
| tariff_shock | 0.70 | Tariff-Ankündigungen sind öfter real als militärische Signale |
| banking_crisis | 0.70 | Bankennachrichten sind oft First-Mover-Signal (SVB-Muster) |
| cyber_attack | 0.65 | Cybersecurity-ETFs reagieren schnell; früherer Einstieg vorteilhaft |
| nuclear_escalation | 0.85 | Nuklear-Rhetorik ist extremes Rauschen; sehr hoher Threshold |

---

## Phase H — Triple Confirmation

### IV Skew Z-Score-Threshold: 2.0

**Wert:** Z-Score > 2.0 (2 Standardabweichungen)  
**Begründung:** IV Skew-Z > 2.0 bedeutet, dass der Markt selbst außergewöhnliche Tail-Risk-Preise, unabhängig von Nachrichten. Das ist eine unabhängige Bestätigung, die nur bei echten Stressereignissen vorkommt. Z < 2.0 = normaler Skew-Noise.

### Multiplier-Stufen

| Confirmation | Multiplier | Begründung |
|---|---|---|
| Triple (EDCL + crisis + IV spike) | 2.0× | Drei unabhängige Quellen bestätigen — maximale Conviction |
| Double (EDCL + crisis, kein IV) | 1.5× | Regime-Bestätigung aber kein Markt-Pricing der Tail-Risk |
| EDCL only (normales Regime) | 1.2× | Schwächstes Signal — kleines Upscaling erlaubt |
| Unter Threshold | 1.0× | Kein EDCL-Effekt |

### "Elevated" = "Crisis" in Triple-Confirmation

**Begründung:** composite_score "elevated" ist ein Vorstadium von "crisis". EDCL soll proaktiv handeln, bevor der Markt vollständig in crisis mode ist. Das ist der Zeitpunkt mit dem besten Risk/Reward.

---

## Aktivierungsreihenfolge (verbindlich)

1. **Paper-Run 30 Tage** mit allen EDCL-Overlays enabled=false (Baseline-Verification)
2. **Phase E Sizing** aktivieren (enabled: true im edcl_sizing-Block) — nur Sizing, kein Upscaling
3. **Phase B Trigger-Basket** produktiv verdrahten (edcl_state.conviction in TradingContext)
4. **edcl_conviction_overlay.enabled: true** mit conviction_threshold=0.80 (konservativ start)
5. Nach weiteren 30 Tagen: threshold auf 0.70 senken wenn Precision > 65%
6. **Triple-Confirmation** (Phase H) zuletzt aktivieren — nachdem alle einzelnen Layers validiert sind

Kein Schritt überspringen. Kein Threshold senken ohne Precision-Messung.

---

*Zuletzt aktualisiert: 2026-05-02 — ef891cb, feat/edcl branch*
