# Model Card – TEMPLATE

**Template für Model Cards im Assembled Trading AI System.**

Dieses Template sollte für jedes Modell/Strategie kopiert und mit spezifischen Informationen ausgefüllt werden.

---

## 1. Model Overview

**Model ID:** `[model_id]`  
**Type:** `[STRATEGY | QA_METRICS | RISK | SCENARIO | ML_TOOLING | PIPELINE]`  
**Tier:** `[TIER_1 | TIER_2 | TIER_3]`  
**Status:** `[active | pilot | retired]`  
**Owner:** `[Name]`  
**Created:** `[YYYY-MM-DD]`  
**Last Updated:** `[YYYY-MM-DD]`

### Purpose

[1-2 Absätze beschreiben, was das Modell macht und welchen Zweck es erfüllt.]

### Key Components

- [Komponente 1: Kurzbeschreibung]
- [Komponente 2: Kurzbeschreibung]
- [Komponente 3: Kurzbeschreibung]

---

## 2. Data & Features

### Data Sources

**Primary Data:**
- [Datenquelle 1]: [Beschreibung, Format, Frequenz]
- [Datenquelle 2]: [Beschreibung, Format, Frequenz]

**Secondary Data (optional):**
- [Datenquelle 3]: [Beschreibung, Format, Frequenz]

### Feature Groups

**Technical Analysis Features:**
- [Feature-Gruppe 1]: [Beschreibung, z.B. "Moving Averages (20, 50, 200)"]
- [Feature-Gruppe 2]: [Beschreibung, z.B. "RSI, ATR"]

**Event Features (if applicable):**
- [Feature-Gruppe 3]: [Beschreibung, z.B. "Insider Net Buy 20d, Shipping Congestion Score 7d"]

**Other Features:**
- [Weitere Features]: [Beschreibung]

### Data Requirements

- **Minimum Data Period:** [z.B. "252 trading days (1 year)"]
- **Required Symbols:** [z.B. "All symbols in universe"]
- **Data Quality Checks:** [z.B. "No missing values in price data, valid timestamps (UTC)"]

---

## 3. Training & Validation

### Backtest Configuration

**Standard Parameters:**
- Start Capital: `[default: 10000.0]`
- Frequency: `[1d | 5min]`
- Cost Model: `[Commission: X bps, Spread: Y, Impact: Z]`

**Strategy-Specific Parameters:**
- Parameter 1: `[Name]` = `[Value]` (Beschreibung)
- Parameter 2: `[Name]` = `[Value]` (Beschreibung)

### Validation Methodology

**Backtest Periods:**
- Training Period: `[Start]` to `[End]` (if applicable)
- Validation Period: `[Start]` to `[End]`
- Out-of-Sample Period: `[Start]` to `[End]` (if available)

**Walk-Forward Analysis (if applicable):**
- Train Window: `[Size, e.g., "252 days (1 year)"]`
- Test Window: `[Size, e.g., "63 days (1 quarter)"]`
- Step Size: `[Size, e.g., "21 days (1 month)"]`
- Window Type: `[rolling | expanding]`

### Performance Metrics

**Key Metrics Evaluated:**
- Sharpe Ratio (annualized)
- Sortino Ratio (annualized)
- CAGR (Compound Annual Growth Rate)
- Max Drawdown (absolute and percentage)
- VaR 95% / ES 95%
- [Weitere modellspezifische Metriken]

**QA Gates:**
- Sharpe Ratio Threshold: `[min: X, warning: Y]`
- Max Drawdown Limit: `[block: -X%, warning: -Y%]`
- Turnover Threshold: `[max: X, warning: Y]`
- [Weitere Gate-Thresholds]

### Validation Results

**Last Validation Date:** `[YYYY-MM-DD]`  
**Validation Method:** `[Backtest | Walk-Forward | Out-of-Sample | etc.]`

**Performance Summary:**
- Sharpe Ratio: `[Value]`
- CAGR: `[Value]`
- Max Drawdown: `[Value]`
- [Weitere relevante Metriken]

---

## 4. Assumptions & Limitations

### Key Assumptions

1. **[Annahme 1]:** [Beschreibung, z.B. "Market prices are available with daily frequency"]
2. **[Annahme 2]:** [Beschreibung, z.B. "Transaction costs follow default cost model"]
3. **[Annahme 3]:** [Beschreibung, z.B. "All orders execute at specified prices (no slippage beyond cost model)"]

### Known Limitations

1. **[Limitation 1]:** [Beschreibung, z.B. "Model assumes perfect execution, no market impact beyond cost model"]
2. **[Limitation 2]:** [Beschreibung, z.B. "Limited to historical data availability - may not capture regime changes"]
3. **[Limitation 3]:** [Beschreibung, z.B. "Event data is synthetic/sample - real data quality may differ"]

### Usage Constraints

- **Not Suitable For:** [Beschreibung, z.B. "High-frequency trading (< 5min), illiquid markets"]
- **Recommended For:** [Beschreibung, z.B. "Equity markets, daily/5min frequency, liquid symbols"]
- **Warning Conditions:** [Beschreibung, z.B. "Monitor performance if Sharpe < 0.5 or Max DD > -20%"]

---

## 5. Risk & Governance

### Tier Classification

**Tier:** `[TIER_1 | TIER_2 | TIER_3]`

**Rationale:**
[Begründung für Tier-Klassifikation, z.B. "TIER_1 da produktiv relevant und direkt in Trading-Entscheidungen involviert"]

### Risk Factors

**Operational Risk:**
- [Risiko 1]: [Beschreibung, z.B. "Data quality issues could lead to incorrect signals"]
- [Risiko 2]: [Beschreibung, z.B. "Model degradation over time without recalibration"]

**Model Risk:**
- [Risiko 1]: [Beschreibung, z.B. "Parameter sensitivity - performance depends on MA windows"]
- [Risiko 2]: [Beschreibung, z.B. "Assumptions about market regime may not hold"]

**Market Risk:**
- [Risiko 1]: [Beschreibung, z.B. "Limited to historical scenarios - may not cover tail events"]

### Monitoring & Drift Detection

**Performance Monitoring:**
- [Metrik 1]: [Schwellwert, z.B. "Sharpe Ratio < 0.5 → Warning"]
- [Metrik 2]: [Schwellwert, z.B. "Max Drawdown > -20% → Block"]

**Data Drift Detection (if applicable):**
- [Check 1]: [Beschreibung, z.B. "Feature distribution changes > X%"]
- [Check 2]: [Beschreibung, z.B. "Missing data rate > Y%"]

**Kill Switch / Pre-Trade Dependencies:**

- QA Gates: `[Block if QA Gates overall_result = "BLOCK"]`
- Data Quality: `[Block if prices/orders missing or invalid]`
- Performance Thresholds: `[Block if Sharpe < X or Max DD > Y]`

### Governance

**Review Schedule:** `[z.B. "Quarterly" | "Bi-annually" | "As needed"]`  
**Next Review Date:** `[YYYY-MM-DD]`  
**Approval Required For:**
- Parameter changes: `[Yes/No]`
- Deployment to production: `[Yes/No]`
- [Weitere Governance-Anforderungen]

---

## 6. Change History

| Date | Version | Author | Change Description |
|------|---------|--------|-------------------|
| YYYY-MM-DD | 1.0.0 | [Name] | Initial model card creation |
| YYYY-MM-DD | 1.1.0 | [Name] | [Beschreibung der Änderung] |
| YYYY-MM-DD | 1.2.0 | [Name] | [Beschreibung der Änderung] |

**Version Format:** `MAJOR.MINOR.PATCH`
- MAJOR: Breaking changes or fundamental modifications
- MINOR: New features or significant improvements
- PATCH: Bug fixes or minor adjustments

---

## 7. References & Related Documentation

**Code Location:**
- Module: `[z.B. "src/assembled_core/qa/metrics.py"]`
- Tests: `[z.B. "tests/test_qa_metrics.py"]`

**Related Models:**
- [Verwandtes Modell 1]: [Link/Kurzbeschreibung]
- [Verwandtes Modell 2]: [Link/Kurzbeschreibung]

**Documentation:**
- [Dokument 1]: [Link/Kurzbeschreibung]
- [Dokument 2]: [Link/Kurzbeschreibung]

---

## Notes

[Zusätzliche Notizen, bekannte Issues, zukünftige Verbesserungen, etc.]

