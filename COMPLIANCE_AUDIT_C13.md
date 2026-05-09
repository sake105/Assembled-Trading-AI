# MNPI Compliance Audit — C13 Critic
**Assembled-Trading-AI Disclosure Pipeline Review**  
**Date:** 2026-05-09  
**Classification:** Risk Report  

---

## EXECUTIVE SUMMARY

The disclosure pipeline exhibits **one critical unmitigated MNPI risk** and three material timing gaps. The system claims "public data only, no MNPI" but fails to enforce or document disclosure **availability lag** for congressional trades and Form 4 filings. This creates a pathway for look-ahead bias that could constitute material non-public information use.

---

## CRITICAL FINDING: Congressional Trade Lag Undeclared

### The Issue

**SEC Form 4 insider trades** and **House PTR (Periodic Transaction Reports)** are PUBLIC data sources, but they have well-documented **publication delays**:

- **House PTR:** 45-day delay after transaction (mandatory by law; actual reporting often 90+ days)
- **Form 4:** 2 business days after transaction (also mandatory); EDGAR feeds lag further

The system's `congress_features.py` module **does not enforce or document this lag**.

```python
# congress_features.py, line 30:
def add_congress_features(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    disclosure_latency_days: int = 10,  # <-- HARDCODED TO 10 DAYS
) -> pd.DataFrame:
```

**Red flag:** 10 days is **far shorter** than actual House PTR lag (45-90 days). This allows the backtest to use trades that were not yet publicly known.

### Proof

In the current codebase:
- `latency.py` line 80–126 provides a `disclosure_latency_days` parameter.
- `congress_features.py` **defaults to 10 days** (line 30).
- House PTR pipeline config (pipeline.py) does **not override** this; default is used in training (build_factor_panel.py line 28).
- **No validation** checks whether the delay matches actual legislative/regulatory timelines.

### Impact

A trader using this system in **live mode**:
1. Sees a congressional trade in the EDGAR/House PTR feed (publicly disclosed).
2. Computes `congress_total_amount_90d` using only 10-day-delayed data.
3. The trade is factually still **"not yet known"** under House PTR rules (45-90 day window).
4. Acts on a feature that represents undisclosed information → **MNPI violation.**

### Example

- **June 1 (Transaction day):** Rep X buys 10,000 shares of ticker ABC.
- **July 15 (House PTR due):** Disclosure filed (45-day window).
- **July 25 (System sees it):** Pipeline fetches Form 4 / House PTR feed.
- **August 4 (Feature computed):** `congress_total_amount_90d` includes the June 1 trade at 64 days delay.
  - BUT: House PTR is still within the **45–90 day official lag window.**
  - System is using information that **wasn't yet publicly available** on June 15, July 1, or July 10.

---

## MATERIAL FINDING 2: Form 4 Timestamp Ambiguity

**Form 4 metadata mixes three dates:**

1. **`published`** (line 165, fetch_edgar.py) — EDGAR feed timestamp.
2. **`timestamp`** (line 144, fetch_edgar.py) — Not clearly defined in code.
3. **`disclosure_date`** (congress_features.py line 79) — **Never explicitly set** for Form 4 events; defaults to `timestamp + 1 day` (latency.py line 56).

**The problem:**
- Fetching Form 4 from EDGAR at 2:00 PM UTC does **not mean** the trade was known at 2:00 PM.
- The actual **transaction date** (when the insider bought/sold) is buried in the Form 4 PDF and **not extracted** by the current parser (fetch_edgar.py only reads metadata, line 47: "No heavy parsing").
- Feature computation then uses `published + 1 day` as the knowledge cutoff, which is **incorrect**.

**Code evidence:**
```python
# fetch_edgar.py, lines 47–189:
# "No heavy parsing; metadata only."
# Missing: form_4_transaction_date, insider_name, transaction_type, shares, price
```

---

## MATERIAL FINDING 3: Insider Data Corruption (Masked by "Unusable" Label)

**Current status:** Insider data is marked "all 'unknown'" and disabled (memory, session-2026-05-05).

**But the codebase still references it:**
- `insider_features.py` — computes `insider_activity_score` (potentially zero-filled).
- `altdata_earnings_insider_factors.py` — includes insider vega/volume factors.
- `multifactor_v2.py` strategy pulls `insider_activity_score` (signals/multifactor_v2.py, line 152).

**The compliance risk:** If insider data is ever re-enabled (e.g., via a new Finnhub/Alpaca API key or third-party feed), the system will **not enforce disclosure lag** for insider trades. Insider trades have their own **Form 4 + SEC reporting lag**, distinct from general corporate actions.

---

## MATERIAL FINDING 4: No Audit Trail for Disclosure Availability

The pipeline **does not record or validate:**

1. **When** each disclosure was fetched (vs. when it was published).
2. **Whether** the fetch timestamp qualifies as "known" under SEC/House rules.
3. **Which delay assumption** was used for each feature (congress_features.py defaults; not configurable per-source).

**Example audit hole:**
```
June 15, 2026 (backtest timestamp):
  → System fetches EDGAR → sees Form 4 filed June 13.
  → Assumes disclosure_date = June 14 (timestamp + 1 day).
  → NO RECORD of whether June 13 is actually within the Form 4 lag window.
  → NO RECORD of whether this event should have been known on June 15.
```

---

## MISSING SAFEGUARD: No Disclosure Lag Validation

The system does **not validate** that:
- Form 4 lag ≥ 2 business days
- House PTR lag ≥ 45 calendar days
- No form uses transaction_date (only published/timestamp)

**Code gap:**
```python
# No validator like:
def validate_disclosure_lag(events: pd.DataFrame, lag_days: int):
    """Ensure lag_days >= regulatory minimum (45 for PTR, 2 for Form 4)."""
    pass  # Not implemented
```

---

## FORWARD PROPOSAL: MNPI Compliance Audit Mechanism

### Objective
Institutionalize disclosure lag tracking and prevent future accidental use of undisclosed information.

### Components

#### 1. **Disclosure Lag Registry** (configs/disclosures/lag_policy.yaml)
```yaml
sources:
  form4:
    min_lag_days: 2
    typical_lag_days: 5
    rule: "transaction_date + min_lag_days"
    
  house_ptr:
    min_lag_days: 45
    typical_lag_days: 60
    rule: "transaction_date + min_lag_days"
    
  earnings_release:
    min_lag_days: 0
    typical_lag_days: 0
    rule: "announcement_date"
```

#### 2. **Lag Enforcement in Latency Pipeline** (latency.py)
```python
def validate_and_apply_disclosure_lag(
    events: pd.DataFrame,
    source_id: str,
    lag_policy: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ensure disclosure_date respects lag_policy[source_id].
    Return: (validated_events, [warnings if actual_lag < min_lag])
    """
    source_cfg = lag_policy.get(source_id)
    if not source_cfg:
        raise ValueError(f"Source {source_id} not in lag_policy")
    
    min_lag = source_cfg["min_lag_days"]
    rule = source_cfg["rule"]
    
    # Extract transaction_date from event (required)
    if "transaction_date" not in events.columns:
        raise ValueError(f"{source_id}: transaction_date required for lag validation")
    
    events["disclosure_date"] = (
        pd.to_datetime(events["transaction_date"]) + 
        pd.Timedelta(days=min_lag)
    )
    
    return events, []
```

#### 3. **PIT Gating in Features** (congress_features.py)
```python
def add_congress_features(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    lag_policy: Dict[str, Any],  # NEW
) -> pd.DataFrame:
    """
    Validate disclosure_date >= lag_policy minimum.
    Emit warning if event was not yet publicly known at as_of.
    """
    source_id = events.get("source_id", "house_ptr")
    events = validate_and_apply_disclosure_lag(
        events, source_id, lag_policy
    )
    
    # Only use events that were known at as_of
    known_events = filter_events_as_of(
        events, as_of, disclosure_col="disclosure_date"
    )
    
    if len(known_events) < len(events):
        logger.warning(
            f"[MNPI Gate] {len(events) - len(known_events)} events "
            f"excluded (not yet known at {as_of})"
        )
    
    # Continue with known_events only
    ...
```

#### 4. **Audit Log** (output/audit/disclosure_audit.jsonl)
```json
{
  "timestamp": "2026-05-09T10:30:00Z",
  "backtest_id": "bt_2026q1_ai_tech",
  "feature": "congress_total_amount_90d",
  "as_of": "2026-01-15",
  "source_id": "form4",
  "events_included": 12,
  "events_excluded_not_yet_known": 3,
  "min_lag_applied_days": 2,
  "audit_status": "PASS"
}
```

#### 5. **Backtest Startup Check**
```python
def check_disclosure_lag_compliance(
    backtest_cfg: Dict[str, Any],
    lag_policy: Dict[str, Any],
) -> bool:
    """Validate lag_policy and feature modules before backtest run."""
    for source_id in backtest_cfg.get("disclosure_sources", []):
        if source_id not in lag_policy:
            raise ValueError(f"Source {source_id} missing from lag_policy")
    return True
```

#### 6. **Documentation** (DISCLOSURE_COMPLIANCE.md)
- List each source, its lag, and how it's enforced.
- Example: "Form 4 events use transaction_date + 2 days; filtered via filter_events_as_of()."
- Monthly audit: "all backtests ≥ form4:min_lag_days and ≥ house_ptr:min_lag_days."

### Timeline
- **Week 1:** Add lag_policy.yaml + validate functions.
- **Week 2:** Integrate into congress_features.py and test PIT gating.
- **Week 3:** Run audit log on historical backtests (identify any violations).
- **Week 4:** Document in DISCLOSURE_COMPLIANCE.md; gate all new backtests.

---

## SUMMARY TABLE

| Finding | Severity | Current State | Risk |
|---------|----------|---------------|------|
| Congress trade lag (10d vs. 45d) | **CRITICAL** | Hardcoded; no override | Look-ahead bias; MNPI violation |
| Form 4 timestamp ambiguity | **HIGH** | Metadata only; transaction_date missing | Feature uses unknown lag |
| Insider data references | **HIGH** | Disabled but reachable | Future re-enable could break MNPI |
| No audit trail | **MEDIUM** | No records of lag assumptions | Compliance evidence missing |
| No lag validation | **MEDIUM** | No enforcement mechanism | Silent use of wrong lags possible |

---

## CONCLUSION

The system is **not MNPI-safe in its current state**. The disclosure pipeline fetches public data but **does not enforce the publication lag** required by law. This creates a compliance exposure that would need explicit remediation before live trading.

**Recommended action:** Implement the audit mechanism above. This shifts from "assume it's correct" to "verify it is correct, every time."
