# B2 CPCV Leakage Check — trend_baseline Walk-Forward

Run date: 2026-05-29  
Period: 2018-01-01 → 2025-12-31  
Config: train=252d / test=252d / step=252d / min_train=200d / purge=0d  
MA warmup buffer: 90 trading-day bars (≈135 calendar days, conservative upper bound)  
Folds generated: **10**  

> **Purpose:** Verify GO_LIVE_CHECKLIST B2 — no information leakage between
> train and test folds in the `_oos_wf_trend_baseline.py` walk-forward design.
> Folds are built by calling `generate_walk_forward_splits()` — the same
> production function used by `_oos_wf_trend_baseline.py`.
> Does not re-run the backtest. Checks fold boundary geometry only.

---

## 1. Fold Boundary Summary

WalkForwardWindow convention: `train_end` is EXCLUSIVE; `test_start` is INCLUSIVE;
`test_end` is EXCLUSIVE.

| Fold | Train Start | Train End (excl) | Test Start (incl) | Test End (excl) | n_train_days | n_test_days |
|------|------------|-----------------|-------------------|-----------------|-------------|-------------|
| 1 | 2018-01-01 | 2018-09-10 | 2018-09-10 | 2019-05-20 | 252 | 252 |
| 2 | 2018-09-10 | 2019-05-20 | 2019-05-20 | 2020-01-27 | 252 | 252 |
| 3 | 2019-05-20 | 2020-01-27 | 2020-01-27 | 2020-10-05 | 252 | 252 |
| 4 | 2020-01-27 | 2020-10-05 | 2020-10-05 | 2021-06-14 | 252 | 252 |
| 5 | 2020-10-05 | 2021-06-14 | 2021-06-14 | 2022-02-21 | 252 | 252 |
| 6 | 2021-06-14 | 2022-02-21 | 2022-02-21 | 2022-10-31 | 252 | 252 |
| 7 | 2022-02-21 | 2022-10-31 | 2022-10-31 | 2023-07-10 | 252 | 252 |
| 8 | 2022-10-31 | 2023-07-10 | 2023-07-10 | 2024-03-18 | 252 | 252 |
| 9 | 2023-07-10 | 2024-03-18 | 2024-03-18 | 2024-11-25 | 252 | 252 |
| 10 | 2024-03-18 | 2024-11-25 | 2024-11-25 | 2025-08-04 | 252 | 252 |

---

## 2. Leakage Checks

**Check 1 — No Train/Test Overlap: PASS**

**Check 2 — MA Warmup Sourced from Training Period: PASS**
  _Using conservative 135-calendar-day bound for 90 trading-day warmup._

**Check 3 — Cross-Fold Test Window Independence: PASS**
  _Config drift detector: trivially satisfied when STEP_SIZE_DAYS == TEST_WINDOW_DAYS. Would fire if step_size < test_window_days._

**Check 4 — Non-Negative Purge Gap (test_start >= train_end): PASS**
  _Regression guard: fires if generate_walk_forward_splits ever produces a negative purge gap._

---

## 3. Overall Verdict

**PASS**

All 10 production folds pass all four leakage checks.

- **No train/test overlap**: test_start >= train_end in all folds.
- **MA warmup in training**: 90 trading-day warmup (≈135 cal days) fits within 252-day training window in all folds.
- **Cross-fold independence**: test windows are non-overlapping (STEP_SIZE_DAYS == TEST_WINDOW_DAYS).
- **Purge gap non-negative**: no calendar-day regression in fold generator.

Combined with A3 PIT regression tests (signal-level look-ahead verification),
the walk-forward design of `_oos_wf_trend_baseline.py` is leak-free.
**GO_LIVE B2 criterion satisfied.**

---

_Script: `scripts/_cpcv_tb_leakage_check.py`_  
_Production WF: `scripts/_oos_wf_trend_baseline.py`_  
_Fold generator: `src/assembled_core/qa/walk_forward.py::generate_walk_forward_splits`_  
_CPCV module: `src/assembled_core/qa/cpcv_validation.py`_  
_GO_LIVE B2: `docs/GO_LIVE_CHECKLIST.md`_  