# Feed divergence — yfinance full-history cache vs live Alpaca daily store

Run date (UTC): 2026-06-01  
**Status: DIAGNOSTIC NOTE.** Bounds how far the free yfinance feed used by the full-history robustness study diverges from the live `output/aggregates/daily.parquet` (Alpaca) that the production `sector_rotation_bias` factor actually reads. It also self-diagnoses which adjustment basis the live store uses.  

Symbols: 9 of 9 (XLK, XLF, XLE, XLV, XLI, XLU, XLP, XLY, SPY); compared on common trading days.  
Overlap window: **2018-01-02 → 2026-05-29** (the live store's full sector-ETF coverage; the yfinance cache spans 1998+, so the overlap = the entire live window).  
Live-store basis (self-diagnosed): **total-return (split+dividend adjusted)** — live `close` tracks yfinance **Adj Close**.  
Pooled median divergence — raw-vs-live: **717.3 bps** · adj-vs-live: **0.00 bps** · matched basis: **0.00 bps**.  

## Verdict

On the MATCHING basis the two feeds are effectively identical: pooled median divergence is **0.00 bps** (worst single symbol-day **0.01 bps**, XLK 2018-11-29). The live close matches yfinance **Adj Close**, not raw Close — i.e. `output/aggregates/daily.parquet` `close` is **total-return (split+dividend adjusted)**. Feed-independence is therefore established on the correct (total-return) basis: the yfinance full-history robustness study sits on prices materially identical to the live Alpaca store over the shared window, so its REJECTED verdict carries over as a fair cross-check.  

**Correction (supersedes the price-type wording in the prior docs):** both `docs/results/2026_06_sector_rotation_oos.md` and `docs/results/2026_06_sector_rotation_oos_fullhist.md` describe the live store as "raw close". It is in fact total-return adjusted. This does NOT change any verdict (an adjusted/total-return book still fails to beat SPY on a deflated, significant basis, and the SPY benchmark uses the same basis), and it means the live falsification was already on total-return prices — so the fullhist `adj` mode, not `raw`, is the true live-methodology match.

## Per-symbol divergence (basis points)

_`raw bps` = yfinance raw Close vs live; `adj bps` = yfinance Adj Close vs live; the smaller identifies the live basis. `matched max` = worst single day on the matched basis._

| Symbol | Common bars | Overlap | raw bps (med) | adj bps (med) | matched (med) | matched (max) | max date |
|---|---|---|---|---|---|---|---|
| XLK | 2104 | 2018-01-02→2026-05-29 | 332.4 | 0.00 | 0.00 | 0.01 | 2018-11-29 |
| XLF | 2104 | 2018-01-02→2026-05-29 | 768.9 | 0.00 | 0.00 | 0.01 | 2022-03-07 |
| XLE | 2104 | 2018-01-02→2026-05-29 | 1616.3 | 0.00 | 0.00 | 0.01 | 2018-05-09 |
| XLV | 2104 | 2018-01-02→2026-05-29 | 717.3 | 0.00 | 0.00 | 0.01 | 2018-04-25 |
| XLI | 2104 | 2018-01-02→2026-05-29 | 683.1 | 0.00 | 0.00 | 0.01 | 2018-04-26 |
| XLU | 2113 | 2018-01-02→2026-05-29 | 1373.8 | 0.00 | 0.00 | 0.01 | 2022-06-28 |
| XLP | 2113 | 2018-01-02→2026-05-29 | 1186.1 | 0.00 | 0.00 | 0.01 | 2018-06-04 |
| XLY | 2113 | 2018-01-02→2026-05-29 | 368.6 | 0.00 | 0.00 | 0.01 | 2020-07-22 |
| SPY | 2104 | 2018-01-02→2026-05-29 | 600.6 | 0.00 | 0.00 | 0.01 | 2020-04-13 |

## Caveats (binding)

- **Self-diagnosed basis.** The script does not assume raw or adjusted; it compares the live close to BOTH yfinance series and reports whichever matches. The matched basis came in at ~0 bps, the other at hundreds of bps (the cumulative dividend adjustment), which is what reveals the live store is total-return adjusted.
- **Agreement ≠ identical corporate-action handling on every day.** A ~0 bps median means the feeds agree on the level; per-symbol matched-max + date are listed so any outlier day is inspectable.
- **Verdicts unaffected.** The price-type correction changes wording, not numbers: the live and fullhist REJECTED verdicts stand. Total-return (adjusted) is the more correct backtest basis anyway; only the prior docs' "raw" label was wrong.
- **Read-only.** No production module, live state or network touched.

---
_Script: `scripts/_sector_fullhist_feed_divergence.py` (read-only diagnostic; reuses the cache path + symbol set from `scripts/_oos_wf_sector_rotation_fullhist.py`)._  
_Inputs: `output/research/sector_fullhist_yf.parquet` (yfinance, gitignored) vs `output/aggregates/daily.parquet` (live Alpaca store)._  
_Companion to: `docs/results/2026_06_sector_rotation_oos_fullhist.md` and the live verdict `docs/results/2026_06_sector_rotation_oos.md`._  