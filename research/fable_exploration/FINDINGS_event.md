# Intraday / Event problem space — findings (round 5, 2026-06-13)

The closure's §5.5 parked idea ("Hormuz → oil-long within hours", event-driven directional).
Tackled honestly: first established what's testable, then tested it.

## Data wall — true intraday is not testable for free
Intraday price store: `output/aggregates/5min.parquet` = 2 symbols / 3 days;
`base_1min` = 2 symbols / 1 month; `assembled_intraday` = 3 symbols. **No usable
intraday history.** Multi-year intraday is paid (Polygon/databento); free (yfinance)
is ~60d 5min / ~7d 1min — too short for a robust backtest. The `news_alpha` event
stream is only ~5 months (2025-12→2026-05), 440 events. So an hours-timescale
event backtest cannot be run.

## What IS testable — the directional thesis at daily/monthly resolution
Used the FREE daily GPR geopolitical-risk index (Caldara-Iacoviello, 1985-2026, 15,134
obs — the canonical "Hormuz" signal) as the trigger and the production
`events/news_alpha/asset_router.py` mapping as the direction (geo-risk → long
defense LMT/NOC/RTX/ITA + gold GLD; supply shock → long energy XLE/XOM/USO). Tested
whether a GPR spike predicts those sectors' outperformance vs SPY. `gpr_event_study.py`.

### Result — NULL (and wrong-signed at monthly)
Monthly, high-GPR-spike month → NEXT-month excess vs SPY (n=496 months, 28 spikes):
| sector | spike next-mo excess | t | non-spike | diff |
|--------|---------------------|---|-----------|------|
| XLE energy | −1.64% | −1.11 | +0.27% | −1.91% |
| ITA defense | −0.81% | −1.06 | +0.28% | −1.09% |
| LMT | −2.24% | −1.32 | +0.52% | −2.76% |
| GLD gold | +0.88% | +0.79 | −0.04% | +0.92% |

Daily, GPR spike (cross +2σ) → forward 5/10/21d excess: every sector |t| < 1.1 (noise).

## Verdict
The geopolitical-event directional thesis is **priced-in by daily resolution** —
energy/defense even MEAN-REVERT after a GPR spike (wrong sign), gold is weakly positive
but insignificant. No tradeable edge at the resolution we can measure. The effect, if it
survives at all, lives in the first minutes — exactly the slice that needs paid intraday
data, has the highest costs, and is the most arbitraged. **No edge for free here either.**
PEAD and the announcement-premium (the other event families) were already tested dead at
daily resolution. Earnings/macro intraday-reaction: same intraday-data wall.

## UPDATE — first-minutes test executed via paid Polygon intraday (2026-06-13)
Built a working Polygon minute-bar ingester (`data/pull_polygon_intraday.py`, loads .env,
paginates, handles 429; tier = ~2yr history + ~5 req/min). Pulled 107 earnings events
(2024-06→2026-05, 20 mega-cap tech, minute bars incl. extended hours) keyed to precise
XBRL acceptance timestamps. Test (`experiment/first_minutes_test.py` + `_split.py`):
- **No continuation** — entering at the liquid open in the gap direction REVERSES:
  open→close signed-by-gap −0.70% (t=−2.56); corr(gap, intraday drift) = −0.20. Intraday
  post-earnings momentum is dead, like daily PEAD.
- **Real but non-deployable reversal:** gap-UP fades −0.84% (t=−2.10); fade strategy
  +0.70%/trade gross (t=2.56), +0.54% @16bps (t=1.98). BUT (1) the significant side needs
  SHORTING gap-ups — project is long-only/no-leverage; (2) the long-only slice (buy
  gap-downs) is insignificant (t=1.45 gross → t=0.62 net of 30bps); (3) FAILS DSR@40
  (0.41); (4) one 2024-26 window, mega-caps, n=107.
**Verdict: the intraday/event space yields a statistically-real earnings-gap-overreaction
reversal, but it is short-side, capacity-thin, DSR-failing, and uncapturable by the
long-only mandate. No deployable edge — consistent with all 5 directions.**
