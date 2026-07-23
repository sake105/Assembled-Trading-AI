# Phase 2 — Pre-registered hypotheses (2026-06-13)

Pre-registered BEFORE the mill so the verdict can't be back-fitted. Each: what,
why (economics, not pattern), data, and what would FALSIFY it. Status updated as
the mill runs.

---

## H1 — Insider open-market PURCHASE signal  [PRIMARY → goes to the mill]
- **What:** Long an equal-weight portfolio of names with ≥1 open-market insider
  PURCHASE (Form 4 code `P`, non-derivative, price>0) accepted in the trailing
  ~63 trading days; rebalanced; vs SPY and 60/40, risk-adjusted. Variant: cluster
  (≥2 distinct insiders / 30d).
- **Why (economics):** Voluntary open-market buys are information-rich; insiders
  buy because they expect appreciation. Sells are noisy (liquidity/diversification).
  Among the most replicated insider anomalies: Lakonishok & Lee (2001),
  Jeng-Metrick-Zeckhauser (2003), Cohen-Malloy-Pomorski (2012, opportunistic vs
  routine). Was a DEAD factor (100% "unknown") before the 2026-06 Form 4 reparse.
- **Data:** `form4_insider_full.parquet` (P buys, `available_at` PIT) + TR-adj prices.
- **Raw probe:** SPY-excess +4.28%/60d, t=3.73 (n=540) — BUT overlapping windows
  inflate t and survivorship inflates the level. Portfolio-level mill required.
- **FALSIFIED if:** OOS walk-forward portfolio Sharpe ≤ SPY/60-40 Sharpe; OR not
  DSR-significant at 5% with honest trial count; OR edge gone after realistic
  costs; OR concentrated in <3 names/years; OR killed by the survivorship/
  illiquidity stress test (return haircut, size/vol buckets).
- **STATUS 2026-06-13 → FAILED (real in-sample, not deployable).** Full-universe
  Sharpe 1.134>SPY 0.775, DSR-sig, PBO 0.393, cost-robust — BUT drop-top-3-names →
  0.788≈SPY (concentration); high-liquidity slice 0.563<SPY (investable slice has no
  edge); recent half 0.968≈SPY 0.967 (decayed); MaxDD −30% fails §4.4. See `ERGEBNIS.md`.

## H2 — Large congressional PURCHASES, 60d hold  [SECONDARY → mill only if cheap]
- **What:** Long names with a STOCK-Act disclosed buy >$50k (statutory midpoint),
  60d hold, vs SPY/60-40.
- **Why:** Possible committee/timing information advantage (contested post-STOCK-Act;
  Ziobrowski pre-2012 strong, later studies weak).
- **Data:** `congress_trades_full.parquet` (buys, `disclosure_date`=available_at).
- **Raw probe:** all-buys +0.80%/60d (t=2.69); >$50k +2.19%/60d (t=2.11, n=218).
- **FALSIFIED if:** same criteria. Expected to die on DSR + House-mirror data-quality
  caveat (no SLA, House-dominant). Low prior.

## H3 — PEAD / SUE drift  [PROBED → NULL, does NOT enter the mill]
- **Why hoped:** Most-robust anomaly; claimed large-cap persistence; real PIT EPS now
  available; explicitly parked-not-falsified in the closure (§5.5/§7.2).
- **Result (raw probe):** +5d Spearman IC +0.040 only; +20/60d IC ≈ 0 and the
  top−bottom SUE tercile spread is the WRONG sign (−0.39% / −1.87%); by-year r60 IC
  flips sign nearly every year. The classic 60d drift is absent/inverted on this
  large-cap survivor universe — consistent with PEAD's documented decay and its
  concentration in small/illiquid names (McLean-Pontiff 2016).
- **Verdict:** fails the cheap gate. NOT advanced. Honest null — the strongest
  a-priori candidate did not survive contact with the data.

## H4 — Earnings-announcement PREMIUM (timing tilt)  [round 2]
- **What:** Hold names in the ~5 trading days BEFORE their (predictable-cadence)
  quarterly earnings announcement; exit before the report bar. Equal-weight basket
  vs SPY. NOT the surprise-sorted PEAD drift (H3, dead) — this is the unconditional
  attention/risk premium that accrues pre-announcement.
- **Why:** Frazzini & Lamont (2007), Barber et al. — stocks earn higher returns in
  the window when investors expect an earnings release (attention/uncertainty premium).
  Announcement dates are highly predictable from the ~91d cadence → PIT-safe.
- **Data:** XBRL earliest-acceptance per (sym, quarter) = announcement date; TR-adj prices.
- **FALSIFIED if:** pre-announcement window SPY-excess ≈ 0 or insignificant; OR the
  basket Sharpe ≤ SPY net of (high) turnover costs; OR concentrated/regime-fragile.
- **STATUS 2026-06-13 → DEAD as a strategy.** Pre-window effect real (t=3.88) but
  decaying (t<1.4 since 2024); basket Sharpe 0.74@10bps/0.57@20bps **< SPY 0.775**,
  MaxDD −42%, 3/9 folds, fails DSR@33 (prob 0.31). Turnover eats it.

## H5 — Aggregate insider net-buying as a MARKET-TIMING / defensive overlay  [round 2]
- **What:** Cross-firm trailing net-buy ratio P/(P+S) of open-market insider trades
  (both codes are open-market) → scale SPY exposure (risk-off when insiders net-sell).
  Compared to buy-hold SPY and a realized-vol overlay, judged on Sharpe AND MaxDD.
- **Why:** Seyhun (1988, 1992) — AGGREGATE insider trading predicts AGGREGATE market
  returns (a market-level, not cross-sectional, effect). **Survivorship-robust**: the
  traded instrument is SPY, so the closure's killer biases (concentration, illiquidity,
  delisting) barely apply. Most deployable of anything tried.
- **Data:** all 9,744 P-buys + 311k S-sells (`available_at` PIT), SPY returns.
- **FALSIFIED if:** signal→forward-SPY correlation ≈ 0; OR the timed overlay does not
  improve Sharpe AND/OR MaxDD vs buy-hold SPY out-of-sample; OR it only "works" by
  sitting in cash during the 2020/2022 drawdowns by luck (check signal lead, not lag).
- **STATUS 2026-06-13 → DEAD.** Raw IC +0.137 was a non-stationarity artifact.
  Detrended: Sharpe 0.572 < SPY 0.792; a CONSTANT 41% SPY (0.792/−15% MaxDD) beats the
  insider-TIMED 41%-avg (0.572/−24%) → negative timing value. Hurts a vol-target overlay,
  fails exclude-2020, monthly t=1.65 (insignificant), fails DSR. Strictly dominated by
  the incumbent `vol_target_overlay`.

## H6 — Short-flow level (FINRA RegSHO) long-low-short tilt  [round 3]
- **What:** Daily ShortVolume/TotalVolume ratio (FINRA RegSHO, free, FULL universe
  breadth 195/195). Long-only: hold the bottom-tercile (least-shorted) names; avoid
  the top tercile. Slow-moving level → low turnover. vs SPY/60-40.
- **Why:** Boehmer-Jones-Zhang (2008) informed-shorting + Asquith-Pathak-Ritter
  short-interest anomaly: heavily-shorted names underperform. Survivorship works
  AGAINST finding it (shorted names that went bust are absent → biases the short
  leg UP) → a clean result here is robust.
- **Data:** `research/.../data/short_volume_*.parquet` (RegSHO daily) + TR-adj prices.
- **Raw gate (2023-24): PASS on the LEVEL** — cross-sectional daily IC −0.018@5d
  (t=−3.1), −0.044@20d (t=−7.4), correct sign, full breadth. BUT abnormal/change
  version DEAD (IC≈0) → it's a slow CHARACTERISTIC, not a timing signal.
- **FALSIFIED if:** long-low-shortflow basket Sharpe ≤ equal-weight-universe (no tilt
  value) or ≤ SPY; OR fully explained by low-beta/value (just a re-skin of dead factors);
  OR fails DSR at the cumulative trial count; OR only the (un-tradeable) short leg works.
- **STATUS 2026-06-13 → DEAD.** Long-low tilt Sharpe 1.069 < **EW-universe 1.133** (tilt
  subtracts value); L/S Sharpe −0.07 (significant IC, zero economic spread); fails DSR@34
  (0.834). Not low-beta (corr +0.08). The decisive by-product: EW-universe 1.133 ≈
  insider-base 1.134 → the ~1.13 Sharpe is the SURVIVORSHIP BASELINE, not any signal.

## Interactions (Insider×Fundamentals-drift, Congress×Sector) — deliberately deferred
The prompt invited interaction mining. Holding off ON PURPOSE: interactions multiply
the multiple-testing surface and the data is sparse (insider 540 events, congress
thin). Will only test an interaction if H1 survives the mill standalone — and then a
SINGLE pre-specified interaction (insider-buy × positive-quality), DSR-counted.
