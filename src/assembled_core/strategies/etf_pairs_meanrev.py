"""ETF-Pairs Cointegration Mean-Reversion strategy.

Trades mean-reversion on cointegrated ETF pairs using a rolling Engle-Granger
cointegration test and Z-score entry/exit thresholds.

Default pairs (all US-listed, daily OHLCV available via Alpaca):
  SPY/IVV   — S&P 500 large-cap duplicates
  GDX/GDXJ  — senior / junior gold miners
  XLE/VDE   — energy sector duplicates (GDXJ inception Nov 2009 limits history)
  EWA/EWC   — Australia / Canada country ETFs
  XLF/KBE   — broad financials / bank ETF
  XLK/VGT   — technology sector duplicates

Signal logic per pair (strictly causal — only bars ≤ current bar used):
  1. Rolling cointegration test (Engle-Granger) on last ``cointegration_window``
     bars of log prices.  If p > 0.05 the pair is inactive; any open position
     is exited immediately (cointegration breakdown).
  2. OLS hedge ratio β from the same window:
       Spread_t = log(A_t) − β × log(B_t)
  3. Z-score normalised over last ``zscore_window`` bars of the spread series
     (within the cointegration window):
       Z_t = (Spread_t − μ_{60}) / σ_{60}
  4. Entry:
       Z < −entry_z → Long spread  (Long A, Short B)
       Z > +entry_z → Short spread (Short A, Long B)
  5. Exit:
       |Z| < exit_z (mean-reversion complete)  OR
       |Z| > stop_z (stop-loss, spread diverged)

Two modes via ``long_only`` flag:
  False (full, default): both legs emitted as LONG + SHORT signals.
  True  (long-only): only the long leg per pair is emitted.

Portfolio allocation: equal-weight across active pair positions.
  In full (long-short) mode each active pair contributes two legs (±1/k weight),
  so gross notional exposure = 2.0 × capital regardless of k.  The net is ~0
  (market-neutral pairs design).  The downstream pipeline must apply a
  gross-exposure cap if 200% gross is unacceptable for the account.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    from statsmodels.tsa.stattools import coint as eg_coint

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False

logger = logging.getLogger(__name__)

_DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("SPY", "IVV"),
    ("GDX", "GDXJ"),
    ("XLE", "VDE"),
    ("EWA", "EWC"),
    ("XLF", "KBE"),
    ("XLK", "VGT"),
]

_DEFAULT_COINT_WINDOW = 252
_DEFAULT_ZSCORE_WINDOW = 60
_DEFAULT_ENTRY_Z = 2.0
_DEFAULT_EXIT_Z = 0.5
_DEFAULT_STOP_Z = 3.5

# State codes (int8)
_FLAT = 0
_LONG_SPREAD = 1  # Long A, Short B
_SHORT_SPREAD = -1  # Short A, Long B

_EMPTY = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
_EMPTY_POS = pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])


def _compute_pair_states(
    la: np.ndarray,
    lb: np.ndarray,
    *,
    coint_window: int,
    zscore_window: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    stop_cooldown_bars: int = 5,
) -> np.ndarray:
    """Bar-by-bar position state for a single ETF pair.

    Args:
        la: log-price array for asset A, length n.
        lb: log-price array for asset B, length n.
        stop_cooldown_bars: bars after a stop-loss during which re-entry is
            blocked.  Default 5.  Prevents immediate re-entry into a diverging
            spread: after a stop at |Z|>stop_z, the spread may still satisfy
            the entry threshold (|Z|>entry_z) on the very next bar.

    Returns:
        int8 array of length n with values _FLAT (0), _LONG_SPREAD (1),
        _SHORT_SPREAD (-1).  Elements 0..coint_window-2 are always _FLAT.
    """
    n = len(la)
    states = np.zeros(n, dtype=np.int8)

    if not _HAS_STATSMODELS:
        logger.error("[etf_pairs] statsmodels unavailable — all states flat")
        return states

    state = _FLAT
    cooldown = 0  # bars remaining before re-entry is allowed after stop-loss

    for t in range(coint_window - 1, n):
        # Decrement first; stop-loss sets cooldown=N after this point,
        # so the stop bar is protected by just_exited=True and the N-bar
        # block starts at t+1.
        if cooldown > 0:
            cooldown -= 1

        la_w = la[t - coint_window + 1 : t + 1]
        lb_w = lb[t - coint_window + 1 : t + 1]

        # Guard: degenerate windows (NaN / inf) → flat
        if not (np.all(np.isfinite(la_w)) and np.all(np.isfinite(lb_w))):
            if state != _FLAT:
                logger.warning(
                    "[etf_pairs] NaN/inf in log-price window at t=%d — forced flat from state=%d",
                    t,
                    state,
                )
            state = _FLAT
            states[t] = _FLAT
            continue

        # Step 1: Engle-Granger cointegration test (AIC lag selection)
        try:
            _, p_value, _ = eg_coint(la_w, lb_w, autolag="AIC")
        except Exception as exc:
            logger.debug("[etf_pairs] coint exception at t=%d: %s", t, exc)
            state = _FLAT
            states[t] = _FLAT
            continue

        if p_value > 0.05:
            # Not cointegrated — exit any open position
            state = _FLAT
            states[t] = _FLAT
            continue

        # Step 2: OLS hedge ratio — la = α + β·lb + ε
        try:
            res = OLS(la_w, add_constant(lb_w)).fit()
        except Exception as exc:
            logger.debug("[etf_pairs] OLS exception at t=%d: %s", t, exc)
            state = _FLAT
            states[t] = _FLAT
            continue

        beta = float(res.params[1])
        # Reject economically implausible hedge ratios — extreme OLS beta can
        # arise from ill-conditioned windows and would invert spread direction.
        if not (0.1 <= abs(beta) <= 10.0):
            logger.debug(
                "[etf_pairs] implausible beta=%.3f at t=%d; skipping bar", beta, t
            )
            state = _FLAT
            states[t] = _FLAT
            continue

        # Step 3: Spread and Z-score
        spread = la_w - beta * lb_w  # shape (coint_window,)
        s_z = spread[-zscore_window:]
        mu = float(s_z.mean())
        sigma = float(np.std(s_z, ddof=1))
        if sigma < 1e-8:
            # Degenerate spread (data quality issue) — exit any open position.
            # No cooldown applied: next bar may re-enter if spread recovers.
            # Sustained degenerate data will keep triggering this guard each bar.
            logger.debug(
                "[etf_pairs] degenerate spread sigma at t=%d; exiting position", t
            )
            state = _FLAT
            states[t] = _FLAT
            continue
        z = (float(spread[-1]) - mu) / sigma

        # Step 4: State machine — exit first, then entry.
        # Invariant: positions never flip directly LONG↔SHORT; must pass through FLAT.
        # (entry block below only fires when state == _FLAT after exit processing)
        just_exited = False
        if state != _FLAT:
            if abs(z) > stop_z:
                # Stop-loss: enforce cooldown to prevent immediate re-entry
                logger.debug(
                    "[etf_pairs] stop-loss fired at t=%d Z=%.2f cooldown=%d",
                    t,
                    z,
                    stop_cooldown_bars,
                )
                state = _FLAT
                just_exited = True
                cooldown = stop_cooldown_bars
            elif abs(z) < exit_z:
                # Normal mean-reversion exit
                state = _FLAT
                just_exited = True

        # No same-bar re-entry; no re-entry during stop-loss cooldown
        if state == _FLAT and not just_exited and cooldown == 0:
            if z < -entry_z:
                state = _LONG_SPREAD
            elif z > entry_z:
                state = _SHORT_SPREAD

        states[t] = state

    return states


def generate_etf_pairs_signals_from_prices(
    prices: pd.DataFrame,
    *,
    pairs: list[tuple[str, str]] | None = None,
    cointegration_window: int = _DEFAULT_COINT_WINDOW,
    zscore_window: int = _DEFAULT_ZSCORE_WINDOW,
    entry_z: float = _DEFAULT_ENTRY_Z,
    exit_z: float = _DEFAULT_EXIT_Z,
    stop_z: float = _DEFAULT_STOP_Z,
    long_only: bool = False,
    stop_cooldown_bars: int = 5,
) -> pd.DataFrame:
    """Generate full time-series signals for all ETF pairs.

    Returns one row per active position leg per bar from the first bar with a
    complete cointegration window onward.

    Column schema: timestamp | symbol | direction ("LONG" / "SHORT") | score (weight).

    Weights are equal across active pairs; in full mode both legs of a pair
    share the same weight (the long leg's weight equals the short leg's weight).
    In long-only mode only the long leg of each active pair is emitted.

    Args:
        prices: long-format DataFrame with columns timestamp, symbol, close.
        pairs: list of (A, B) symbol tuples.  Defaults to the six default pairs.
        cointegration_window: rolling bars for EG test and OLS fit.
        zscore_window: trailing bars within the coint window for Z normalisation.
        entry_z: |Z| threshold for entering a position.
        exit_z: |Z| threshold for exiting (mean-reversion complete).
        stop_z: |Z| threshold for stop-loss exit.
        long_only: if True emit only long legs (no SHORT signals).
        stop_cooldown_bars: bars after a stop-loss during which re-entry is blocked.

    Returns:
        DataFrame[timestamp, symbol, direction, score] sorted by timestamp.
        Empty schema DataFrame if data is insufficient.
    """
    if pairs is None:
        pairs = list(_DEFAULT_PAIRS)

    if zscore_window >= cointegration_window:
        logger.warning(
            "[etf_pairs] zscore_window (%d) >= cointegration_window (%d) — "
            "clamping to coint_window-1",
            zscore_window,
            cointegration_window,
        )
        zscore_window = cointegration_window - 1

    all_syms: set[str] = set()
    for a, b in pairs:
        all_syms.add(a)
        all_syms.add(b)

    present = set(prices["symbol"].unique())
    missing = all_syms - present
    if missing:
        logger.warning(
            "[etf_pairs] missing symbols in prices panel: %s", sorted(missing)
        )
        return _EMPTY.copy()

    subset = prices[prices["symbol"].isin(all_syms)].copy()

    # Warn on duplicate (timestamp, symbol) rows — pivot_table default 'mean'
    # would silently average them; use 'last' to match close-price semantics.
    n_dupes = subset.duplicated(["timestamp", "symbol"]).sum()
    if n_dupes:
        logger.warning(
            "[etf_pairs] %d duplicate (timestamp, symbol) rows in prices — using 'last'",
            n_dupes,
        )

    # Pivot to wide format; forward-fill only (no bfill — would propagate future
    # prices backwards for symbols with late inception dates, introducing look-ahead bias)
    pivot = subset.pivot_table(
        index="timestamp", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    pivot = pivot.ffill()

    dates = pivot.index
    n = len(dates)

    if n < cointegration_window:
        logger.debug(
            "[etf_pairs] insufficient bars: %d < %d required",
            n,
            cointegration_window,
        )
        return _EMPTY.copy()

    # Guard non-positive prices before log (corrupted feed / corporate-action artefact)
    price_arr = pivot.values.astype(np.float64)
    n_nonpos = int(np.sum(price_arr <= 0))
    if n_nonpos:
        logger.warning(
            "[etf_pairs] %d non-positive prices in panel — replacing with NaN", n_nonpos
        )
        price_arr = np.where(price_arr > 0, price_arr, np.nan)

    # log-price matrix: shape (n, n_symbols)
    log_mat = np.log(price_arr)
    col_idx = {sym: i for i, sym in enumerate(pivot.columns)}

    # Run simulation for each valid pair
    pair_states: dict[tuple[str, str], np.ndarray] = {}
    for a, b in pairs:
        if a not in col_idx or b not in col_idx:
            logger.warning("[etf_pairs] pair (%s/%s) not in pivot — skipping", a, b)
            continue
        la = log_mat[:, col_idx[a]]
        lb = log_mat[:, col_idx[b]]
        states = _compute_pair_states(
            la,
            lb,
            coint_window=cointegration_window,
            zscore_window=zscore_window,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            stop_cooldown_bars=stop_cooldown_bars,
        )
        pair_states[(a, b)] = states

    if not pair_states:
        return _EMPTY.copy()

    # Build output rows: equal-weight across active pairs per bar
    rows: list[dict] = []
    for t in range(cointegration_window - 1, n):
        ts = dates[t]

        active: list[tuple[str, str, int]] = []
        for (a, b), arr in pair_states.items():
            st = int(arr[t])
            if st != _FLAT:
                active.append((a, b, st))

        k = len(active)
        if k == 0:
            continue

        weight = 1.0 / k

        for a, b, state in active:
            if state == _LONG_SPREAD:
                rows.append(
                    {"timestamp": ts, "symbol": a, "direction": "LONG", "score": weight}
                )
                if not long_only:
                    rows.append(
                        {
                            "timestamp": ts,
                            "symbol": b,
                            "direction": "SHORT",
                            "score": weight,
                        }
                    )
            else:  # _SHORT_SPREAD
                if not long_only:
                    rows.append(
                        {
                            "timestamp": ts,
                            "symbol": a,
                            "direction": "SHORT",
                            "score": weight,
                        }
                    )
                rows.append(
                    {"timestamp": ts, "symbol": b, "direction": "LONG", "score": weight}
                )

    if not rows:
        logger.debug("[etf_pairs] no active positions generated in signal window")
        return _EMPTY.copy()

    result = pd.DataFrame(rows)
    return result.sort_values("timestamp").reset_index(drop=True)


def compute_signals(
    prices: pd.DataFrame,
    *,
    pairs: list[tuple[str, str]] | None = None,
    cointegration_window: int = _DEFAULT_COINT_WINDOW,
    zscore_window: int = _DEFAULT_ZSCORE_WINDOW,
    entry_z: float = _DEFAULT_ENTRY_Z,
    exit_z: float = _DEFAULT_EXIT_Z,
    stop_z: float = _DEFAULT_STOP_Z,
    long_only: bool = False,
    stop_cooldown_bars: int = 5,
) -> pd.DataFrame:
    """Return latest-bar signals for the paper trading cycle.

    Delegates to generate_etf_pairs_signals_from_prices and returns only the
    rows for the most recent timestamp.
    """
    full = generate_etf_pairs_signals_from_prices(
        prices,
        pairs=pairs,
        cointegration_window=cointegration_window,
        zscore_window=zscore_window,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        long_only=long_only,
        stop_cooldown_bars=stop_cooldown_bars,
    )
    if full is None or full.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    # If the most recent input bar has no active positions, the strategy is FLAT
    # today.  Return empty rather than the most-recent-active-bar (which could be
    # weeks stale and would cause the OMS to replay a historical position).
    prices_latest_ts = prices["timestamp"].max()
    latest_ts = full["timestamp"].max()
    if latest_ts < prices_latest_ts:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    return full[full["timestamp"] == latest_ts].reset_index(drop=True)


def compute_target_positions(
    signals: pd.DataFrame,
    capital: float,
    **_kwargs,
) -> pd.DataFrame:
    """Derive target positions from ETF-pairs signals.

    LONG signals map to positive target_weight; SHORT signals to negative.
    target_qty = 0.0 — the downstream pipeline converts target_weight → shares
    using current prices (same contract as trend_baseline, dual_momentum,
    vol_target_overlay).

    Gross exposure note: in full (long-short) mode both legs of an active pair
    carry weight ±1/k, so gross = 2 × net (200% gross at all times).  Long-only
    mode gross = 100%.  A gross-exposure cap must be enforced upstream if needed.

    Args:
        signals: DataFrame from compute_signals (columns symbol, direction, score).
        capital: current capital base (USD); unused here, included for interface
                 compatibility.

    Returns:
        DataFrame[symbol, target_weight, target_qty].
    """
    _empty = pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    if signals is None or signals.empty:
        return _empty
    required = {"symbol", "direction", "score"}
    if not required.issubset(signals.columns):
        return _empty

    rows = []
    for _, row in signals.iterrows():
        w = float(row["score"]) if row["direction"] == "LONG" else -float(row["score"])
        rows.append({"symbol": row["symbol"], "target_weight": w, "target_qty": 0.0})

    if not rows:
        return _empty
    return pd.DataFrame(rows)
