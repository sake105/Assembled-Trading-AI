"""LiveDecisionEngine — inkrementelle State-Updates für Master-Allocator.

Latenz-Ziel
-----------
``decide_next()`` < 10 ms (typisch 1-5 ms auf modernen CPUs).

Konzept
-------
Statt full-rebuild der Master-Pipeline bei jedem Daily-Update:
1. **State** speichert pre-computed Rolling-Statistiken (cumsum für Mom,
   EMA-Werte, Realized-Vol-Trail, Monthly-Rebalance-Cur-Weights).
2. **update_with_new_day()**: aktualisiert State mit neuem Tagesreturn —
   O(N) statt O(N × T).
3. **decide_next()**: berechnet aus State direkt die heutige Allokation —
   pure numpy, kein groupby/pandas-rolling.

Lifecycle
---------
1. ``LiveDecisionEngine.bootstrap_from_history(prices_panel, xa_panel)``
   einmaliger Aufbau aus Historical-Data.
2. Persist State: ``engine.save_state(path)``.
3. Live-Loop: jede neue Tages-Bar → ``engine.update_with_new_day(new_row)``
   + ``engine.decide_next()`` → Order-Liste.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class LiveEngineConfig:
    sa_weight: float = 0.70
    sa_target_vol_annual: float = 0.12
    sa_vol_window: int = 60
    sa_max_leverage: float = 2.0
    sa_smoothing_window: int = 5
    xa_target_vol_annual: float = 0.10
    xa_vol_window: int = 60
    xa_max_leverage: float = 2.0
    xa_mom_lookback: int = 252
    xa_mom_skip: int = 21
    xa_mom_top_n: int = 5
    xa_hybrid_weight: float = 0.50
    eq_mom_lookback: int = 252
    eq_mom_skip: int = 21
    eq_quantile_long: float = 0.30
    # --- Geo-Stress-Overlay (optional) ---
    enable_geo_overlay: bool = False
    geo_min_multiplier: float = 0.30
    geo_max_multiplier: float = 1.10
    # --- News-Impact-Tilt (optional) ---
    enable_news_tilt: bool = False
    news_tilt_strength: float = 0.30


@dataclass
class EngineState:
    """In-Memory-State für inkrementelle Updates."""

    # Equity (für Mom-12/1-Faktor)
    eq_log_return_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Wide-format log-returns (date × symbol), nur die letzten max_history Tage."""

    eq_factor_returns: list[float] = field(default_factory=list)
    """Historische Faktor-Returns für Vol-Targeting."""

    # Cross-Asset
    xa_log_return_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    xa_ew_returns: list[float] = field(default_factory=list)
    xa_mom_top_weights: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )
    days_since_xa_rebalance: int = 0

    max_history: int = 504  # 2 years buffer
    last_date: pd.Timestamp | None = None

    # Optional overlays — None when disabled, populated via attach_*() methods.
    geo_daily_overlay: pd.DataFrame | None = None
    """Optional [date × {multiplier, state, composite_z}] frame. Set via attach_geo_overlay()."""
    current_geo_multiplier: float = 1.0

    news_tilt_scores: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    """Per-symbol z-score tilt added to mom rank when news-tilt enabled."""


def _compute_realized_vol_annual(returns: list[float], window: int) -> float:
    """Realized annualisierte Vol aus Rolling-Window."""
    if len(returns) < min(10, window // 4):
        return np.nan
    w = returns[-window:]
    return float(np.std(w, ddof=1) * np.sqrt(252))


def _vol_target_leverage(
    returns: list[float], target_vol: float, window: int, max_lev: float, smoothing: int
) -> float:
    """Single-leverage-Wert für aktuellen Tag."""
    rv = _compute_realized_vol_annual(returns, window)
    if not np.isfinite(rv) or rv == 0:
        return 1.0
    lev = target_vol / rv
    # Apply smoothing: simple-moving-avg der letzten N-Werte (hier vereinfacht: clip)
    return float(np.clip(lev, 0.0, max_lev))


class LiveDecisionEngine:
    """High-frequency-friendly Master-Allocator mit O(1) State-Updates."""

    def __init__(self, config: LiveEngineConfig | None = None):
        self.config = config or LiveEngineConfig()
        self.state = EngineState()

    # ====================================================================
    # Optional overlay attachments
    # ====================================================================
    def attach_geo_overlay(self, geo_daily: pd.DataFrame) -> None:
        """Attach pre-computed geo-stress overlay (e.g. from geo_stress_composite).

        Args:
            geo_daily: DataFrame indexed by UTC daily timestamps with at least
                a ``multiplier`` column in [0, 1.1]. Lookup happens in
                update_with_new_day; missing days fall back to 1.0.
        """
        if "multiplier" not in geo_daily.columns:
            raise ValueError("geo_daily must contain 'multiplier' column")
        self.state.geo_daily_overlay = geo_daily.copy()

    def attach_news_tilt_scores(self, scores: pd.Series) -> None:
        """Attach per-symbol news-tilt z-scores (used in eq-top-N selection)."""
        self.state.news_tilt_scores = scores.copy()

    # ====================================================================
    # Bootstrap aus Historical Data
    # ====================================================================
    def bootstrap_from_history(
        self,
        eq_returns_wide: pd.DataFrame,
        xa_returns_wide: pd.DataFrame,
        warmup_days: int | None = None,
    ) -> None:
        """Einmaliger Aufbau aus Historical-Returns.

        Args:
            eq_returns_wide: Equity-Returns (date × symbol).
            xa_returns_wide: Cross-Asset-Returns (date × symbol).
            warmup_days: wieviele Tage aus History behalten (default = max_history).
        """
        st = self.state
        keep = warmup_days or st.max_history

        st.eq_log_return_history = np.log1p(eq_returns_wide.fillna(0)).tail(keep).copy()
        st.xa_log_return_history = np.log1p(xa_returns_wide.fillna(0)).tail(keep).copy()

        # Bootstrap eq_factor_returns: replay last `keep` days
        eq_factor = self._compute_eq_factor_returns_from_history(
            eq_returns_wide.tail(keep)
        )
        st.eq_factor_returns = eq_factor.dropna().tolist()

        # Bootstrap xa_ew_returns
        xa_ew = xa_returns_wide.mean(axis=1).tail(keep).fillna(0).tolist()
        st.xa_ew_returns = xa_ew

        # Bootstrap xa_mom-Top-Weights: compute mom on history, pick top-N at last date
        st.xa_mom_top_weights = self._compute_mom_top_n(
            st.xa_log_return_history, len(st.xa_log_return_history) - 1
        )
        st.days_since_xa_rebalance = 0

        st.last_date = eq_returns_wide.index[-1] if not eq_returns_wide.empty else None

    def _compute_eq_factor_returns_from_history(
        self, eq_returns_wide: pd.DataFrame
    ) -> pd.Series:
        """Re-compute Equity-Factor-Returns für Bootstrap (volle History)."""
        cfg = self.config
        log_r = np.log1p(eq_returns_wide.fillna(0))
        cumsum = log_r.cumsum()
        # Mom-12/1: from (t-lookback) to (t-skip)
        mom = (
            np.exp(cumsum.shift(cfg.eq_mom_skip) - cumsum.shift(cfg.eq_mom_lookback))
            - 1.0
        )
        # Mark insufficient history as NaN
        insufficient = np.arange(len(eq_returns_wide)) < cfg.eq_mom_lookback
        if insufficient.any():
            mom.iloc[insufficient] = np.nan

        # Cross-section rank per row: select top quantile, equal-weight
        T, N = mom.shape
        positions = np.zeros((T, N), dtype=np.float64)
        mom_arr = mom.to_numpy()
        ret_arr = eq_returns_wide.fillna(0).to_numpy()
        for t in range(T):
            row = mom_arr[t]
            valid = np.isfinite(row)
            nv = valid.sum()
            if nv == 0:
                continue
            vals = row[valid]
            order = np.argsort(np.argsort(vals)) + 1
            ranks = order / nv
            top_mask = ranks >= 1 - cfg.eq_quantile_long
            n_top = int(top_mask.sum())
            if n_top == 0:
                continue
            valid_idx = np.where(valid)[0]
            positions[t, valid_idx[top_mask]] = 1.0 / n_top
        # t-1 lag: today's position from yesterday's mom
        positions_lagged = np.vstack([np.zeros((1, N)), positions[:-1]])
        pnl = (positions_lagged * ret_arr).sum(axis=1)
        return pd.Series(pnl, index=eq_returns_wide.index)

    def _compute_mom_top_n(
        self, log_returns_wide: pd.DataFrame, end_idx: int
    ) -> pd.Series:
        """Berechne aktuelle Top-N-Mom-Weights aus log-returns-history."""
        cfg = self.config
        if end_idx < cfg.xa_mom_lookback:
            return pd.Series(0.0, index=log_returns_wide.columns)
        cumsum = log_returns_wide.iloc[: end_idx + 1].cumsum()
        if len(cumsum) < cfg.xa_mom_lookback + 1:
            return pd.Series(0.0, index=log_returns_wide.columns)
        mom_row = (
            np.exp(
                cumsum.iloc[-cfg.xa_mom_skip - 1]
                - cumsum.iloc[-cfg.xa_mom_lookback - 1]
            )
            - 1.0
        )
        valid = mom_row.dropna()
        if len(valid) < cfg.xa_mom_top_n:
            return pd.Series(0.0, index=log_returns_wide.columns)
        top = valid.nlargest(cfg.xa_mom_top_n).index
        weights = pd.Series(0.0, index=log_returns_wide.columns)
        weights[top] = 1.0 / cfg.xa_mom_top_n
        return weights

    # ====================================================================
    # Inkrementelle Updates
    # ====================================================================
    def update_with_new_day(
        self,
        date: pd.Timestamp,
        eq_returns: pd.Series,
        xa_returns: pd.Series,
        eq_factor_return: float | None = None,
    ) -> None:
        """O(1) state-update mit neuen Tagesreturns.

        Args:
            date: Datum des neuen Tages.
            eq_returns: Returns je Equity-Symbol.
            xa_returns: Returns je Cross-Asset-Symbol.
            eq_factor_return: Optional pre-computed Equity-Faktor-Return.
        """
        cfg = self.config
        st = self.state

        # WICHTIG: Reihenfolge ist relevant für PIT-Korrektheit.
        # Compute factor-return MIT YESTERDAY's history (vor dem Append), dann appende.
        # Das stellt sicher dass Top-Picks aus Daten bis T-1 stammen, nicht bis T.

        # Step 1: Compute eq_factor_return BEFORE appending today's returns
        if eq_factor_return is None:
            # _compute_today_eq_factor_return nutzt aktuelle history (= bis T-1)
            # und today's eq_returns für apply.
            eq_factor_return = self._compute_today_eq_factor_return(eq_returns)

        # Step 2: NOW append eq log returns (history now contains today)
        new_eq_log = np.log1p(eq_returns.fillna(0))
        new_eq_log.name = date
        st.eq_log_return_history = pd.concat(
            [st.eq_log_return_history, new_eq_log.to_frame().T]
        ).tail(st.max_history)

        # Step 3: Append xa log returns (after compute)
        new_xa_log = np.log1p(xa_returns.fillna(0))
        new_xa_log.name = date
        st.xa_log_return_history = pd.concat(
            [st.xa_log_return_history, new_xa_log.to_frame().T]
        ).tail(st.max_history)
        st.eq_factor_returns.append(eq_factor_return)
        if len(st.eq_factor_returns) > st.max_history:
            st.eq_factor_returns = st.eq_factor_returns[-st.max_history :]

        # Append xa_ew return
        xa_ew_today = float(xa_returns.fillna(0).mean())
        st.xa_ew_returns.append(xa_ew_today)
        if len(st.xa_ew_returns) > st.max_history:
            st.xa_ew_returns = st.xa_ew_returns[-st.max_history :]

        # Geo-overlay lookup (PIT: use today's multiplier, computed from up-to-T-1 data)
        if cfg.enable_geo_overlay and st.geo_daily_overlay is not None:
            try:
                mult = float(
                    st.geo_daily_overlay["multiplier"]
                    .reindex([date], method="ffill")
                    .iloc[0]
                )
            except (KeyError, IndexError, ValueError):
                mult = 1.0
            if not np.isfinite(mult):
                mult = 1.0
            st.current_geo_multiplier = float(
                np.clip(mult, cfg.geo_min_multiplier, cfg.geo_max_multiplier)
            )
        else:
            st.current_geo_multiplier = 1.0

        # Increment rebalance counter
        st.days_since_xa_rebalance += 1
        # Monthly rebalance: each ~21 trading days
        if st.days_since_xa_rebalance >= 21 or (
            isinstance(date, pd.Timestamp) and date.is_month_end
        ):
            st.xa_mom_top_weights = self._compute_mom_top_n(
                st.xa_log_return_history, len(st.xa_log_return_history) - 1
            )
            st.days_since_xa_rebalance = 0

        st.last_date = date

    def _compute_today_eq_factor_return(self, eq_returns: pd.Series) -> float:
        """Compute today's eq-factor return using picks from history-as-of-T-1.

        WICHTIG: Aufgerufen VOR dem Append des heutigen Returns in update_with_new_day.
        Daher ist ``st.eq_log_return_history`` letzter Eintrag = T-1 (gestern).

        Konvention (matches Bootstrap ``_compute_eq_factor_returns_from_history``):
        Mom_t basiert auf cumsum[t-skip] - cumsum[t-lookback], dann t-1-shifted
        für apply-on-day-t. Hier: picks für day-T basieren auf mom_{T-1}, also:

            mom_{T-1} = cumsum[T-1-skip] - cumsum[T-1-lookback]

        Mit history bis T-1 (iloc[-1] = T-1):
            iloc[-1-skip] = T-1-skip
            iloc[-1-lookback] = T-1-lookback
        """
        cfg = self.config
        st = self.state
        if len(st.eq_log_return_history) < cfg.eq_mom_lookback + 1:
            return 0.0
        cumsum = st.eq_log_return_history.cumsum()
        mom_row = (
            np.exp(
                cumsum.iloc[-1 - cfg.eq_mom_skip]
                - cumsum.iloc[-1 - cfg.eq_mom_lookback]
            )
            - 1.0
        )
        valid = mom_row.dropna()
        if len(valid) == 0:
            return 0.0
        n_top = max(1, int(np.ceil(cfg.eq_quantile_long * len(valid))))
        top_syms = valid.nlargest(n_top).index
        # Equal-weight apply to today's eq_returns
        return float(eq_returns.reindex(top_syms).fillna(0).mean())

    # ====================================================================
    # Decision (heutige Allokation)
    # ====================================================================
    def decide_next(self) -> dict:
        """Berechne aktuelle Allokation aus State.

        Returns:
            dict mit:
            - sa_leverage: Vol-Target-Leverage auf Equity-Faktor
            - xa_top_weights: pd.Series mit Cross-Asset-Mom-Top-N Weights
            - xa_voltarget_leverage: Vol-Target-Leverage für XA-EW
            - master_weights: kombinierte Weights (sa_weight × leverage + ...)
            - timestamp: t-Stempel
        """
        t0 = time.perf_counter()
        cfg = self.config
        st = self.state

        # Vol-Target SA
        sa_lev = _vol_target_leverage(
            st.eq_factor_returns,
            cfg.sa_target_vol_annual,
            cfg.sa_vol_window,
            cfg.sa_max_leverage,
            cfg.sa_smoothing_window,
        )

        # Vol-Target XA-EW
        xa_ew_lev = _vol_target_leverage(
            st.xa_ew_returns,
            cfg.xa_target_vol_annual,
            cfg.xa_vol_window,
            cfg.xa_max_leverage,
            cfg.sa_smoothing_window,
        )

        # Geo-overlay applies to BOTH SA and XA leverage (system-wide risk-off)
        geo_mult = st.current_geo_multiplier if cfg.enable_geo_overlay else 1.0
        sa_lev = sa_lev * geo_mult
        xa_ew_lev = xa_ew_lev * geo_mult

        # XA Hybrid: 50% VT-EW + 50% Mom-Top-N
        n_xa = len(st.xa_log_return_history.columns)
        xa_ew_per_symbol = xa_ew_lev / max(n_xa, 1)
        xa_hybrid_per_symbol = pd.Series(
            cfg.xa_hybrid_weight * xa_ew_per_symbol,
            index=st.xa_log_return_history.columns,
        )
        xa_hybrid_per_symbol += (
            1.0 - cfg.xa_hybrid_weight
        ) * st.xa_mom_top_weights.reindex(xa_hybrid_per_symbol.index).fillna(0)

        # Master-Mix
        # Equity-side: sa_weight × leverage applied to equity factor (signal-only here;
        # actual symbol-level weights need yesterday's mom-12/1 top picks)
        eq_top_weights = self._compute_eq_factor_top_weights_today()

        master = {
            "timestamp": st.last_date,
            "sa_leverage": float(sa_lev),
            "xa_ew_leverage": float(xa_ew_lev),
            "xa_top_weights": st.xa_mom_top_weights.copy(),
            "xa_hybrid_weights": xa_hybrid_per_symbol,
            "eq_top_weights": eq_top_weights,
            "sa_weight": cfg.sa_weight,
            "xa_weight": 1.0 - cfg.sa_weight,
            "geo_multiplier": float(st.current_geo_multiplier),
            "decision_latency_ms": (time.perf_counter() - t0) * 1000,
        }
        return master

    def _compute_eq_factor_top_weights_today(self) -> pd.Series:
        """Top-N Mom-12/1-Picks für heute (für tatsächliches Order-Routing).

        Wenn ``cfg.enable_news_tilt`` aktiv und ``state.news_tilt_scores``
        nicht leer ist, wird der Rank-Score durch
        ``z(mom) + news_tilt_strength * news_z`` ersetzt.
        """
        cfg = self.config
        st = self.state
        if len(st.eq_log_return_history) < cfg.eq_mom_lookback:
            return pd.Series(0.0, index=st.eq_log_return_history.columns)
        cumsum = st.eq_log_return_history.cumsum()
        mom_row = (
            np.exp(
                cumsum.iloc[-cfg.eq_mom_skip - 1]
                - cumsum.iloc[-cfg.eq_mom_lookback - 1]
            )
            - 1.0
        )
        valid = mom_row.dropna()
        if len(valid) == 0:
            return pd.Series(0.0, index=st.eq_log_return_history.columns)

        # Optional news-tilt: z-normalize mom, add tilt-strength * news_z
        if cfg.enable_news_tilt and not st.news_tilt_scores.empty:
            mu = valid.mean()
            sd = valid.std(ddof=0)
            mom_z = (valid - mu) / sd if sd > 0 else valid * 0.0
            news_z = st.news_tilt_scores.reindex(valid.index).fillna(0.0)
            combined = mom_z + cfg.news_tilt_strength * news_z
            valid = combined

        n_top = max(1, int(np.ceil(cfg.eq_quantile_long * len(valid))))
        top = valid.nlargest(n_top).index
        weights = pd.Series(0.0, index=st.eq_log_return_history.columns)
        weights[top] = 1.0 / n_top
        return weights

    # ====================================================================
    # State Persistence
    # ====================================================================
    def save_state(self, path: str | Path) -> None:
        """Persist State als pickle (für Live-Loop-Restart)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.state, f)

    def load_state(self, path: str | Path) -> None:
        """Lade State aus pickle."""
        with open(path, "rb") as f:
            self.state = pickle.load(f)

    def state_summary(self) -> dict:
        """Kompaktes State-Summary für Monitoring."""
        st = self.state
        return {
            "last_date": str(st.last_date) if st.last_date else None,
            "n_eq_history_days": len(st.eq_log_return_history),
            "n_xa_history_days": len(st.xa_log_return_history),
            "n_eq_factor_returns": len(st.eq_factor_returns),
            "n_xa_top_weights_nonzero": int((st.xa_mom_top_weights > 0).sum()),
            "days_since_xa_rebalance": st.days_since_xa_rebalance,
        }


__all__ = ["LiveDecisionEngine", "LiveEngineConfig", "EngineState"]
