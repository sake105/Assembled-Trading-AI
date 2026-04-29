"""Cross-asset carry signal v2 — extended to FX rate differentials, commodity roll, crypto funding."""
from __future__ import annotations

import numpy as np
import pandas as pd


class UniversalCarrySignal:
    """Cross-asset carry signal across FX, bonds, commodities, and crypto.

    All outputs are z-scored signals in [-1, +1].
    """

    def fx_carry(self, rate_differentials: pd.DataFrame) -> pd.DataFrame:
        """Long high-yield currencies, short low-yield.

        Parameters
        ----------
        rate_differentials:
            DataFrame of short-term rate differentials (currency vs USD) per period.
            Columns are currency identifiers (e.g. EUR, JPY, GBP).

        Returns
        -------
        DataFrame of rank-normalized signals in [-1, +1].
        """
        if rate_differentials.empty:
            return pd.DataFrame()
        ranked = rate_differentials.rank(axis=1, pct=True)
        return (ranked - 0.5) * 2

    def commodity_carry(self, futures_curves: pd.DataFrame) -> pd.DataFrame:
        """Long backwardation, short contango (1-month roll-yield signal).

        Parameters
        ----------
        futures_curves:
            DataFrame with MultiIndex columns (date, contract) where contracts
            include "M1" (front) and "M2" (second month). Or simple wide-format
            with columns named "{commodity}_M1" and "{commodity}_M2".

        Returns
        -------
        Rank-normalized carry signal in [-1, +1].
        """
        if futures_curves.empty:
            return pd.DataFrame()

        # Support both MultiIndex and wide-format columns
        try:
            if isinstance(futures_curves.columns, pd.MultiIndex):
                front = futures_curves.xs("M1", level=1, axis=1)
                second = futures_curves.xs("M2", level=1, axis=1)
            else:
                m1_cols = [c for c in futures_curves.columns if str(c).endswith("_M1")]
                m2_cols = [c for c in futures_curves.columns if str(c).endswith("_M2")]
                if not m1_cols:
                    return pd.DataFrame()
                names = [c.replace("_M1", "") for c in m1_cols]
                front = futures_curves[m1_cols].rename(columns=dict(zip(m1_cols, names)))
                second = futures_curves[m2_cols].rename(columns=dict(zip(m2_cols, names)))
        except Exception:
            return pd.DataFrame()

        carry = (front - second) / front.abs().replace(0, np.nan)
        ranked = carry.rank(axis=1, pct=True)
        return (ranked - 0.5) * 2

    def crypto_carry(self, funding_rates: pd.DataFrame) -> pd.DataFrame:
        """Long-spot/short-perp when perpetual funding is positive.

        Parameters
        ----------
        funding_rates:
            8-hourly perpetual funding rates DataFrame (columns = assets).
            Positive = longs pay shorts; negative = shorts pay longs.

        Returns
        -------
        Annualised carry signal clipped to [-1, +1].
        """
        if funding_rates.empty:
            return pd.DataFrame()
        # Annualise: 3 payments/day × 365 = 1095 periods/year
        annualised = funding_rates.rolling(8, min_periods=1).mean() * 1095
        return annualised.clip(-1, 1)

    def bond_carry(self, yield_curves: pd.DataFrame) -> pd.DataFrame:
        """Long steep (high carry) bonds, short flat.

        Parameters
        ----------
        yield_curves:
            DataFrame with columns including short-rate and long-rate per market.
            Expects columns ending in "_short" and "_long".

        Returns
        -------
        Rank-normalized term-premium signal in [-1, +1].
        """
        if yield_curves.empty:
            return pd.DataFrame()
        short_cols = [c for c in yield_curves.columns if str(c).endswith("_short")]
        long_cols = [c for c in yield_curves.columns if str(c).endswith("_long")]
        if not short_cols or not long_cols:
            return pd.DataFrame()
        names = [c.replace("_short", "") for c in short_cols]
        short_df = yield_curves[short_cols].rename(columns=dict(zip(short_cols, names)))
        long_df = yield_curves[long_cols].rename(columns=dict(zip(long_cols, names)))
        spread = long_df - short_df
        ranked = spread.rank(axis=1, pct=True)
        return (ranked - 0.5) * 2

    def combine(
        self,
        fx: pd.DataFrame | None = None,
        commodity: pd.DataFrame | None = None,
        crypto: pd.DataFrame | None = None,
        bond: pd.DataFrame | None = None,
        weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Equal-weight (or custom-weight) combination of available carry signals.

        Returns a single composite carry DataFrame aligned to common dates.
        """
        components: dict[str, pd.DataFrame] = {}
        if fx is not None and not fx.empty:
            components["fx"] = fx
        if commodity is not None and not commodity.empty:
            components["commodity"] = commodity
        if crypto is not None and not crypto.empty:
            components["crypto"] = crypto
        if bond is not None and not bond.empty:
            components["bond"] = bond

        if not components:
            return pd.DataFrame()

        w = weights or {k: 1.0 / len(components) for k in components}
        frames: list[pd.DataFrame] = []
        for name, df in components.items():
            wt = w.get(name, 1.0 / len(components))
            frames.append(df * wt)

        if len(frames) == 1:
            return frames[0]
        combined = pd.concat(frames, axis=1)
        return combined.T.groupby(level=0).mean().T
