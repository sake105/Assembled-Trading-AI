"""Brinson-Hood-Beebower performance attribution."""

from __future__ import annotations

import pandas as pd


class BrinsonAttribution:
    """Decompose active return into Allocation, Selection, and Interaction effects."""

    def __init__(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
    ) -> None:
        self.w_p = portfolio_weights
        self.w_b = benchmark_weights

    def attribute(
        self,
        sector_returns_portfolio: pd.DataFrame,
        sector_returns_benchmark: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return DataFrame with allocation/selection/interaction/active_total per period."""
        w_diff = self.w_p - self.w_b
        allocation = (w_diff * sector_returns_benchmark).sum(axis=1)
        selection = (
            self.w_b * (sector_returns_portfolio - sector_returns_benchmark)
        ).sum(axis=1)
        interaction = (
            w_diff * (sector_returns_portfolio - sector_returns_benchmark)
        ).sum(axis=1)
        return pd.DataFrame(
            {
                "allocation": allocation,
                "selection": selection,
                "interaction": interaction,
                "active_total": allocation + selection + interaction,
            }
        )

    def summary(
        self,
        sector_returns_portfolio: pd.DataFrame,
        sector_returns_benchmark: pd.DataFrame,
    ) -> dict[str, float]:
        df = self.attribute(sector_returns_portfolio, sector_returns_benchmark)
        return {k: float(df[k].sum()) for k in df.columns}
