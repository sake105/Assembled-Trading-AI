"""Volume-Synchronized PIN (VPIN) — toxic order-flow detector."""

from __future__ import annotations

import numpy as np
import pandas as pd


class VPINCalculator:
    """Estimate VPIN via volume-bucket imbalance rolling mean.

    Flash-Crash predictor: VPIN > 0.7 historically precedes dislocations.
    """

    def __init__(self, n_buckets: int = 50, bucket_size_pct_adv: float = 0.01) -> None:
        self.n_buckets = n_buckets
        self.bucket_size_pct_adv = bucket_size_pct_adv

    def compute(self, trades: pd.DataFrame, avg_daily_volume: float) -> pd.Series:
        """Compute VPIN time series.

        Parameters
        ----------
        trades:
            DataFrame with columns ``volume``, ``buy_volume``, ``sell_volume`` indexed by time.
            If ``buy_volume``/``sell_volume`` absent, bulk-classification via tick-rule is applied.
        avg_daily_volume:
            Average daily volume used to set bucket size.

        Returns
        -------
        pd.Series of VPIN values indexed same as ``trades``.
        """
        if trades.empty:
            return pd.Series(dtype=float, name="vpin")

        df = trades.copy()
        bucket_size = max(1, int(avg_daily_volume * self.bucket_size_pct_adv))

        if "buy_volume" not in df.columns or "sell_volume" not in df.columns:
            df = self._tick_classify(df)

        imbalances = self._bucket_imbalances(df, bucket_size)
        if len(imbalances) < self.n_buckets:
            return pd.Series(np.full(len(df), np.nan), index=df.index, name="vpin")

        rolling_vpin = (
            pd.Series(imbalances)
            .rolling(self.n_buckets, min_periods=self.n_buckets)
            .mean()
        )
        # Reindex back to original trade timestamps (last trade per bucket)
        bucket_times = self._bucket_end_times(df, bucket_size)
        vpin_series = pd.Series(rolling_vpin.values, index=bucket_times, name="vpin")
        return vpin_series.reindex(df.index).ffill()

    # ------------------------------------------------------------------
    def _tick_classify(self, df: pd.DataFrame) -> pd.DataFrame:
        if "price" not in df.columns:
            df["buy_volume"] = df["volume"] * 0.5
            df["sell_volume"] = df["volume"] * 0.5
            return df
        price_diff = df["price"].diff().fillna(0)
        buy_flag = (price_diff >= 0).astype(float)
        df = df.copy()
        df["buy_volume"] = df["volume"] * buy_flag
        df["sell_volume"] = df["volume"] * (1 - buy_flag)
        return df

    def _bucket_imbalances(self, df: pd.DataFrame, bucket_size: int) -> list[float]:
        cumvol = df["volume"].cumsum().values
        vol = df["volume"].values
        buy = df["buy_volume"].values
        sell = df["sell_volume"].values
        imbalances: list[float] = []
        bucket_idx = 1
        b_buy = b_sell = 0.0
        for i in range(len(df)):
            remaining = bucket_size * bucket_idx - (cumvol[i - 1] if i > 0 else 0)
            fill = min(vol[i], max(0.0, remaining))
            ratio = fill / max(vol[i], 1e-9)
            b_buy += buy[i] * ratio
            b_sell += sell[i] * ratio
            if cumvol[i] >= bucket_size * bucket_idx:
                total = b_buy + b_sell
                imbalances.append(abs(b_buy - b_sell) / max(total, 1e-9))
                b_buy = b_sell = 0.0
                bucket_idx += 1
        return imbalances

    def _bucket_end_times(self, df: pd.DataFrame, bucket_size: int) -> list:
        cumvol = df["volume"].cumsum().values
        times: list = []
        bucket_idx = 1
        for i in range(len(df)):
            if cumvol[i] >= bucket_size * bucket_idx:
                times.append(df.index[i])
                bucket_idx += 1
        return times

    @staticmethod
    def threshold() -> float:
        """VPIN > 0.7 historically signals elevated toxic flow."""
        return 0.7
