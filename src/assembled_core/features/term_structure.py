"""VIX curve and yield-curve shape features."""

from __future__ import annotations

import pandas as pd


class TermStructureFeatures:
    """Extract term-structure features from VIX futures quotes and treasury yields."""

    # Expected columns for VIX input: vix_spot, vix_1m, vix_2m, vix_3m, vix_6m
    # Expected columns for yield input: y_3m, y_2y, y_5y, y_10y, y_30y

    def vix_term_structure(self, vix_quotes: pd.DataFrame) -> pd.DataFrame:
        """Return vix_slope_short, vix_slope_long, vix_contango, vix_curvature, vix_pc1/2/3.

        Parameters
        ----------
        vix_quotes:
            DataFrame with at minimum columns ``vix_spot`` and ``vix_3m``.
            Optional: ``vix_1m``, ``vix_2m``, ``vix_6m``.
        """
        out = pd.DataFrame(index=vix_quotes.index)

        spot = vix_quotes.get("vix_spot", vix_quotes.iloc[:, 0])
        m3 = vix_quotes.get("vix_3m", vix_quotes.iloc[:, -1])
        m1 = vix_quotes.get("vix_1m", spot)
        m2 = vix_quotes.get("vix_2m", (spot + m3) / 2)

        out["vix_slope_short"] = m1 - spot
        out["vix_slope_long"] = m3 - spot
        out["vix_contango"] = (m3 - spot).gt(0).astype(int)
        out["vix_curvature"] = m2 - (spot + m3) / 2

        if out.shape[0] >= 3:
            try:
                from sklearn.decomposition import PCA  # noqa: PLC0415

                cols = [
                    c
                    for c in ["vix_spot", "vix_1m", "vix_2m", "vix_3m", "vix_6m"]
                    if c in vix_quotes.columns
                ]
                if len(cols) >= 3:
                    X = vix_quotes[cols].dropna()
                    pca = PCA(n_components=min(3, len(cols)))
                    pca.fit(X)
                    components = pca.transform(X)
                    for i in range(components.shape[1]):
                        pc = pd.Series(
                            components[:, i], index=X.index, name=f"vix_pc{i + 1}"
                        )
                        out[f"vix_pc{i + 1}"] = pc
            except ImportError:
                pass

        return out

    def yield_curve_features(self, treasury_yields: pd.DataFrame) -> pd.DataFrame:
        """Return yc_2y10y, yc_3m10y, yc_inverted.

        Parameters
        ----------
        treasury_yields:
            DataFrame with columns ``y_2y``, ``y_3m``, ``y_10y`` (at minimum).
        """
        out = pd.DataFrame(index=treasury_yields.index)

        y2 = treasury_yields.get("y_2y")
        y3m = treasury_yields.get("y_3m")
        y10 = treasury_yields.get("y_10y")

        if y2 is not None and y10 is not None:
            out["yc_2y10y"] = y10 - y2
        if y3m is not None and y10 is not None:
            out["yc_3m10y"] = y10 - y3m

        if "yc_2y10y" in out.columns and "yc_3m10y" in out.columns:
            out["yc_inverted"] = ((out["yc_2y10y"] < 0) | (out["yc_3m10y"] < 0)).astype(
                int
            )
        elif "yc_2y10y" in out.columns:
            out["yc_inverted"] = (out["yc_2y10y"] < 0).astype(int)
        elif "yc_3m10y" in out.columns:
            out["yc_inverted"] = (out["yc_3m10y"] < 0).astype(int)

        return out

    def combined(
        self,
        vix_quotes: pd.DataFrame | None = None,
        treasury_yields: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        if vix_quotes is not None and not vix_quotes.empty:
            parts.append(self.vix_term_structure(vix_quotes))
        if treasury_yields is not None and not treasury_yields.empty:
            parts.append(self.yield_curve_features(treasury_yields))
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, axis=1)
