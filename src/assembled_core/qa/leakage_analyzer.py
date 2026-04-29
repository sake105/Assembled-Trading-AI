"""Look-ahead bias and recursive (feature leakage) bias detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LeakageReport:
    feature: str
    leakage_type: str  # "lookahead" | "recursive"
    evidence: str
    severity: str  # "high" | "medium" | "low"
    details: dict[str, Any] = field(default_factory=dict)


class LeakageAnalyzer:
    """Detect common forms of data leakage in financial feature pipelines.

    Checks:
    - Look-ahead bias: feature at time t uses information from t+k (k>0)
    - Recursive bias: target variable appears in its own feature set
    - Normalization leakage: scaler fit on full dataset before train/test split
    """

    def __init__(self, max_lag_check: int = 5, correlation_threshold: float = 0.95) -> None:
        self.max_lag_check = max_lag_check
        self.correlation_threshold = correlation_threshold

    def check_lookahead(
        self, features: pd.DataFrame, target: pd.Series
    ) -> list[LeakageReport]:
        """Flag features with suspiciously high correlation to future target values."""
        reports: list[LeakageReport] = []
        target_aligned = target.reindex(features.index)

        for col in features.columns:
            feat = features[col].dropna()
            for lag in range(1, self.max_lag_check + 1):
                shifted_target = target_aligned.shift(-lag)
                common = feat.index.intersection(shifted_target.dropna().index)
                if len(common) < 20:
                    continue
                corr = abs(feat.loc[common].corr(shifted_target.loc[common]))
                if np.isnan(corr):
                    continue
                if corr >= self.correlation_threshold:
                    reports.append(
                        LeakageReport(
                            feature=col,
                            leakage_type="lookahead",
                            evidence=f"|corr(feat_t, target_t+{lag})| = {corr:.3f} >= {self.correlation_threshold}",
                            severity="high" if corr >= 0.99 else "medium",
                            details={"lag": lag, "correlation": float(corr)},
                        )
                    )
                    break  # report first offending lag only

        return reports

    def check_recursive(
        self, features: pd.DataFrame, target: pd.Series
    ) -> list[LeakageReport]:
        """Flag features that are derived directly from the target at the same timestamp."""
        reports: list[LeakageReport] = []
        target_aligned = target.reindex(features.index)

        for col in features.columns:
            feat = features[col].dropna()
            common = feat.index.intersection(target_aligned.dropna().index)
            if len(common) < 20:
                continue
            corr = abs(feat.loc[common].corr(target_aligned.loc[common]))
            if np.isnan(corr):
                continue
            if corr >= self.correlation_threshold:
                reports.append(
                    LeakageReport(
                        feature=col,
                        leakage_type="recursive",
                        evidence=f"|corr(feat_t, target_t)| = {corr:.3f} >= {self.correlation_threshold}",
                        severity="high",
                        details={"correlation": float(corr)},
                    )
                )

        return reports

    def check_normalization_leakage(
        self,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        fitted_on: str = "full",
    ) -> list[LeakageReport]:
        """Warn if test set statistics suggest scaler was fit on full dataset.

        Heuristic: if test-set mean is within fitted training distribution by more than
        3 standard deviations, it might indicate the scaler saw test data.
        This is a soft check — not definitive.
        """
        if fitted_on != "full":
            return []
        reports: list[LeakageReport] = []
        for col in train_features.columns:
            if col not in test_features.columns:
                continue
            train_mean = train_features[col].mean()
            train_std = train_features[col].std()
            test_mean = test_features[col].mean()
            if train_std == 0:
                continue
            z = abs((test_mean - train_mean) / train_std)
            if z < 0.1:
                reports.append(
                    LeakageReport(
                        feature=col,
                        leakage_type="normalization",
                        evidence=f"Test mean ≈ train mean (z={z:.3f}); possible full-dataset scaler fit",
                        severity="low",
                        details={"z_score": float(z), "train_mean": float(train_mean),
                                 "test_mean": float(test_mean)},
                    )
                )
        return reports

    def check_primary_meta_split(
        self,
        primary_train_index: pd.Index,
        primary_val_index: pd.Index,
        meta_train_index: pd.Index,
    ) -> list[LeakageReport]:
        """Detect Primary↔Meta train/val split contamination (A6 critical check).

        In stacked/ensemble pipelines the meta-model must be trained ONLY on
        out-of-sample predictions from the primary model. If meta_train_index
        overlaps primary_train_index, the primary's in-sample (memorised)
        predictions leak into meta training → meta model is overfit on noise.

        Correct workflow:
            primary trains on [t0, t1)
            primary generates OOS predictions on [t1, t2)   ← meta trains here
        Leaked workflow:
            primary trains on [t0, t1)
            meta trains on [t0, t1) using IS primary predictions ← contaminated
        """
        overlap = len(primary_train_index.intersection(meta_train_index))
        total_meta = len(meta_train_index)
        if overlap == 0:
            return []

        contamination_pct = overlap / total_meta * 100.0
        return [
            LeakageReport(
                feature="meta_model_training_set",
                leakage_type="primary_meta_split",
                evidence=(
                    f"{overlap}/{total_meta} ({contamination_pct:.1f}%) meta-train rows overlap "
                    "with primary-train set — meta model trained on in-sample primary predictions"
                ),
                severity="high" if contamination_pct > 50.0 else "medium",
                details={
                    "primary_train_size": len(primary_train_index),
                    "primary_val_size": len(primary_val_index),
                    "meta_train_size": total_meta,
                    "overlap_size": overlap,
                    "contamination_pct": round(contamination_pct, 2),
                },
            )
        ]

    def full_check(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        train_features: pd.DataFrame | None = None,
        test_features: pd.DataFrame | None = None,
        primary_train_index: pd.Index | None = None,
        primary_val_index: pd.Index | None = None,
        meta_train_index: pd.Index | None = None,
    ) -> list[LeakageReport]:
        """Run all available checks and return combined report."""
        reports = self.check_lookahead(features, target)
        reports += self.check_recursive(features, target)
        if train_features is not None and test_features is not None:
            reports += self.check_normalization_leakage(train_features, test_features)
        if (
            primary_train_index is not None
            and primary_val_index is not None
            and meta_train_index is not None
        ):
            reports += self.check_primary_meta_split(
                primary_train_index, primary_val_index, meta_train_index
            )
        return reports

    @staticmethod
    def summarize(reports: list[LeakageReport]) -> dict[str, Any]:
        return {
            "total": len(reports),
            "high": sum(1 for r in reports if r.severity == "high"),
            "medium": sum(1 for r in reports if r.severity == "medium"),
            "low": sum(1 for r in reports if r.severity == "low"),
            "features_flagged": list({r.feature for r in reports}),
        }
