"""News feature validation framework.

Three-level validation pipeline from 34_NEWS_GROUND_TRUTH.md:
  Level A — classification accuracy against gold datasets
  Level B — economic relevance via event-study (MacKinlay 1997)
  Level C — tradability via IC / quantile-spread / cost-adjusted edge
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Level A — Classification Accuracy
# ---------------------------------------------------------------------------


def classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy and F1 metrics for 3-class sentiment classification."""
    if labels is None:
        labels = ["positive", "negative", "neutral"]

    n = len(y_true)
    if n == 0 or len(y_pred) != n:
        raise ValueError("y_true and y_pred must be non-empty and equal length")

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    true_idx = [label_to_idx.get(y, -1) for y in y_true]
    pred_idx = [label_to_idx.get(y, -1) for y in y_pred]

    k = len(labels)
    cm = [[0] * k for _ in range(k)]
    correct = 0
    for t, p in zip(true_idx, pred_idx):
        if t == -1 or p == -1:
            continue
        cm[t][p] += 1
        if t == p:
            correct += 1

    accuracy = correct / n

    per_class_f1 = []
    for i in range(k):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(k) if j != i)
        fn = sum(cm[i][j] for j in range(k) if j != i)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1))

    total = sum(sum(row) for row in cm)
    weighted_f1 = sum(
        f1 * sum(cm[i]) / total if total > 0 else 0.0
        for i, f1 in enumerate(per_class_f1)
    )

    # Pos↔Neg confusion rate (directional error — most critical)
    pos_i = label_to_idx.get("positive", 0)
    neg_i = label_to_idx.get("negative", 1)
    pos_neg_errors = cm[pos_i][neg_i] + cm[neg_i][pos_i]
    directional_error_rate = pos_neg_errors / n

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {lab: f1 for lab, f1 in zip(labels, per_class_f1)},
        "confusion_matrix": cm,
        "directional_error_rate": directional_error_rate,
        "n": n,
    }


def check_level_a(metrics: dict[str, Any], dataset_name: str = "") -> dict[str, bool]:
    """Apply production thresholds from 34_NEWS_GROUND_TRUTH.md §3.1."""
    is_own_gold = "own" in dataset_name.lower() or "ata" in dataset_name.lower()
    f1_thresh = 0.65 if is_own_gold else 0.80

    return {
        "accuracy_ge_85pct": metrics["accuracy"] >= 0.85,
        "macro_f1_ge_thresh": metrics["macro_f1"] >= f1_thresh,
        "directional_error_le_5pct": metrics["directional_error_rate"] <= 0.05,
    }


def load_gold_dataset(path: str | Path) -> tuple[list[str], list[str]]:
    """Load a .jsonl gold dataset. Each line: {text, label}."""
    texts: list[str] = []
    labels: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            texts.append(record["text"])
            labels.append(str(record.get("label") or "").lower())
    return texts, labels


# ---------------------------------------------------------------------------
# Level B — Economic Relevance (Event Study)
# ---------------------------------------------------------------------------


def compute_market_model(
    ticker_returns: pd.Series,
    market_returns: pd.Series,
    estimation_start: pd.Timestamp,
    estimation_end: pd.Timestamp,
    min_obs: int = 100,
) -> tuple[float | None, float | None, float | None]:
    """OLS market model: R_it = alpha + beta * R_mt + eps.

    Returns (alpha, beta, residual_std). Returns (None, None, None) if data
    insufficient.
    """
    mask = (ticker_returns.index >= estimation_start) & (
        ticker_returns.index <= estimation_end
    )
    r_t = ticker_returns[mask].dropna()
    r_m = market_returns.reindex(r_t.index).dropna()
    r_t = r_t.reindex(r_m.index)

    if len(r_t) < min_obs:
        return None, None, None

    x = r_m.values
    y = r_t.values
    x_mean, y_mean = x.mean(), y.mean()
    cov = ((x - x_mean) * (y - y_mean)).mean()
    var_x = ((x - x_mean) ** 2).mean()
    if var_x == 0:
        return None, None, None

    beta = cov / var_x
    alpha = y_mean - beta * x_mean
    residuals = y - (alpha + beta * x)
    return float(alpha), float(beta), float(residuals.std())


def compute_abnormal_returns(
    ticker_returns: pd.Series,
    market_returns: pd.Series,
    alpha: float,
    beta: float,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
) -> pd.Series:
    """AR_it = R_it - (alpha + beta * R_mt) in event window."""
    mask = (ticker_returns.index >= event_start) & (ticker_returns.index <= event_end)
    r_t = ticker_returns[mask].dropna()
    r_m = market_returns.reindex(r_t.index).fillna(0.0)
    expected = alpha + beta * r_m
    return r_t - expected


def event_study(
    events_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    market_returns: pd.Series,
    estimation_window: tuple[int, int] = (-250, -11),
    event_window: tuple[int, int] = (-1, 5),
    min_est_obs: int = 100,
) -> pd.DataFrame:
    """MacKinlay (1997) event study.

    Args:
        events_df: columns [ticker, event_date, sentiment_label]
        returns_df: (date x ticker) daily returns DataFrame
        market_returns: Series of benchmark (SPY) daily returns
        estimation_window: (days_before_event, days_before_event) relative
        event_window: (days_relative, days_relative) around event_date

    Returns:
        DataFrame with [ticker, event_date, label, car, n_event_days, resid_std]
    """
    # Pre-extract per-ticker return series to avoid repeated column lookups
    _rets_by_ticker = {
        t: returns_df[t].dropna()
        for t in events_df["ticker"].unique()
        if t in returns_df.columns
    }

    results = []

    for row in events_df.itertuples(index=False):
        ticker = row.ticker
        event_date = pd.Timestamp(row.event_date)
        label = getattr(row, "sentiment_label", "unknown")

        ticker_rets = _rets_by_ticker.get(ticker)
        if ticker_rets is None:
            continue

        est_start = event_date + pd.Timedelta(days=estimation_window[0])
        est_end = event_date + pd.Timedelta(days=estimation_window[1])
        evt_start = event_date + pd.Timedelta(days=event_window[0])
        evt_end = event_date + pd.Timedelta(days=event_window[1])

        alpha, beta, resid_std = compute_market_model(
            ticker_rets, market_returns, est_start, est_end, min_obs=min_est_obs
        )
        # compute_market_model returns (alpha, beta, resid_std) all-None or
        # all-non-None together; beta is None iff alpha is None, so guarding both
        # never changes which iterations continue — it only narrows for typing.
        if alpha is None or beta is None:
            continue

        ars = compute_abnormal_returns(
            ticker_rets, market_returns, alpha, beta, evt_start, evt_end
        )
        if len(ars) == 0:
            continue

        results.append(
            {
                "ticker": ticker,
                "event_date": event_date,
                "label": label,
                "car": float(ars.sum()),
                "n_event_days": len(ars),
                "resid_std": resid_std,
            }
        )

    return pd.DataFrame(
        results,
        columns=["ticker", "event_date", "label", "car", "n_event_days", "resid_std"],
    )


def car_significance_report(event_study_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """One-sample t-test: mean CAR per label significantly != 0.

    Returns dict[label] with {n, mean_car, median_car, std_car, t_stat, p_value,
    significant_5pct}.
    """
    try:
        from scipy import stats as _stats

        _stats_available = True
    except ImportError:
        _stats_available = False

    report: dict[str, dict[str, Any]] = {}
    for label, group in event_study_df.groupby("label"):
        car_vals = group["car"].dropna().values
        n = len(car_vals)
        mean_car = float(np.mean(car_vals))
        if n < 2:
            report[str(label)] = {
                "n": n,
                "mean_car": mean_car,
                "median_car": mean_car,
                "std_car": 0.0,
                "t_stat": None,
                "p_value": None,
                "significant_5pct": False,
            }
            continue

        std_car = float(np.std(car_vals, ddof=1))
        if _stats_available:
            t_stat, p_value = _stats.ttest_1samp(car_vals, 0.0)
            t_stat, p_value = float(t_stat), float(p_value)
        else:
            # Manual t-stat
            se = std_car / np.sqrt(n)
            t_stat = mean_car / se if se > 0 else 0.0
            # approximate p-value via normal (acceptable for n > 30)
            from math import erfc, sqrt

            p_value = float(erfc(abs(t_stat) / sqrt(2)))

        report[str(label)] = {
            "n": n,
            "mean_car": mean_car,
            "median_car": float(np.median(car_vals)),
            "std_car": std_car,
            "t_stat": t_stat,
            "p_value": p_value,
            "significant_5pct": p_value < 0.05 if p_value is not None else False,
        }
    return report


# ---------------------------------------------------------------------------
# Level C — Tradability (IC / quantile spread / cost-adjusted edge)
# ---------------------------------------------------------------------------


def compute_ic(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    method: str = "spearman",
) -> float:
    """Spearman or Pearson IC between factor and forward returns.

    Both Series must share the same index; NaNs are dropped pairwise.
    """
    combined = pd.concat([factor_values, forward_returns], axis=1).dropna()
    if len(combined) < 10:
        return float("nan")

    x = combined.iloc[:, 0].values
    y = combined.iloc[:, 1].values

    if method == "spearman":
        # Rank transform then Pearson
        x_ranked = _rank(x)
        y_ranked = _rank(y)
        return float(_pearson(x_ranked, y_ranked))
    return float(_pearson(x, y))


def _rank(arr: np.ndarray) -> np.ndarray:
    temp = arr.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(1, len(arr) + 1)
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom == 0:
        return 0.0
    return float((xm * ym).sum() / denom)


def compute_quantile_returns(
    factor_values: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 5,
) -> pd.Series:
    """Mean forward return per factor quantile bucket."""
    combined = pd.concat(
        [factor_values.rename("factor"), forward_returns.rename("fwd")], axis=1
    ).dropna()
    if len(combined) < n_quantiles * 5:
        return pd.Series(dtype=float)

    combined["quantile"] = pd.qcut(
        combined["factor"], q=n_quantiles, labels=False, duplicates="drop"
    )
    return combined.groupby("quantile")["fwd"].mean()


def net_edge_after_costs(
    q_returns: pd.Series,
    turnover_rate: float = 0.20,
    cost_bps_per_side: float = 8.0,
) -> dict[str, float]:
    """Estimate annualised net edge after round-trip transaction costs.

    Args:
        q_returns: daily mean returns per quantile (index 0..N-1)
        turnover_rate: fraction of portfolio turning over per day
        cost_bps_per_side: one-way cost in basis points

    Returns:
        dict with gross_edge_annual_pct, cost_annual_pct, net_edge_annual_pct
    """
    if q_returns.empty:
        return {
            "gross_edge_annual_pct": float("nan"),
            "cost_annual_pct": float("nan"),
            "net_edge_annual_pct": float("nan"),
        }

    q_last = q_returns.index[-1]
    q_first = q_returns.index[0]
    spread = float(q_returns.loc[q_last] - q_returns.loc[q_first])
    daily_cost = 2.0 * cost_bps_per_side * turnover_rate / 10_000.0
    net_daily = spread - daily_cost
    return {
        "gross_edge_annual_pct": spread * 252 * 100,
        "cost_annual_pct": daily_cost * 252 * 100,
        "net_edge_annual_pct": net_daily * 252 * 100,
    }


# ---------------------------------------------------------------------------
# Production Gate
# ---------------------------------------------------------------------------

GATE_THRESHOLDS: dict[str, float] = {
    "level_a_fpb_macro_f1": 0.70,
    "level_a_own_gold_macro_f1": 0.60,
    "level_b_car_significance_p": 0.05,  # must be < threshold
    "level_b_car_magnitude_bps": 30.0,
    "level_c_ic_mean": 0.02,
    "level_c_quantile_spread_bps": 15.0,
    "level_c_net_edge_after_costs_pct": 3.0,
    "look_ahead_anonymization_agreement": 0.85,
}


def news_feature_production_ready(
    feature_name: str,
    validation_results: dict[str, float | None],
) -> tuple[bool, dict[str, bool]]:
    """Gate function from 34_NEWS_GROUND_TRUTH.md §7.

    All 8 criteria must pass for production approval.
    Returns (all_passed, per_criterion_result).
    """
    p_keys = {"level_b_car_significance_p"}
    passed: dict[str, bool] = {}

    for key, threshold in GATE_THRESHOLDS.items():
        val = validation_results.get(key)
        if val is None:
            passed[key] = False
            continue
        if key in p_keys:
            passed[key] = val < threshold
        else:
            passed[key] = val >= threshold

    all_passed = all(passed.values())
    return all_passed, passed


def gate_summary(
    feature_name: str,
    all_passed: bool,
    per_criterion: dict[str, bool],
) -> str:
    """Human-readable gate summary."""
    status = "PRODUCTION-READY" if all_passed else "SHADOW-MODE ONLY"
    lines = [f"Feature: {feature_name}  →  {status}"]
    for k, v in per_criterion.items():
        mark = "✓" if v else "✗"
        lines.append(f"  {mark} {k}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model inference helpers (34_NEWS_GROUND_TRUTH.md §5 — optional deps)
# ---------------------------------------------------------------------------


def run_finbert_tone(texts: list[str]) -> list[str]:
    """Run FinBERT-tone classification on a list of texts.

    Requires: pip install transformers torch (or transformers[torch]).
    Returns list of 'positive' | 'negative' | 'neutral' labels.
    Falls back to ['neutral'] * len(texts) if transformers not available.
    """
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import-not-found]
    except ImportError:
        import logging

        logging.getLogger(__name__).warning(
            "transformers not installed — run_finbert_tone returns neutral for all texts"
        )
        return ["neutral"] * len(texts)

    pipe = hf_pipeline(
        "text-classification",
        model="yiyanghkust/finbert-tone",
        tokenizer="yiyanghkust/finbert-tone",
        max_length=128,
        truncation=True,
    )
    return [pipe(t)[0]["label"].lower() for t in texts]


def run_haiku_zeroshot(texts: list[str], anthropic_client: Any) -> list[str]:
    """3-class zero-shot sentiment via claude-haiku-4-5-20251001.

    Args:
        texts: Headlines or short news snippets.
        anthropic_client: Instantiated anthropic.Anthropic() client.

    Returns list of 'positive' | 'negative' | 'neutral'.
    Falls back to 'neutral' on any API error.
    """
    import logging

    logger = logging.getLogger(__name__)
    results: list[str] = []
    for text in texts:
        try:
            msg = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=5,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Classify the sentiment of this financial headline as exactly one of: "
                            "positive, negative, neutral.\n\nHeadline: " + text
                        ),
                    }
                ],
            )
            label = msg.content[0].text.strip().lower()
            if label not in {"positive", "negative", "neutral"}:
                label = "neutral"
            results.append(label)
        except Exception as exc:
            logger.warning("run_haiku_zeroshot error for text %r: %s", text[:40], exc)
            results.append("neutral")
    return results


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def entity_anonymize(headline: str, ticker: str, company_name: str) -> str:
    """Replace ticker symbol and company name with neutral placeholders.

    Prevents the model from learning a spurious ticker-sentiment association.
    Replacement is case-sensitive for the ticker (tickers are typically uppercase)
    and case-insensitive for the company name.

    Args:
        headline: Raw news headline.
        ticker: Ticker symbol (e.g. 'AAPL').
        company_name: Full company name (e.g. 'Apple Inc.').

    Returns:
        Anonymized headline with ticker → 'XYZ' and company → 'Company XYZ'.
    """
    import re

    result = re.sub(r"\b" + re.escape(ticker) + r"\b", "XYZ", headline)
    result = re.sub(re.escape(company_name), "Company XYZ", result, flags=re.IGNORECASE)
    return result


# ---------------------------------------------------------------------------
# Factor series builder (34_NEWS_GROUND_TRUTH.md §8 — Alphalens integration)
# ---------------------------------------------------------------------------


def build_factor_series(
    news_events_df: "pd.DataFrame",
    tickers_universe: list[str],
    dates: "pd.DatetimeIndex",
) -> "pd.Series":
    """Build a MultiIndex (date, asset) → factor_value Series for Alphalens.

    Args:
        news_events_df: DataFrame with columns ['ticker', 'date', 'sentiment_numeric'].
                        sentiment_numeric in [-1, +1]; multiple events per day are averaged.
        tickers_universe: Full list of tickers in the universe.
        dates: All trading dates to include (missing values filled with 0.0).

    Returns:
        pd.Series with MultiIndex (date, asset) named 'news_sentiment_factor'.
        Missing (date, ticker) combinations are filled with 0.0 (no news = neutral).
    """
    df = news_events_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    grouped = df.groupby(["date", "ticker"])["sentiment_numeric"].mean()

    full_index = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(dates).normalize(), tickers_universe],
        names=["date", "asset"],
    )

    grouped.index.names = ["date", "asset"]
    factor = grouped.reindex(full_index, fill_value=0.0)
    factor.name = "news_sentiment_factor"
    return factor
