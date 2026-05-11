"""Cross-Asset News-Spillover — Welche News über Asset A beeinflussen Asset B?

Theorie
-------
1. **Co-Mention-Network**: Wenn ein Headline beide A und B nennt, sind sie in
   "narrative-cluster". → A's sentiment leakt zu B.
2. **Supply-Chain-Linkage**: News über Lieferanten bewegen Endkunden.
3. **Sector-Spillover**: News über Sector-Leader → Sector.

Mathematisch
------------
Spillover-Matrix S_AB = correlation(sentiment_A, return_B_{t+h}) für h > 0.
Hohe S_AB ⇒ Sentiment(A) prognostiziert Return(B).

Anwendung
---------
- Trade B based on news about A (lead-lag).
- Industry/sector-level news-aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def co_mention_matrix(
    news_df: pd.DataFrame,
    text_col: str = "headline",
    symbol_col: str = "symbol",
    min_co_mentions: int = 3,
) -> pd.DataFrame:
    """Build symbol-symbol co-mention matrix from headlines.

    Args:
        news_df: DataFrame with ``[headline, symbol]`` — multiple rows per
            article (one per mentioned symbol) ODER mit ``article_id`` Spalte.
        min_co_mentions: minimum joint mentions to register link.

    Returns:
        Symbol × Symbol DataFrame with co-mention counts.
    """
    if news_df.empty:
        return pd.DataFrame()
    # Group by headline text (assume identical text = same article)
    grouped = news_df.groupby(text_col)[symbol_col].apply(set)
    symbols_all = sorted(set(news_df[symbol_col].unique()))
    n = len(symbols_all)
    idx = {s: i for i, s in enumerate(symbols_all)}
    M = np.zeros((n, n), dtype=int)
    for symbols_in_article in grouped:
        symbols_list = list(symbols_in_article)
        for i in range(len(symbols_list)):
            for j in range(i + 1, len(symbols_list)):
                a, b = idx[symbols_list[i]], idx[symbols_list[j]]
                M[a, b] += 1
                M[b, a] += 1
    df = pd.DataFrame(M, index=symbols_all, columns=symbols_all)
    df = df.where(df >= min_co_mentions, 0)
    return df


def sentiment_spillover_matrix(
    news_df: pd.DataFrame,
    returns_panel: pd.DataFrame,
    horizon_days: int = 5,
    sentiment_col: str = "sentiment",
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Spillover-Matrix S_AB = corr(sentiment_A_t, cum_return_B_{t+1..t+h}).

    Args:
        news_df: [date, symbol, sentiment].
        returns_panel: [date, symbol, return].
        horizon_days: forward cumulative return horizon.

    Returns:
        DataFrame N×N — S_AB = how much A's sentiment predicts B's forward return.
    """
    if news_df.empty or returns_panel.empty:
        return pd.DataFrame()
    # Build daily sentiment panel
    news_df = news_df.copy()
    news_df[date_col] = pd.to_datetime(news_df[date_col], utc=True).dt.normalize()
    sent_panel = (
        news_df.groupby([date_col, symbol_col])[sentiment_col]
        .mean()
        .unstack(fill_value=0)
    )
    # Forward cumulative returns
    pivot_ret = returns_panel.pivot_table(
        index=date_col, columns=symbol_col, values="return"
    )
    pivot_ret.index = pd.to_datetime(pivot_ret.index, utc=True).normalize()
    forward = pivot_ret.shift(-1).rolling(horizon_days).sum().shift(-(horizon_days - 1))

    common_dates = sent_panel.index.intersection(forward.index)
    if len(common_dates) < 30:
        return pd.DataFrame()
    sent_panel = sent_panel.loc[common_dates]
    forward = forward.loc[common_dates]

    common_syms = sent_panel.columns.intersection(forward.columns)
    sent_syms = sent_panel[common_syms]
    fwd_syms = forward[common_syms]

    # Drop rows where ALL sentiment or ALL forward are NaN — sonst per-pair
    aligned = pd.concat({"s": sent_syms, "f": fwd_syms}, axis=1)
    aligned = aligned.dropna(how="any")
    n = len(common_syms)
    if len(aligned) < 30:
        return pd.DataFrame(np.zeros((n, n)), index=common_syms, columns=common_syms)

    sent_arr = aligned["s"][common_syms].values
    fwd_arr = aligned["f"][common_syms].values
    # Vectorized cross-correlation via standardization + dot-product / n.
    # Σ_AB = corr(sent_A, fwd_B) = E[(sent_A − μ_A_s)(fwd_B − μ_B_f)] / (σ_A_s σ_B_f)
    mu_s = sent_arr.mean(axis=0)
    mu_f = fwd_arr.mean(axis=0)
    sd_s = sent_arr.std(axis=0, ddof=0)
    sd_f = fwd_arr.std(axis=0, ddof=0)
    # Mask zero-std assets
    sd_s_safe = np.where(sd_s == 0, 1, sd_s)
    sd_f_safe = np.where(sd_f == 0, 1, sd_f)
    sz = (sent_arr - mu_s) / sd_s_safe
    fz = (fwd_arr - mu_f) / sd_f_safe
    M = (sz.T @ fz) / len(aligned)  # (n, n) cross-correlation
    # Zero out rows/cols where one std was 0
    zero_s = sd_s == 0
    zero_f = sd_f == 0
    if zero_s.any():
        M[zero_s, :] = 0
    if zero_f.any():
        M[:, zero_f] = 0
    return pd.DataFrame(M, index=common_syms, columns=common_syms)


def propagate_news_to_followers(
    news_df: pd.DataFrame,
    spillover_matrix: pd.DataFrame,
    sentiment_col: str = "sentiment",
    threshold: float = 0.15,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Propagate news-sentiment to "follower" assets via spillover-matrix.

    For each news about A, generate signals for all B with S_AB > threshold.

    Returns:
        DataFrame [date, target_symbol, propagated_sentiment].
    """
    if spillover_matrix.empty:
        return pd.DataFrame()
    rows = []
    for _, r in news_df.iterrows():
        src = r[symbol_col]
        if src not in spillover_matrix.index:
            continue
        for tgt, s_ab in spillover_matrix.loc[src].items():
            if s_ab < threshold or src == tgt:
                continue
            rows.append(
                {
                    date_col: r[date_col],
                    "source_symbol": src,
                    "target_symbol": tgt,
                    "propagated_sentiment": float(r[sentiment_col]) * float(s_ab),
                    "spillover_strength": float(s_ab),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "co_mention_matrix",
    "sentiment_spillover_matrix",
    "propagate_news_to_followers",
]
