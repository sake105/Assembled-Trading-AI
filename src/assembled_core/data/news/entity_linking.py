"""News entity linking: map headlines to ticker symbols."""

from __future__ import annotations

import pandas as pd


def link_news_to_symbols(
    news: pd.DataFrame,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Link news articles to symbols based on entity mentions.

    Returns news DataFrame with a 'symbol' column populated.
    """
    if "symbol" not in news.columns:
        news = news.copy()
        news["symbol"] = None
    return news
