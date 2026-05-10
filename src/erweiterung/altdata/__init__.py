"""Kostenlose alternative Datenquellen.

Jeder Fetcher liefert ein ``FetchResult`` mit deterministischem DataFrame-Schema.
Disk-Cache via Parquet (siehe ``_base.get_cache_dir``). Rate-Limits konservativ.

Quellenuebersicht
-----------------
- wikipedia_pageviews : Wikipedia REST API (offizielle Pageview-API, frei)
- google_trends       : pytrends (inoffiziell, rate-limited)
- gdelt_extended      : GDELT 2.0 GKG/Events (frei via BigQuery / web-DL)
- fred_md             : FRED-MD monatliches Macro-Panel (frei, McCracken/Ng)
- yahoo_options       : Yahoo Finance Options Chains via yfinance (frei)
- sec_edgar           : SEC EDGAR Filings (offizielle JSON-Index, frei)
- finra_short         : FINRA Short Interest (offiziell veröffentlichte CSVs)
- cftc_cot            : CFTC Commitments of Traders (offiziell, wöchentlich)
- reddit_pushshift    : Reddit Sentiment via pushshift.io / pmaw (frei, archiv)
- coingecko_crypto    : CoinGecko free tier (Crypto-Korrelations-Signale)
- worldbank_macro     : World Bank Open Data (frei)
"""

from erweiterung._base import FetchResult

__all__ = [
    "FetchResult",
]
