"""Evidence check: which FREE source actually returns price history for known
DELISTED / ACQUIRED / BANKRUPT tickers (the survivorship-bias hole)? Stooq direct
CSV (ToS-clean, no key) + yfinance. Read-only, just prints coverage."""

from __future__ import annotations
import io
import urllib.request
import pandas as pd

# (ticker, what happened)
NAMES = [
    ("AAPL", "control: still listed"),
    ("CELG", "Celgene -> acquired by BMY 2019"),
    ("ATVI", "Activision -> acquired by MSFT 2023"),
    ("TWTR", "Twitter -> taken private 2022"),
    ("SIVB", "SVB Financial -> bankrupt Mar 2023"),
    ("FRC", "First Republic -> failed/seized May 2023"),
    ("XLNX", "Xilinx -> acquired by AMD 2022"),
    ("CERN", "Cerner -> acquired by ORCL 2022"),
    ("LEH", "Lehman -> bankrupt 2008"),
    ("BSC", "Bear Stearns -> JPM 2008"),
    ("WCOM", "WorldCom -> bankrupt 2002"),
    ("ENRNQ", "Enron -> bankrupt 2001"),
]


def stooq(ticker):
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
        raw = urllib.request.urlopen(req, timeout=15).read().decode()
        if "Date" not in raw[:50]:
            return None
        df = pd.read_csv(io.StringIO(raw))
        if df.empty or "Date" not in df.columns:
            return None
        return df
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def yf(ticker):
    try:
        import yfinance as yflib

        df = yflib.download(ticker, period="max", progress=False, auto_adjust=False)
        return df if df is not None and len(df) else None
    except Exception as e:
        return f"ERR:{type(e).__name__}"


print(f"{'ticker':7} {'event':38} | {'STOOQ rows/range':28} | yfinance")
print("-" * 110)
for t, ev in NAMES:
    s = stooq(t)
    if isinstance(s, pd.DataFrame):
        srep = f"{len(s):5d}  {s['Date'].iloc[0]}..{s['Date'].iloc[-1]}"
    else:
        srep = str(s)
    y = yf(t)
    if isinstance(y, pd.DataFrame):
        yrep = f"{len(y)} rows {y.index.min().date()}..{y.index.max().date()}"
    else:
        yrep = str(y)
    print(f"{t:7} {ev:38} | {srep:28} | {yrep}")
print("\n[DONE] free-data coverage probe")
