"""Read-only data-reality check for Fable exploration. Prints schemas/coverage
for the new PIT data + the price store. No writes outside stdout."""

from __future__ import annotations
import os
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def describe(path, n=4):
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return None
    df = pd.read_parquet(path)
    print(f"[OK] {path}  shape={df.shape}")
    print("dtypes:\n", df.dtypes)
    print("head:\n", df.head(n))
    return df


# 1. Fundamentals
banner("FUNDAMENTALS (XBRL company facts)")
f = describe("data/raw/fundamentals/fundamentals_xbrl_full.parquet")
if f is not None:
    print("\nn symbols:", f["symbol"].nunique())
    print("available_at notna%:", round(100 * f["available_at"].notna().mean(), 2))
    eps = f[f["tag"].str.contains("EarningsPerShare", case=False, na=False)]
    print("\nEPS-like tags:\n", eps["tag"].value_counts().head(20))
    print("\nforms:\n", f["form"].value_counts().head(12))
    # quarterly EPS diluted coverage
    diff = (pd.to_datetime(f["period_end"]) - pd.to_datetime(f["period_start"])).dt.days
    f2 = f.assign(dur=diff)
    q = f2[(f2["tag"] == "EarningsPerShareDiluted") & (f2["dur"].between(80, 100))]
    print("\nquarterly EPSDiluted rows:", len(q), "symbols:", q["symbol"].nunique())
    print(
        "per-symbol quarterly EPS count (describe):\n",
        q.groupby("symbol").size().describe(),
    )

# 2. Insider Form 4
banner("INSIDER (SEC Form 4)")
ins = describe("data/raw/insider_congress/form4_insider_full.parquet")
if ins is not None:
    print("\nn symbols:", ins["symbol"].nunique() if "symbol" in ins else "?")
    for c in ("transaction_type", "type", "acquired_disposed", "is_derivative"):
        if c in ins.columns:
            print(f"\n{c}:\n", ins[c].value_counts(dropna=False).head(10))
    for c in ("available_at", "transaction_date", "filing_date"):
        if c in ins.columns:
            s = pd.to_datetime(ins[c], errors="coerce", utc=True)
            print(
                f"{c}: min={s.min()} max={s.max()} notna%={round(100 * s.notna().mean(), 1)}"
            )

# 3. Congress
banner("CONGRESS (STOCK-Act PTR)")
c = describe("data/raw/insider_congress/congress_trades_full.parquet")
if c is not None:
    for col in ("transaction_type", "type", "chamber"):
        if col in c.columns:
            print(f"\n{col}:\n", c[col].value_counts(dropna=False).head(8))

# 4. Price store
banner("PRICE STORE (live aggregates)")
for p in ("output/aggregates/daily.parquet",):
    d = describe(p)
    if d is not None:
        dt = pd.to_datetime(d["date"] if "date" in d else d.index, errors="coerce")
        print("date range:", dt.min(), "->", dt.max())
        if "symbol" in d:
            print("n symbols:", d["symbol"].nunique())

# 5. Universe
banner("UNIVERSE")
u = "data/universe/master_universe_panel.csv"
if os.path.exists(u):
    udf = pd.read_csv(u)
    print(udf.shape, list(udf.columns))
    print(udf.head(6))
