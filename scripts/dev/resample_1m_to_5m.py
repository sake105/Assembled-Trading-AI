#!/usr/bin/env python
import sys
import argparse
import pathlib as pl
import pandas as pd

pd.options.mode.use_inf_as_na = True

def resample_dir(indir: pl.Path) -> pd.DataFrame:
    files = sorted(indir.glob("*.parquet"))
    if not files:
        print("[RESAMPLE] NOFILES in", indir, file=sys.stderr)
        sys.exit(2)

    out_chunks = []
    for p in files:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[RESAMPLE] skip {p.name}: read error -> {e}", file=sys.stderr)
            continue

        # Schema vereinheitlichen
        cols = {c.lower(): c for c in df.columns}
        need = {"timestamp","symbol","close"}
        if not need.issubset({c.lower() for c in df.columns}):
            print(f"[RESAMPLE] skip {p.name}: missing cols -> {df.columns.tolist()}", file=sys.stderr)
            continue

        df = df.rename(columns={cols["timestamp"]:"timestamp",
                                cols["symbol"]:"symbol",
                                cols["close"]:"close"}).loc[:,["timestamp","symbol","close"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp","close"])

        if df.empty:
            print(f"[RESAMPLE] skip {p.name}: empty after clean", file=sys.stderr)
            continue

        # Robustes Resample pro Symbol (symbol explizit wieder anfügen)
        parts = []
        for sym, g in df.groupby("symbol"):
            g = g.sort_values("timestamp")
            g5 = (g.set_index("timestamp")["close"].resample("5min").last().reset_index())
            g5["symbol"] = sym
            parts.append(g5)
        d5 = pd.concat(parts, ignore_index=True)
        d5 = d5.dropna(subset=["close"])
        d5 = d5[["symbol","timestamp","close"]].sort_values(["timestamp","symbol"]).reset_index(drop=True)

        if {"symbol","timestamp","close"} - set(d5.columns):
            print(f"[RESAMPLE] skip {p.name}: bad shape after resample -> {d5.columns.tolist()}", file=sys.stderr)
            continue

        print(f"[RESAMPLE] OK {p.name} -> rows={len(d5)} first={d5['timestamp'].min()} last={d5['timestamp'].max()}")
        out_chunks.append(d5)

    if not out_chunks:
        print("[RESAMPLE] NOFILES or all invalid", file=sys.stderr)
        sys.exit(2)

    out = pd.concat(out_chunks, ignore_index=True)
    out = out.sort_values(["timestamp","symbol"]).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input-dir", required=True, help="Ordner mit 1m-Parquets")
    ap.add_argument("-o","--output",    required=True, help="Ziel-Parquet (5min)")
    args = ap.parse_args()

    indir = pl.Path(args.input_dir)
    if not indir.exists():
        print(f"[RESAMPLE] input dir not found: {indir}", file=sys.stderr)
        sys.exit(2)

    d5 = resample_dir(indir)
    outp = pl.Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    d5.to_parquet(outp, index=False)
    print(f"[RESAMPLE] DONE rows={len(d5)} symbols={d5['symbol'].nunique()} "
          f"first={d5['timestamp'].min()} last={d5['timestamp'].max()} -> {outp}")

if __name__ == "__main__":
    main()
