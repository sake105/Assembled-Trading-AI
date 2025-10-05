import os, warnings
os.environ.setdefault("TQDM_DISABLE","1")

# Nur diese eine FutureWarning von yfinance unterdruecken:
warnings.filterwarnings(
    "ignore",
    message=r"YF\.download\(\) has changed argument auto_adjust default to True",
    category=FutureWarning
)

def _smart_flatten(cols):
    try:
        if hasattr(cols, "levels"):
            tup = [tuple(map(str, c)) for c in cols]
            if tup and all(len(t)==2 for t in tup):
                lv0 = [t[0] for t in tup]
                lv1 = [t[1] for t in tup]
                if len(set(lv1)) == 1:
                    return lv0
                else:
                    return [f"{a}_{b}" for a,b in tup]
    except Exception:
        pass
    out=[]
    for c in cols:
        if isinstance(c, tuple):
            out.append("_".join(str(x) for x in c if str(x)))
        else:
            out.append(str(c))
    return out

def _ensure_datetime_index(df):
    try:
        import pandas as pd
        if hasattr(df, "index") and isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
            except Exception:
                try: df.index = pd.to_datetime(df.index, errors="coerce").tz_convert(None)
                except Exception: pass
    except Exception:
        pass
    return df

def _ensure_date_cols(df):
    try:
        import pandas as pd
        if df is None or not hasattr(df, "columns"): return df
        try:
            if hasattr(df.columns, "levels"):
                df.columns = _smart_flatten(df.columns)
            else:
                df.columns = list(map(str, df.columns))
        except Exception:
            pass
        try:
            if pd.Index(df.columns).duplicated().any():
                df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
        except Exception:
            pass
        df = _ensure_datetime_index(df)
        for c in ("Dividends","Stock Splits"):
            if c not in df.columns:
                try: df[c] = 0.0
                except Exception: pass
        if "Date" not in df.columns and hasattr(df, "index"):
            try:
                from pandas import DatetimeIndex
                if isinstance(df.index, DatetimeIndex):
                    df["Date"] = df.index
            except Exception:
                pass
        if "date" not in df.columns and "Date" in df.columns:
            try: df["date"] = df["Date"]
            except Exception: pass
        if "Adj Close" not in df.columns and "Close" in df.columns:
            try: df["Adj Close"] = df["Close"]
            except Exception: pass
        return df
    except Exception:
        return df

def _filter_kwargs(fn, kw):
    try:
        import inspect
        ps = inspect.signature(fn).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ps.values()):
            return kw
        return {k:v for k,v in kw.items() if k in ps}
    except Exception:
        for bad in ("progress","threads","actions","auto_adjust","group_by"): kw.pop(bad, None)
        return kw

def _patch_yf():
    try:
        import yfinance as yf
        if hasattr(yf, "download"):
            _orig_dl = yf.download
            def dl(*a, **k):
                k = _filter_kwargs(_orig_dl, dict(k))
                for bad in ("progress","threads","group_by"): k.pop(bad, None)
                # Erzwinge das alte Verhalten:
                k.setdefault("auto_adjust", False)
                df = _orig_dl(*a, **k)
                return _ensure_date_cols(df)
            yf.download = dl
        if hasattr(yf, "Ticker"):
            _orig_hist = yf.Ticker.history
            def hist(self, *a, **k):
                k = _filter_kwargs(_orig_hist, dict(k))
                for bad in ("group_by","actions"): k.pop(bad, None)
                # Erzwinge das alte Verhalten:
                k.setdefault("auto_adjust", False)
                df = _orig_hist(self, *a, **k)
                return _ensure_date_cols(df)
            yf.Ticker.history = hist
    except Exception:
        pass

def _patch_pd_io():
    try:
        import pandas as pd
        _rp = pd.read_parquet
        def rp(*a, **k):
            df = _rp(*a, **k); return _ensure_date_cols(df)
        pd.read_parquet = rp
        _rc = pd.read_csv
        def rc(*a, **k):
            df = _rc(*a, **k); return _ensure_date_cols(df)
        pd.read_csv = rc
    except Exception:
        pass

_patch_yf()
_patch_pd_io()
