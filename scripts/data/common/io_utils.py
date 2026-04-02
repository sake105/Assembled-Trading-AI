from __future__ import annotations
p.mkdir(parents=True, exist_ok=True)


# --- HTTP (stdlib, ohne requests) ---
def http_get_json(url: str, headers: dict | None = None, retries: int = 3, backoff: float = 0.8):
h = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
if headers:
h.update(headers)
last_ex = None
for i in range(retries):
try:
req = Request(url, headers=h)
with urlopen(req, timeout=30) as resp:
data = resp.read()
if resp.headers.get("Content-Encoding") == "gzip":
data = gzip.decompress(data)
return json.loads(data.decode("utf-8"))
except (HTTPError, URLError, TimeoutError) as ex:
last_ex = ex
time.sleep((i+1)*backoff)
raise last_ex


def http_get_text(url: str, headers: dict | None = None):
h = {"User-Agent": USER_AGENT}
if headers:
h.update(headers)
req = Request(url, headers=h)
with urlopen(req, timeout=30) as resp:
txt = resp.read().decode("utf-8", errors="replace")
return txt


# --- Parquet/CSV helpers ---
SCHEMA_EQ = ["timestamp","symbol","open","high","low","close","volume","provider"]


def to_parquet(df: pd.DataFrame, out_path: Path):
ensure_dir(out_path.parent)
df.to_parquet(out_path, index=False)


# Harmonisierung für OHLC Frames


def normalize_ohlc(df: pd.DataFrame, symbol: str, provider: str, tz="UTC") -> pd.DataFrame:
cols = {c.lower(): c for c in df.columns}
rename = {}
for k in ["open","high","low","close","volume"]:
for c in list(df.columns):
if c.lower() == k:
rename[c] = k
break
if "timestamp" not in [c.lower() for c in df.columns]:
# häufige Varianten
cand = ["time","date","datetime","timestamp"]
for c in df.columns:
if c.lower() in cand:
rename[c] = "timestamp"
break
df = df.rename(columns=rename)
if "timestamp" not in df.columns:
raise ValueError("timestamp column not found after rename")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["symbol"] = symbol
df["provider"] = provider
# minimale Sortierung & Spaltenreihenfolge
keep = [c for c in SCHEMA_EQ if c in df.columns]
df = df[keep].sort_values(["timestamp","symbol"]).reset_index(drop=True)
return df