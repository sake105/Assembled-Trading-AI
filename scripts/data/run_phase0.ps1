# scripts\data\run_phase0.ps1
# Minimal-robust unter PS 5/7: einfache Params, keine ParamSets/Here-Docs.

param(
  [string]$Symbols     = "AAPL,MSFT",
  [string]$Crypto      = "BTC,ETH",
  [string]$Pairs       = "EURUSD,EURGBP",
  [string]$Interval    = "5min",
  [string]$OutputRoot  = "data\raw"
)

$ErrorActionPreference = 'Stop'

function Stamp([string]$tag, [string]$msg) {
  $ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
  Write-Host "[$ts] [$tag] $msg"
}

# --- Pfade stabil ermitteln ---
$scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$repoRoot   = Split-Path -Parent (Split-Path -Parent $scriptRoot)

# --- Python auflösen ---
function Resolve-Python {
  $venvPy = Join-Path $repoRoot ".venv\Scripts\python.exe"
  if (Test-Path -LiteralPath $venvPy) { return $venvPy }

  $pyexe = Get-Command python -ErrorAction SilentlyContinue
  if ($pyexe) { return "python" }

  $pylauncher = Get-Command py.exe -ErrorAction SilentlyContinue
  if ($pylauncher) { return "py -3" }

  return $null
}

# --- Pakete sicherstellen (PowerShell-safe; kein Here-Doc) ---
function Ensure-PipPkgs([string]$pyExe, [string[]]$pkgs) {
  foreach ($p in $pkgs) {
    $code = ('import importlib,sys' + "`n" +
             'try:' + "`n" +
             ('    importlib.import_module("{0}")' -f $p) + "`n" +
             '    sys.exit(0)' + "`n" +
             'except Exception:' + "`n" +
             '    sys.exit(1)')
    & $pyExe -c $code | Out-Null
    $need = ($LASTEXITCODE -ne 0)
    if ($need) {
      Stamp "PYENV" "Installiere Paket: $p"
      & $pyExe -m pip install $p | Out-Host
    }
  }
}

# --- Startinfo ---
Stamp "PYENV" ("Repo : {0}" -f $repoRoot)
Stamp "PYENV" ("Venv : {0}" -f (Join-Path $repoRoot ".venv"))

$py = Resolve-Python
if (-not $py) { throw "Python nicht gefunden (.venv\Scripts\python.exe, python oder py -3)." }

try {
  $ver = & $py -c "import sys;print(sys.version)"
  Stamp "PYENV" ("Python: {0}" -f $ver.Trim())
} catch {
  Stamp "PYENV" "Python-Version konnte nicht gelesen werden."
}

# requirements.txt (wenn vorhanden)
$req = Join-Path $repoRoot "requirements.txt"
if (Test-Path -LiteralPath $req) {
  Stamp "PYENV" "requirements.txt gefunden → installiere…"
  & $py -m pip install --upgrade pip | Out-Host
  & $py -m pip install -r $req      | Out-Host
} else {
  Stamp "PYENV" "requirements.txt fehlt – überspringe Bulk-Install."
}

# Einzelpakete, die wir hier brauchen
Ensure-PipPkgs -pyExe $py -pkgs @("yfinance","pandas","requests","pyarrow","fastparquet")

# --- Outputstruktur anlegen (Join-Path-FIX) ---
$rootOut = Join-Path $repoRoot $OutputRoot

$paths = @()
$paths += Join-Path $rootOut "equities_eod\stooq"
$paths += Join-Path $rootOut ("intraday\alphavantage\{0}" -f $Interval)
$paths += Join-Path $rootOut "crypto\coingecko"
$paths += Join-Path $rootOut "fx\ecb"

foreach ($p in $paths) {
  New-Item -ItemType Directory -Force -Path $p | Out-Null
}

# --- Puller-Skripte ---
$pullDir = Join-Path $repoRoot "scripts\data\pullers"
$stooqPy = Join-Path $pullDir "pull_stooq_eod.py"
$alphaPy = Join-Path $pullDir "pull_alpha_intraday.py"
$cgPy    = Join-Path $pullDir "pull_coingecko_ohlc.py"
$ecbPy   = Join-Path $pullDir "pull_ecb_fxref.py"

Stamp "LIVE" ("Repo:   {0}" -f $repoRoot)
Stamp "LIVE" ("Output: {0}" -f $rootOut)
Stamp "LIVE" ("Symbols: {0}" -f $Symbols)
Stamp "LIVE" ("Crypto:  {0}" -f $Crypto)
Stamp "LIVE" ("Pairs:   {0}" -f $Pairs)
Stamp "LIVE" ("Interv.: {0}" -f $Interval)

# --- 1) Equities EoD (yfinance)
$stooqPy = Join-Path $pullDir "pull_stooq_eod.py"
if (Test-Path -LiteralPath $stooqPy) {
  $out = Join-Path $rootOut "equities_eod\stooq"
  Stamp "PULL" "EoD (yfinance) → $out"
  & $py $stooqPy --symbols $Symbols --years 5 --out $out
  $exit = $LASTEXITCODE
  if ($exit -eq 0) {
    # ok
  } elseif ($exit -eq 2) {
    Stamp "PULL" "WARN: EoD leer (ExitCode 2) – fahre fort."
  } else {
    throw "Stooq Pull fehlgeschlagen (ExitCode $exit)."
  }
} else {
  Stamp "PULL" "Stooq-Skript fehlt, überspringe: $stooqPy"
}


# --- 2) Intraday via yfinance (kein Key)
$alphaPy = Join-Path $pullDir "pull_alpha_intraday.py"
if (Test-Path -LiteralPath $alphaPy) {
  $out = Join-Path $rootOut ("intraday\alphavantage\{0}" -f $Interval)
  Stamp "PULL" "Intraday (yfinance, $Interval) → $out"
  & $py $alphaPy --symbols $Symbols --interval $Interval --days 5 --out $out
  if ($LASTEXITCODE -ne 0) { throw "Intraday Pull fehlgeschlagen (ExitCode $LASTEXITCODE)." }
} else {
  Stamp "PULL" "Intraday-Skript fehlt, überspringe: $alphaPy"
}

# --- 3) CoinGecko OHLC (optional)
if (Test-Path -LiteralPath $cgPy) {
  $out = Join-Path $rootOut "crypto\coingecko"
  Stamp "PULL" "CoinGecko OHLC → $out"
  & $py $cgPy --coins $Crypto --out $out
  if ($LASTEXITCODE -ne 0) { throw "CoinGecko Pull fehlgeschlagen (ExitCode $LASTEXITCODE)." }
} else {
  Stamp "PULL" "CoinGecko-Skript fehlt, überspringe: $cgPy"
}

# --- 4) EZB FX Referenzkurse (optional)
if (Test-Path -LiteralPath $ecbPy) {
  $out = Join-Path $rootOut "fx\ecb"
  Stamp "PULL" "ECB FX Referenzkurse → $out"
  & $py $ecbPy --pairs $Pairs --out $out
  if ($LASTEXITCODE -ne 0) { throw "ECB Pull fehlgeschlagen (ExitCode $LASTEXITCODE)." }
} else {
  Stamp "PULL" "ECB-Skript fehlt, überspringe: $ecbPy"
}

Stamp "DONE" "Phase-0 Dateningest abgeschlossen."

