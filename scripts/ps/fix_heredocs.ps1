<# fix_heredocs.ps1
   Konvertiert alle Dateien im angegebenen Ordner, die Bash-Heredocs mit PY enthalten.
#>
param(
  [string]$Root = ".",
  [string[]]$Include = @("*.md","*.txt","*.ps1"),
  [string]$OutSuffix = ".ps1"
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here "ps_py_utils.ps1")  # lädt Convert-HeredocPY

$files = foreach ($pat in $Include) {
  Get-ChildItem -Path $Root -Include $pat -Recurse -File -ErrorAction SilentlyContinue
}

foreach ($f in $files) {
  $raw = Get-Content -LiteralPath $f.FullName -Raw
  if ($raw -match "python\s*-\s*<<'?PY'?") {
    $converted = Convert-HeredocPY -Text $raw
    $outPath = "$($f.FullName)$OutSuffix"
    Set-Content -LiteralPath $outPath -Value $converted -Encoding UTF8
    Write-Host "[OK] $($f.FullName) -> $outPath"
  }
}

