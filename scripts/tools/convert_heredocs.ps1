param(
  [Parameter(Mandatory)] [string]$InputPath,
  [Parameter()] [string]$OutFile = $( [IO.Path]::ChangeExtension($InputPath, ".ps1") )
)

$raw = Get-Content -LiteralPath $InputPath -Raw -Encoding UTF8

# Regex: python - <<'PY'  ...  PY   (multiline, non-greedy)
$rx = [regex]"python\s*-\s*<<'PY'\s*(?<code>[\s\S]*?)\s*PY"

$converted = $rx.Replace($raw, {
  param($m)
  $py = $m.Groups['code'].Value -replace "`r?`n$", ''  # letztes NL weg
  @"
# --- Auto-generated from heredoc ---
`$__tmp = [System.IO.Path]::GetTempFileName().Replace('.tmp','.py')
@'
$py
'@ | Set-Content -LiteralPath `$__tmp -Encoding UTF8
& $env:VIRTUAL_ENV\Scripts\python.exe `$__tmp
Remove-Item -LiteralPath `$__tmp -ErrorAction SilentlyContinue
"@
})

$null = New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($OutFile))
Set-Content -LiteralPath $OutFile -Value $converted -Encoding UTF8
Write-Host "[OK] Wrote $OutFile"


