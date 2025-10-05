param(
  [int]$SpacesPerTab = 4,
  [switch]$OnlyLeading = $true,
  [switch]$TrimTrailing = $true,
  [switch]$EnsureUtf8 = $true,
  [switch]$DryRun = $false,
  [switch]$Backup = $true
)
$ErrorActionPreference = 'Stop'
$ROOT = (Get-Location).Path
$Fix = Join-Path $ROOT 'scripts/tools/fix_indent.ps1'
if(-not (Test-Path $Fix)){ throw "Not found: $Fix" }

$targets = @('scripts','docs','output','data')
Write-Host "[FIX] running on: $($targets -join ', ')"
& pwsh -File $Fix -Paths $targets -SpacesPerTab $SpacesPerTab -OnlyLeading:$OnlyLeading -TrimTrailing:$TrimTrailing -EnsureUtf8:$EnsureUtf8 -DryRun:$DryRun -Backup:$Backup
if($LASTEXITCODE -ne 0){ throw "Fix failed" }
Write-Host "[FIX] DONE"


