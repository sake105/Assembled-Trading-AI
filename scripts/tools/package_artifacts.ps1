param(
  [string]$OutputDir = (Join-Path $PSScriptRoot "..\..\output")
)

$OutputDir = Resolve-Path $OutputDir
$stamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
$zipPath = Join-Path $OutputDir "run_artifacts_$stamp.zip"

if(Test-Path $zipPath){ Remove-Item $zipPath -Force }

# .NET ZipFile
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($OutputDir, $zipPath)

$zipPathResolved = Resolve-Path $zipPath
Write-Host "[ARTIFACTS] Created $zipPathResolved"
$zipPathResolved
