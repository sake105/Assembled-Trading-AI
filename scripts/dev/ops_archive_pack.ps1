# Ops Evidence Archive: build + verify + archive in one PowerShell function.
# Usage: Source this file, then call New-OpsEvidenceArchive -RunId "ledger_eod_1d" -AsOfDate "2025-01-15" -OutputDir "output" -ArchiveDir "archive"
# Exit codes: 0 = success; 2 = Export failed or invalid JSON; 3 = Verify failed (ok=false); 4 = Copy/Archive failed.
# ASCII-only messages. Stdout is parsed as JSON; stderr is logged separately.

function New-OpsEvidenceArchive {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RunId,
        [Parameter(Mandatory = $true)]
        [string]$AsOfDate,
        [Parameter(Mandatory = $true)]
        [string]$OutputDir,
        [Parameter(Mandatory = $true)]
        [string]$ArchiveDir,
        [switch]$Strict,
        [switch]$NoOptional,
        [string]$VerifyJsonOut = ""
    )
    $ErrorActionPreference = "Stop"
    $exportArgs = @("scripts/export_evidence_pack.py", "--run-id", $RunId, "--as-of-date", $AsOfDate, "--output-dir", $OutputDir)
    if ($Strict) { $exportArgs += "--strict" }
    if ($NoOptional) { $exportArgs += "--no-optional" }
    $exportStdout = py -3 $exportArgs 2>&1 | Out-String
    $exportExit = $LASTEXITCODE
    if ($exportExit -ne 0) {
        Write-Error "Export failed (exit $exportExit)"
        exit 2
    }
    if ($VerifyJsonOut -ne "" -and (Test-Path -LiteralPath $VerifyJsonOut -PathType Container)) {
        $exportJsonFile = Join-Path $VerifyJsonOut "export_${RunId}_${AsOfDate}.json"
        $exportStdout | Set-Content -Path $exportJsonFile -Encoding utf8 -NoNewline:$false
    }
    try {
        $exportJson = $exportStdout | ConvertFrom-Json
    } catch {
        Write-Error "Export output is not valid JSON"
        exit 2
    }
    $req = @("ok", "pack_path", "pack_manifest_path")
    foreach ($k in $req) {
        if (-not (Get-Member -InputObject $exportJson -Name $k -MemberType Properties)) {
            Write-Error "Export JSON missing key: $k"
            exit 2
        }
    }
    if (-not $exportJson.ok) {
        Write-Error "Export returned ok=false"
        exit 2
    }
    $packPath = $exportJson.pack_path -replace '\\', '/'
    $zipFullPath = Join-Path $OutputDir ($packPath -replace '/', [System.IO.Path]::DirectorySeparatorChar)
    $verifyStdout = py -3 scripts/verify_evidence_pack.py --zip $zipFullPath 2>&1 | Out-String
    $verifyExit = $LASTEXITCODE
    if ($VerifyJsonOut -ne "" -and (Test-Path -LiteralPath $VerifyJsonOut -PathType Container)) {
        $verifyJsonFile = Join-Path $VerifyJsonOut "verify_${RunId}_${AsOfDate}.json"
        $verifyStdout | Set-Content -Path $verifyJsonFile -Encoding utf8 -NoNewline:$false
    }
    if ($verifyExit -ne 0) {
        Write-Error "Verify failed (exit $verifyExit)"
        exit 3
    }
    try {
        $verifyJson = $verifyStdout | ConvertFrom-Json
    } catch {
        Write-Error "Verify output is not valid JSON"
        exit 3
    }
    if (-not (Get-Member -InputObject $verifyJson -Name "ok" -MemberType Properties)) {
        Write-Error "Verify JSON missing key: ok"
        exit 3
    }
    if (-not (Get-Member -InputObject $verifyJson -Name "error_code" -MemberType Properties)) {
        Write-Error "Verify JSON missing key: error_code"
        exit 3
    }
    if (-not $verifyJson.ok) {
        $ec = $verifyJson.error_code
        Write-Error "Verify returned ok=false error_code=$ec - not archiving"
        exit 3
    }
    if (-not (Test-Path -LiteralPath $ArchiveDir)) {
        New-Item -ItemType Directory -Path $ArchiveDir -Force | Out-Null
    }
    $archiveName = "pack_${RunId}_${AsOfDate}.zip"
    $archivePath = Join-Path $ArchiveDir $archiveName
    try {
        Copy-Item -LiteralPath $zipFullPath -Destination $archivePath -Force
    } catch {
        Write-Error "Copy/Archive failed: $_"
        exit 4
    }
    Write-Host "Archived to $archivePath"
    exit 0
}
