# H-086/W3 E-051-Re-Run-Driver: laeuft ein Screen-Script 2x (PYTHONHASHSEED 0/42),
# sichert die Result-JSONs je Seed und vergleicht sie byte-genau.
# Laufzeit-Cap je Lauf: 15 min (900 s) -> bei Ueberschreitung Kill + TIMEOUT-Meldung.
param(
    [Parameter(Mandatory = $true)][string]$ScriptName,   # z.B. h029_13f_consensus.py
    [Parameter(Mandatory = $true)][string]$ResultJson    # z.B. h029_results.json
)
$mandat = "F:\Python_Projekt\Aktiengerüst\research\mandat"
$py = "F:\Python_Projekt\Aktiengerüst\.venv\Scripts\python.exe"
$results = Join-Path $mandat "results"
$stem = [IO.Path]::GetFileNameWithoutExtension($ResultJson)

foreach ($seed in @('0', '42')) {
    $env:PYTHONHASHSEED = $seed
    $log = Join-Path $results "$stem.rerun_seed$seed.log"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $py -ArgumentList "`"$(Join-Path $mandat $ScriptName)`"" `
        -WorkingDirectory $mandat -NoNewWindow -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    if (-not $p.WaitForExit(900000)) {
        try { $p.Kill($true) } catch {}
        Write-Output "[TIMEOUT] $ScriptName seed=$seed > 15min -> abgebrochen (nicht re-validiert)"
        exit 2
    }
    $sw.Stop()
    Write-Output "[OK] $ScriptName seed=$seed exit=$($p.ExitCode) dauer=$([int]$sw.Elapsed.TotalSeconds)s"
    if ($p.ExitCode -ne 0) { Write-Output "[ERROR] exit code != 0, siehe $log.err"; exit 1 }
    Copy-Item (Join-Path $results $ResultJson) (Join-Path $results "$stem.rerun_seed$seed.json") -Force
}
$h0 = (Get-FileHash (Join-Path $results "$stem.rerun_seed0.json")).Hash
$h42 = (Get-FileHash (Join-Path $results "$stem.rerun_seed42.json")).Hash
Write-Output "[DETERMINISM] seed0==seed42 byte-identisch: $($h0 -eq $h42)"
