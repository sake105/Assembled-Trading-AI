<# ps_py_utils.ps1
   Utilities für Python-Aufrufe in PowerShell und Heredoc-Konvertierung.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-Py {
    <#
      .SYNOPSIS
        Führt Python-Code aus, der als String (Here-String) übergeben wird.
      .EXAMPLE
        $code = @'
        import sys; print(sys.version)
        '@
        Invoke-Py -Code $code
    #>
    param(
        [Parameter(Mandatory=$true)][string]$Code,
        [string]$PythonExe = "$PSScriptRoot\..\..\..\.venv\Scripts\python.exe"
    )
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        # Fallback: "python" im PATH
        $PythonExe = "python"
    }
    $Code | & $PythonExe -  # <<— Kern: pipe nach "python -"
}

function Invoke-PyFile {
    <#
      .SYNOPSIS
        Führt eine Python-Datei mit der .venv-Python aus (oder "python" als Fallback).
    #>
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [string]$PythonExe = "$PSScriptRoot\..\..\..\.venv\Scripts\python.exe",
        [string[]]$Args
    )
    if (-not (Test-Path -LiteralPath $PythonExe)) { $PythonExe = "python" }
    if (-not (Test-Path -LiteralPath $Path)) { throw "Python-Datei nicht gefunden: $Path" }
    & $PythonExe $Path @Args
}

function Test-Python {
    <#
      .SYNOPSIS
        Zeigt, welche Python-Exe verwendet wird und testet einen Mini-Run.
    #>
    param([string]$PythonExe = "$PSScriptRoot\..\..\..\.venv\Scripts\python.exe")
    if (-not (Test-Path -LiteralPath $PythonExe)) { $PythonExe = "python" }
    Write-Host "[PY] Using:" $PythonExe
    "print('ok-from-python')" | & $PythonExe -
}

function Convert-HeredocPY {
    <#
      .SYNOPSIS
        Konvertiert Bash-Heredocs der Form:  python - <<'PY' ... PY
        → PowerShell-kompatibel (Here-String → python -).

      .DESCRIPTION
        Sucht im Text nach Blöcken:
          python - <<'PY'
          <CODE>
          PY
        und ersetzt sie durch:
          $__py = @'
          <CODE>
          '@
          $__py | python -
    #>
    param(
        [Parameter(Mandatory=$true)][string]$Text
    )

    # Regex: "python - <<'PY'\r?\n(.*?)\r?\nPY"
    $pattern = "(?ms)python\s*-\s*<<'?PY'?\s*\r?\n(.*?)\r?\nPY"
    $result = [Regex]::Replace($Text, $pattern, {
        param($m)
        $code = $m.Groups[1].Value
        # PowerShell-Here-String erzeugen
        $replacement = @"
`$__py = @'
$code
'@
`$__py | python -
"@
        return $replacement
    })

    return $result
}

function Convert-FileHeredocs {
    <#
      .SYNOPSIS
        Liest eine Datei ein, ersetzt alle Bash-PY-Heredocs, schreibt Ergebnis.
    #>
    param(
        [Parameter(Mandatory=$true)][string]$InputPath,
        [Parameter(Mandatory=$true)][string]$OutputPath
    )
    if (-not (Test-Path -LiteralPath $InputPath)) { throw "Input nicht gefunden: $InputPath" }
    $raw = Get-Content -LiteralPath $InputPath -Raw
    $out = Convert-HeredocPY -Text $raw
    $outDir = Split-Path -LiteralPath $OutputPath -Parent
    if ($outDir -and -not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
    Set-Content -LiteralPath $OutputPath -Value $out -Encoding UTF8
    Write-Host "[CONVERT] Wrote $OutputPath"
}

# Quality-of-life: kurzer Alias
Set-Alias py Invoke-Py


