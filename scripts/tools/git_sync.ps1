param(
  [string]$Message = "Auto-sync $(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')",
  [switch]$AllowEmpty,
  [string]$Remote = "origin",
  [string]$Branch = "main"
)

$ErrorActionPreference = 'Stop'

function Log([string]$m, [string]$tag = "SYNC"){
  $ts = (Get-Date).ToUniversalTime().ToString('s') + 'Z'
  Write-Host "[$ts] [$tag] $m"
}
# Robust: ignoriert leere Befehle
function Run([string[]]$args){
  if(-not $args -or $args.Count -eq 0){ Log "(noop)"; return }
  Log ($args -join ' ') "GIT"
  & git.exe -c core.longpaths=true -c advice.detachedHead=false -c fetch.prune=true @args
  if($LASTEXITCODE -ne 0){ throw "git $($args -join ' ') failed (exit $LASTEXITCODE)" }
}

# --- Git vorhanden? ---
try{
  $gitv = (& git --version) 2>$null
  if(-not $gitv){ throw "git not found" }
  Log "Using $gitv" "GIT"
}catch{ throw "Git ist nicht im PATH. Bitte installieren oder PATH prüfen." }

# --- Repo-Root Unicode-sicher ---
try{
  $scriptRoot = Split-Path -Parent $PSCommandPath
  $candidate  = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot '..\..'))
  if(Test-Path -LiteralPath $candidate){ Set-Location -LiteralPath $candidate }
  else {
    $root = ((git rev-parse --show-toplevel) -join "`n").Trim()
    if(-not (Test-Path -LiteralPath $root)){ throw "Not a valid path: $root" }
    Set-Location -LiteralPath $root
  }
}
catch{ throw "Kein Git-Repository gefunden. Bitte im Repo starten." }
Log "Repo: $((Get-Location).Path)" "GIT"

# --- Upstream prüfen (stdout/stderr weg) ---
$hasUpstream = $false
try{
  $null = & git rev-parse --abbrev-ref --symbolic-full-name "@{u}" 1>$null 2>$null
  if($LASTEXITCODE -eq 0){ $hasUpstream = $true }
}catch{}

if(-not $hasUpstream){
  Log "Kein Upstream gefunden – versuche $Remote/$Branch zu setzen" "GIT"
  $remotes = (git remote) -split "`n" | ForEach-Object { $_.Trim() } | Where-Object {$_}
  if(-not ($remotes -contains $Remote)){ throw "Remote '$Remote' fehlt (git remote add $Remote <url>)" }
  Run @("fetch","--all","-p")
  Run @("branch","--set-upstream-to=$Remote/$Branch",$Branch)
}

# --- Basiskonfig ---
try{ git config --global pull.rebase true | Out-Null; git config --global fetch.prune true | Out-Null }catch{}

# --- Staging & Commit ---
Log "Staging changes"
Run @("add","-A")
$diff = (git status --porcelain) -join "`n"
if([string]::IsNullOrWhiteSpace($diff)){
  if($AllowEmpty){ Log "Keine Änderungen – erzeuge Empty-Commit"; Run @("commit","--allow-empty","-m",$Message) }
  else { Log "No changes to commit. Use -AllowEmpty to force empty commit." }
} else {
  Run @("commit","-m",$Message)
}

# --- Sync: fetch → rebase → push (Retry) ---
try{
  Run @("fetch","--all","-p")

  try{
    $cur = (git rev-parse --abbrev-ref HEAD).Trim()
    if($cur -ne $Branch){
      Log "Wechsle auf Branch '$Branch'"
      & git checkout $Branch 1>$null 2>$null
      if($LASTEXITCODE -ne 0){
        Log "Branch '$Branch' existiert lokal nicht – erstelle von $Remote/$Branch"
        Run @("checkout","-b",$Branch,"$Remote/$Branch")
      }
    }
  } catch {}

  Log "Rebase auf $Remote/$Branch"
  & git rebase "$Remote/$Branch"
  if($LASTEXITCODE -ne 0){
    & git rebase --abort 1>$null 2>$null
    throw "Rebase-Konflikte. Bitte manuell lösen (git pull --rebase) und erneut ausführen."
  }

  Log "Push versuch 1"
  & git push $Remote $Branch
  if($LASTEXITCODE -ne 0){
    Log "Push abgelehnt – erneuter fetch+rebase+push" "GIT"
    Run @("fetch","--all","-p")
    & git rebase "$Remote/$Branch"
    if($LASTEXITCODE -ne 0){
      & git rebase --abort 1>$null 2>$null
      throw "Rebase-Konflikte beim zweiten Versuch. Manuelles Eingreifen nötig."
    }
    Run @("push",$Remote,$Branch)
  }
  Log "Pushed successfully"
}
catch{
  Log $_.ToString() "ERR"
  throw
}

# --- Status ---
Log "Status:"
& git status -sb
Log "DONE"
