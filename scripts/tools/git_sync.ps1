param(
  [string]$Message = '',
  [switch]$AllowEmpty = $false,
  [switch]$NoPush = $false,
  [switch]$StashUntracked = $false
)
$ErrorActionPreference = 'Stop'
function Info($m){ $ts=(Get-Date).ToUniversalTime().ToString('s')+'Z'; Write-Host "[$ts] [SYNC] $m" }

# 1) Ensure we're inside a git repo
try {
  git rev-parse --is-inside-work-tree | Out-Null
} catch {
  throw "Not inside a git repository. Run: git init"
}

# 2) Optional: stash untracked before syncing
if($StashUntracked){
  Info 'Stashing untracked changes'
  git stash push --include-untracked -m "pre-sync-$(Get-Date -Format yyyyMMdd_HHmmss)" | Out-Null
}

# 3) Detect changes
$changes = git status --porcelain
if(-not $changes -and -not $AllowEmpty){
  Info 'No changes to commit. Use -AllowEmpty to force empty commit.'
  exit 0
}

# 4) Stage all
Info 'Staging changes'
 git add -A

# 5) Compose message
if([string]::IsNullOrWhiteSpace($Message)){
  $Message = "Auto-sync $(Get-Date -Format yyyy-MM-dd_HH-mm-ss)"
}

# 6) Commit (allow empty if requested)
try {
  if($AllowEmpty){
    git commit --allow-empty -m $Message | Out-Null
  } else {
    git commit -m $Message | Out-Null
  }
  Info "Committed: $Message"
} catch {
  Info 'Nothing to commit (working tree clean)'
  if(-not $NoPush){
    try { git push | Out-Null; Info 'Pushed (no new commits)' } catch { Info "Push skipped: $($_.Exception.Message)" }
  }
  exit 0
}

# 7) Ensure upstream exists (escape @{u} for PowerShell)
$hasUpstream = $false
try {
  git rev-parse --abbrev-ref --symbolic-full-name "@{u}" | Out-Null
  $hasUpstream = $true
} catch {
  $hasUpstream = $false
}
if(-not $hasUpstream){
  $remotes = (git remote) -split "\r?\n" | Where-Object { $_ }
  if($remotes -contains 'origin'){
    git branch -M main | Out-Null
    git push -u origin main | Out-Null
    Info 'Upstream created: origin/main'
  } else {
    Info 'No remote found. Add one with: git remote add origin <URL>'
    if(-not $NoPush){ Info 'Skipping push.' }
    exit 0
  }
}

# 8) Push
if(-not $NoPush){
  try { git push | Out-Null; Info 'Pushed successfully' } catch { throw $_ }
} else {
  Info 'Push disabled by -NoPush'
}

# 9) Show short status
Info 'Status:'
 git status --short
Info 'DONE'
