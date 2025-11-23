param(
  [string]$DefaultBranch = 'main',
  [string]$InitialMessage = 'Sprint 10: project init',
  [switch]$WithLFS = $false
)
$ErrorActionPreference = 'Stop'
$ROOT = (Get-Location).Path
function Info($m){ $ts=(Get-Date).ToUniversalTime().ToString('s')+'Z'; Write-Host "[$ts] [GIT] $m" }

# 1) Guard: ensure git exists
$git = (Get-Command git -ErrorAction Stop).Source
Info "Using git: $git"

# 2) Create .gitignore if missing
$gi = Join-Path $ROOT '.gitignore'
if(-not (Test-Path $gi)){
  Info 'Writing .gitignore'
  @(
    '# Python',
    '__pycache__/',
    '*.py[cod]',
    '*.pyo',
    '*.pyd',
    '',
    '# Virtual env',
    '.venv/',
    '',
    '# VS Code / IDE',
    '.vscode/',
    '.idea/',
    '*.code-workspace',
    '',
    '# OS junk',
    '.DS_Store',
    'Thumbs.db',
    '',
    '# Project outputs',
    'output/',
    'logs/',
    '',
    '# Data caches & temp',
    '*.tmp',
    '*.log',
    '',
    '# Optional: raw data can be big; uncomment to ignore',
    '# data/raw/',
    '',
    '# Optional: parquet/feather via Git LFS',
    '# *.parquet',
    '# *.feather'
  ) | Set-Content -LiteralPath $gi -Encoding utf8NoBOM
}

# 3) Initialize repo if needed
if(-not (Test-Path (Join-Path $ROOT '.git'))){
  Info 'git init'
  git init | Out-Null
  git branch -m $DefaultBranch | Out-Null
}

# 4) Optional: Git LFS
if($WithLFS){
  try{
    git lfs --version | Out-Null
    Info 'Configuring Git LFS (parquet/feather)'
    git lfs install | Out-Null
    git lfs track '*.parquet' '*.feather'
  } catch {
    Info 'Git LFS not installed; skipping LFS setup'
  }
}

# 5) Initial commit
Info 'Staging files'
 git add .
Info "Committing: $InitialMessage"
 git commit -m $InitialMessage

# 6) Show short status
Info 'Status:'
 git status --short
Info 'DONE'

