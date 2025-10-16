<#
.SYNOPSIS
  Sendet eine Nachricht (mit optionalen Ergebnissen) an einen Discord-Webhook.

.DESCRIPTION
  - Liest Webhook-URL in dieser Reihenfolge:
      1) -Webhook Parameter
      2) $env:DISCORD_WEBHOOK
      3) .env im Repo-Root (KEY=VALUE)
      4) Interaktive Eingabe (falls alles andere fehlt)
  - Optionales Persistieren der URL als User-Umgebungsvariable.
  - Kann Ergebnisdaten als JSON-Datei einlesen und als Embed-Felder anhängen.
  - Robuste Validierung + klare Fehlermeldungen.

.PARAMETER Title
  Titel im Embed.

.PARAMETER Content
  Beschreibung/Text im Embed.

.PARAMETER Webhook
  (Optional) Überschreibt die Webhook-URL.

.PARAMETER Persist
  (Optional) Speichert die Webhook-URL dauerhaft als User-Umgebungsvariable (DISCORD_WEBHOOK).

.PARAMETER DataJsonPath
  (Optional) Pfad zu einer JSON-Datei mit Key/Value-Ergebnissen, die als Embed-Felder angehängt werden.
  Beispielinhalt:
    {
      "PF": 1.96,
      "Sharpe": -3.69,
      "Trades": 76,
      "Report": "output/performance_report.md"
    }

.PARAMETER ExtraFields
  (Optional) Hashtable mit zusätzlichen Feldern: @{ "Key1"="Value1"; "Key2"="Value2" }

.EXAMPLE
  pwsh -File .\scripts\tools\notify_discord.ps1 `
    -Title "RunAll abgeschlossen" `
    -Content "Sprint-10 erfolgreich ✅" `
    -DataJsonPath ".\output\run_summary.json"

.EXAMPLE
  pwsh -File .\scripts\tools\notify_discord.ps1 `
    -Title "Backtest" `
    -Content "Erfolgreich" `
    -Webhook "https://discord.com/api/webhooks/..." `
    -Persist
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$Title,

  [Parameter(Mandatory = $true)]
  [string]$Content,

  [Parameter(Mandatory = $false)]
  [string]$Webhook,

  [switch]$Persist,

  [Parameter(Mandatory = $false)]
  [string]$DataJsonPath,

  [Parameter(Mandatory = $false)]
  [hashtable]$ExtraFields
)

$ErrorActionPreference = 'Stop'

function Write-Info([string]$msg){
  $ts = (Get-Date).ToUniversalTime().ToString('s')+'Z'
  Write-Host "[$ts] [DISCORD] $msg"
}

function Test-WebhookFormat([string]$u){
  if([string]::IsNullOrWhiteSpace($u)){ return $false }
  return ($u -match '^https://discord\.com/api/webhooks/.+')
}

function Load-DotEnv($path){
  if(-not (Test-Path $path)){ return @{} }
  $envMap = @{}
  Get-Content -LiteralPath $path | ForEach-Object {
    $line = $_.Trim()
    if($line.Length -eq 0 -or $line.StartsWith('#')){ return }
    $kv = $line -split '=', 2
    if($kv.Count -eq 2){
      $k = $kv[0].Trim()
      $v = $kv[1].Trim()
      if($k){ $envMap[$k] = $v }
    }
  }
  return $envMap
}

function Resolve-RepoRoot {
  # Dieses Skript liegt unter scripts/tools → Repo-Root ist zwei Ebenen höher
  $root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
  return $root
}

# --- 1) Webhook ermitteln ---
$finalWebhook = $null

if(Test-WebhookFormat $Webhook){
  $finalWebhook = $Webhook
}else{
  if(Test-WebhookFormat $env:DISCORD_WEBHOOK){
    $finalWebhook = $env:DISCORD_WEBHOOK
  }else{
    # .env lesen
    $root = Resolve-RepoRoot
    $dotenvPath = Join-Path $root ".env"
    if(Test-Path $dotenvPath){
      Write-Info "Lade .env aus $dotenvPath"
      $map = Load-DotEnv $dotenvPath
      if($map.ContainsKey('DISCORD_WEBHOOK') -and (Test-WebhookFormat $map['DISCORD_WEBHOOK'])){
        $env:DISCORD_WEBHOOK = $map['DISCORD_WEBHOOK']
        $finalWebhook = $env:DISCORD_WEBHOOK
      }
    }
  }
}

if(-not (Test-WebhookFormat $finalWebhook)){
  Write-Host "[WARN] Discord Webhook ist nicht gesetzt."
  $inputWebhook = Read-Host "Bitte gib deinen Discord Webhook-URL ein"
  if(Test-WebhookFormat $inputWebhook){
    $finalWebhook = $inputWebhook
    $env:DISCORD_WEBHOOK = $inputWebhook
    Write-Info "Webhook temporär gesetzt (Session)."
  }else{
    Write-Error "Ungültiger Webhook-Link. Abbruch."
    exit 1
  }
}

# Optional dauerhaft speichern
if($Persist.IsPresent){
  try{
    [Environment]::SetEnvironmentVariable('DISCORD_WEBHOOK', $finalWebhook, 'User')
    Write-Info "Webhook dauerhaft als User-Variable DISCORD_WEBHOOK gespeichert."
    Write-Host "Tipp: Neues Terminal öffnen, damit die Änderung in neuen Sessions wirkt."
  }catch{
    Write-Warning "Konnte Webhook nicht persistent speichern: $($_.Exception.Message)"
  }
}

# --- 2) Embed-Felder aufbauen ---
$fields = New-Object System.Collections.ArrayList

# a) ExtraFields (Hashtable)
if($ExtraFields){
  foreach($k in $ExtraFields.Keys){
    $null = $fields.Add(@{ name = "$k"; value = "$($ExtraFields[$k])"; inline = $true })
  }
}

# b) DataJsonPath (JSON → Felder)
if($DataJsonPath -and (Test-Path $DataJsonPath)){
  try{
    $json = Get-Content -LiteralPath $DataJsonPath -Raw | ConvertFrom-Json
    if($json -is [System.Collections.IDictionary]){
      foreach($k in $json.Keys){
        $val = $json[$k]
        if($val -is [System.Collections.IEnumerable] -and -not ($val -is [string])){
          $val = ($val | ConvertTo-Json -Depth 4)
        }
        $null = $fields.Add(@{ name = "$k"; value = "$val"; inline = $true })
      }
    }else{
      # falls es ein Objekt/Array ist → einmal serialisiert anhängen
      $null = $fields.Add(@{ name = "data"; value = ($json | ConvertTo-Json -Depth 6); inline = $false })
    }
  }catch{
    Write-Warning "Konnte $DataJsonPath nicht als JSON lesen: $($_.Exception.Message)"
  }
}

# --- 3) Payload vorbereiten ---
$timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
$embed = @{
  title       = $Title
  description = $Content
  color       = 5814783
  footer      = @{ text = "Gesendet am $timestamp" }
}

if($fields.Count -gt 0){
  $embed['fields'] = $fields
}

$payload = @{
  username = "Assembled Trading AI"
  embeds   = @($embed)
} | ConvertTo-Json -Depth 8

# --- 4) Senden ---
try{
  Write-Info "Sende Nachricht an Discord…"
  $null = Invoke-RestMethod -Uri $finalWebhook -Method Post -ContentType "application/json" -Body $payload
  Write-Host "[OK] Nachricht erfolgreich an Discord gesendet ✅"
}catch{
  Write-Error "Fehler beim Senden an Discord: $($_.Exception.Message)"
  if($_.Exception.Response -and $_.Exception.Response.StatusCode){
    Write-Error "HTTP Status: $($_.Exception.Response.StatusCode.value__) $($_.Exception.Response.StatusDescription)"
  }
  exit 1
}
