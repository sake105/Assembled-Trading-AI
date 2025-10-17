param(
  [Parameter(Mandatory=$true)][string]$Title,
  [Parameter(Mandatory=$true)][string]$Content,
  [string]$Webhook = $env:DISCORD_WEBHOOK,
  [string[]]$FilePaths = @()
)

if ([string]::IsNullOrWhiteSpace($Webhook)) {
  throw "DISCORD_WEBHOOK ist leer. Setze ihn z.B.: [Environment]::SetEnvironmentVariable('DISCORD_WEBHOOK','<dein_webhook>','User')"
}

function Split-DiscordChunks([string]$text, [int]$max=1800){
  $lines = $text -split "`r?`n"
  $chunks = @()
  $current = ""
  foreach($l in $lines){
    if(($current.Length + $l.Length + 1) -gt $max){
      if($current.Length -gt 0){ $chunks += $current; $current = "" }
      while($l.Length -gt $max){
        $chunks += $l.Substring(0, $max)
        $l = $l.Substring($max)
      }
      $current = $l
    } else {
      if($current.Length -gt 0){ $current += "`n" }
      $current += $l
    }
  }
  if($current.Length -gt 0){ $chunks += $current }
  return ,$chunks
}

try {
  $header = "**$Title**"
  $chunks = Split-DiscordChunks $Content
  if($chunks.Count -eq 0){ $chunks = @("_(leer)_") }

  # 1) Erste Nachricht inkl. Header + optionaler erster Anhang
  $firstPayload = @{
    content = $header + "`n" + $chunks[0]
  }

  if($FilePaths.Count -gt 0){
    # multipart/form-data für Attachments
    $form = @{}
    $form["payload_json"] = ($firstPayload | ConvertTo-Json -Depth 5)
    $i = 0
    foreach($fp in $FilePaths){
      if(Test-Path $fp){
        $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $fp))
        $fn = [System.IO.Path]::GetFileName($fp)
        $form["files[$i]"] = New-Object System.IO.MemoryStream(,$bytes)
        $form["files[$i]"].Position = 0
        $form["filename[$i]"] = $fn
        $i++
      }
    }
    Invoke-RestMethod -Method POST -Uri $Webhook -Form $form | Out-Null
    Start-Sleep -Milliseconds 350
  } else {
    Invoke-RestMethod -Method POST -Uri $Webhook -ContentType 'application/json' -Body ($firstPayload | ConvertTo-Json -Depth 5) | Out-Null
    Start-Sleep -Milliseconds 350
  }

  # 2) Restliche Texte ohne Dateien
  for($i=1;$i -lt $chunks.Count;$i++){
    $payload = @{ content = $chunks[$i] }
    Invoke-RestMethod -Method POST -Uri $Webhook -ContentType 'application/json' -Body ($payload | ConvertTo-Json -Depth 5) | Out-Null
    Start-Sleep -Milliseconds 350
  }

  Write-Host "[DISCORD] OK (messages=$($chunks.Count), files=$($FilePaths.Count))"
} catch {
  Write-Error $_.Exception.Message
  exit 1
}
