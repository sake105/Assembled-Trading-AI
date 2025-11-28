# Security & Secrets Management

## Übersicht

Dieses Dokument beschreibt den Umgang mit Secrets (API-Keys, Passwörtern, Tokens) im Assembled Trading AI Projekt.

**Grundregel:** Niemals Secrets in Git-Repository committen oder in versionierten Dateien speichern.

---

## Arten von Secrets

### Datenanbieter-API-Keys

- **`ALPHAVANTAGE_API_KEY`**: Alpha Vantage API-Key für Intraday/EOD-Daten
- **`FINNHUB_API_KEY`**: Finnhub API-Key (geplant) für Fundamentaldaten
- **`NEWSAPI_KEY`**: NewsAPI-Key (geplant) für News-Feeds

### Broker-API-Keys (Zukunft)

- **`BROKER_API_KEY`**: Broker-API-Key für Order-Execution (aktuell nicht verwendet, SAFE-Bridge)
- **`BROKER_SECRET`**: Broker-API-Secret (aktuell nicht verwendet)

### E-Mail/SMTP (Zukunft)

- **`EMAIL_SMTP_HOST`**: SMTP-Server
- **`EMAIL_SMTP_USER`**: SMTP-Benutzername
- **`EMAIL_SMTP_PASSWORD`**: SMTP-Passwort

### Datenbank (Zukunft)

- **`DB_CONNECTION_STRING`**: Datenbank-Verbindungsstring (falls später DB verwendet wird)

---

## Grundregeln

### ✅ Erlaubt

- **Umgebungsvariablen**: Secrets als Umgebungsvariablen setzen
  ```powershell
  # PowerShell
  $env:ALPHAVANTAGE_API_KEY = "your-key-here"
  ```
  ```bash
  # Bash
  export ALPHAVANTAGE_API_KEY="your-key-here"
  ```

- **`.env`-Dateien (lokal)**: `.env`-Dateien nur lokal, nie in Git
  ```bash
  # .env (lokal, nicht in Git)
  ALPHAVANTAGE_API_KEY=your-key-here
  FINNHUB_API_KEY=your-key-here
  ```

- **Konfigurationsdateien mit Platzhaltern**: `config/datasource.psd1` nutzt `$env:VARIABLE_NAME`
  ```powershell
  # config/datasource.psd1 (versioniert, aber ohne echte Keys)
  AlphaVantage = @{ ApiKey = "$env:ALPHAVANTAGE_API_KEY" }
  ```

- **Secret-Manager**: Externe Secret-Manager (z. B. Azure Key Vault, AWS Secrets Manager) für Produktion

### ❌ Verboten

- **Hardcodierte Keys im Code**: Niemals API-Keys direkt im Python/PowerShell-Code
  ```python
  # ❌ FALSCH
  api_key = "sk-1234567890abcdef"
  ```

- **Secrets in Git**: Niemals `.env`-Dateien oder Dateien mit echten Keys committen
  ```bash
  # ❌ FALSCH
  git add .env
  git commit -m "Add API keys"
  ```

- **Secrets in Konfigurationsdateien**: Keine echten Keys in versionierten Config-Dateien
  ```powershell
  # ❌ FALSCH (in config/datasource.psd1)
  AlphaVantage = @{ ApiKey = "sk-1234567890abcdef" }
  ```

- **Secrets in Logs**: Keine API-Keys in Log-Ausgaben
  ```python
  # ❌ FALSCH
  print(f"API Key: {api_key}")
  ```

---

## .env Datei Struktur

**Beispiel `.env` (lokal, nicht in Git):**

```bash
# Datenanbieter
ALPHAVANTAGE_API_KEY=your-alpha-vantage-key-here
FINNHUB_API_KEY=your-finnhub-key-here
NEWSAPI_KEY=your-newsapi-key-here

# Broker (Zukunft)
# BROKER_API_KEY=your-broker-key-here
# BROKER_SECRET=your-broker-secret-here

# E-Mail/SMTP (Zukunft)
# EMAIL_SMTP_HOST=smtp.example.com
# EMAIL_SMTP_USER=your-email@example.com
# EMAIL_SMTP_PASSWORD=your-password-here

# Datenbank (Zukunft)
# DB_CONNECTION_STRING=postgresql://user:password@host:port/dbname
```

**Wichtig:**
- Diese Datei existiert nur lokal
- Nie in Git committen
- `.gitignore` ignoriert `.env*` Dateien automatisch

---

## Verwendung in Code

### PowerShell

```powershell
# Umgebungsvariable setzen
$env:ALPHAVANTAGE_API_KEY = "your-key"

# In Script verwenden
$apiKey = $env:ALPHAVANTAGE_API_KEY
```

### Python

```python
import os

# Aus Umgebungsvariable lesen
api_key = os.getenv("ALPHAVANTAGE_API_KEY")
if not api_key:
    raise ValueError("ALPHAVANTAGE_API_KEY environment variable not set")

# Oder mit python-dotenv (falls .env verwendet wird)
from dotenv import load_dotenv
load_dotenv()  # Lädt .env Datei (nur lokal)
api_key = os.getenv("ALPHAVANTAGE_API_KEY")
```

### Konfigurationsdateien

**`config/datasource.psd1`** (versioniert, ohne echte Keys):
```powershell
@{
    AlphaVantage = @{ ApiKey = "$env:ALPHAVANTAGE_API_KEY" }
    Finnhub = @{ ApiKey = "$env:FINNHUB_API_KEY" }
    NewsApi = @{ ApiKey = "$env:NEWSAPI_KEY" }
}
```

---

## Key-Rotation nach Leak

**Wenn ein Secret geleakt wurde:**

1. **Sofort rotieren**: Neuen Key beim Anbieter generieren
2. **Alten Key deaktivieren**: Alten Key beim Anbieter deaktivieren
3. **Umgebungsvariablen aktualisieren**: Neue Keys in `.env` oder Umgebungsvariablen setzen
4. **Git-Historie prüfen**: Falls Key versehentlich committed wurde, Git-Historie bereinigen (siehe unten)

**Git-Historie bereinigen (falls nötig):**
```bash
# WICHTIG: Nur wenn Key versehentlich committed wurde
# Benötigt: git filter-branch oder BFG Repo-Cleaner
# → Siehe Git-Dokumentation für Details
```

---

## Best Practices

1. **Minimale Berechtigungen**: API-Keys nur mit minimalen notwendigen Berechtigungen erstellen
2. **Key-Namen dokumentieren**: Welche Keys benötigt werden, ist in `config/datasource.psd1` dokumentiert
3. **Separate Keys für Dev/Prod**: Unterschiedliche Keys für Entwicklung und Produktion
4. **Regelmäßige Rotation**: Keys regelmäßig rotieren (z. B. alle 90 Tage)
5. **Monitoring**: API-Key-Nutzung überwachen, ungewöhnliche Aktivitäten erkennen

---

## Verweise

- **Backend-Doku**: Siehe `docs/backend_core.md` für Details zur Konfiguration
- **Cursor-Regeln**: `.cursor/rules/02-backend-guidelines.md` enthält Secrets-Guidelines
- **Data Sources**: `docs/DATA_SOURCES_BACKEND.md` beschreibt API-Key-Verwendung für Datenanbieter

---

## Checkliste für neue Secrets

- [ ] Secret als Umgebungsvariable definiert (nicht hardcodiert)
- [ ] `.env`-Datei lokal erstellt (nicht in Git)
- [ ] `.gitignore` prüft `.env*` Patterns
- [ ] Konfigurationsdateien nutzen `$env:VARIABLE_NAME` Platzhalter
- [ ] Dokumentation aktualisiert (Key-Name dokumentiert, kein echter Wert)
- [ ] Code prüft auf fehlende Umgebungsvariablen (gibt klare Fehlermeldung)

