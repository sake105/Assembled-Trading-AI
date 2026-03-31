# 20 Security and Secrets

## Zweck

Diese Regeln schützen Secrets, Credentials, sensible Betriebsdaten und sicherheitskritische Repo-Aktionen.

## Harte Verbote

- Lies niemals `.env`, `.env.*`, `secrets/**` oder andere Credential-Dateien, wenn es nicht ausdrücklich für Secret-Migration oder Hardening beauftragt wurde.
- Gib niemals API-Keys, Tokens, Passwörter, private URLs oder ähnliche Inhalte in Chat, Commit-Texten oder Logs aus.
- Kopiere keine Secrets in Tests, Fixtures oder Beispieldateien.
- Erzeuge keine neuen geheimen Dateien im Repo, wenn sie committed werden könnten.
- Verändere keine CI- oder Deployment-Sicherheit stillschweigend.

## Sensible Pfade

Besonders vorsichtig behandeln:

- `.env`
- `.env.*`
- `.github/workflows/**`
- `config/**` sofern Credentials, Endpunkte oder produktionsnahe Settings enthalten sind
- mögliche Broker-, API-, Accounting- oder OMS-nahe Konfigurationen

## Standardverhalten bei Secret-Risiko

Wenn Secret-Risiko erkannt wird:

1. Sofort transparent benennen.
2. Keine Geheimdaten ausgeben.
3. Stattdessen Pfad, Risikoart und empfohlene Gegenmaßnahme nennen.
4. Nur dann ändern, wenn der Auftrag ausdrücklich Security/Hardening betrifft.

## CI-Sicherheitsregeln

- Änderungen an `.github/workflows/**` sind hochsensibel.
- Workflow-Änderungen nie „nebenbei“ mitnehmen.
- Bei Workflow-Änderungen immer separat erklären:
  - warum die Änderung nötig ist
  - welches Verhalten sich ändert
  - welche Jobs oder Plattformen betroffen sind
  - welches Risiko für bestehende CI besteht

## Shell- und Git-Schutz

Folgende Operationen gelten als gefährlich und dürfen nicht automatisch empfohlen oder ausgelöst werden, außer der Nutzer verlangt es ausdrücklich:

- `git reset --hard`
- `git clean -fd`
- ungezielte Löschbefehle
- rekursive Massenänderungen ohne Pfadbegrenzung
- Befehle, die Secrets ausgeben könnten

## Log-Regel

Logs, Konsole und Reports dürfen keine Secrets enthalten.
Wenn unklar ist, ob ein Wert sensibel ist, ist er als sensibel zu behandeln.

## Incident-Regel: Bereits committed Secrets

Diese Regel gilt, wenn Hinweise bestehen, dass Secrets bereits im Repo versioniert sind oder waren —
zum Beispiel eine `.env`-Datei mit echten Keys, die committed wurde.

### Harte Verbote in diesem Fall

- Keinen Secret-Inhalt lesen, wiederholen, zitieren oder in einer Antwort reproduzieren — auch nicht
  zum Zweck der Verifikation oder als Diff-Kontext.
- Nicht behaupten, das Problem sei gelöst, nur weil:
  - die Datei lokal gelöscht wurde
  - die Datei jetzt in `.gitignore` steht
  - der aktuelle Working Tree „sauber" aussieht
- Keine beruhigende Formulierung verwenden, wenn das Risiko noch nicht vollständig beseitigt ist.

### Pflichtverhalten bei erkanntem oder vermutetem Secret-Leak

Claude muss in dieser Reihenfolge reagieren:

1. **Rotationsbedarf benennen.**
   Jeder im Git-Verlauf exponierte Key ist als kompromittiert zu behandeln.
   Rotation beim jeweiligen Provider ist zwingend, unabhängig davon, ob der Key noch aktiv genutzt wird.

2. **Reichweite des Problems klarstellen.**
   Ein gelöschter Commit oder eine `.gitignore`-Ergänzung entfernt den Key nicht aus der Git-History.
   Wer die History klont oder durchsucht, kann den Key weiterhin lesen.
   Das Risiko erstreckt sich auf alle Clones, Forks und CI-Artefakte, die den betroffenen Stand jemals gesehen haben.

3. **`.gitignore`-Status prüfen und benennen.**
   Prüfen, ob die betroffene Datei (z. B. `.env`) jetzt korrekt in `.gitignore` eingetragen ist.
   Falls nicht: darauf hinweisen. Falls ja: explizit sagen, dass das nur künftige Commits schützt,
   nicht die bestehende History.

4. **History-Bereinigung als offene Frage benennen.**
   Ob die History bereinigt werden soll (z. B. via `git filter-repo`), ist eine Projektentscheidung.
   Claude benennt die Option, empfiehlt sie nicht automatisch und führt sie nicht eigenständig aus.
   History-Rewrite ist destruktiv und erfordert expliziten Auftrag.

5. **Nächsten konkreten Schritt nennen.**
   Immer mindestens einen verifizierbaren nächsten Schritt benennen:
   Key rotieren, `.gitignore` prüfen oder CI-Secrets-Scanning einrichten.

### Formulierungsregel

Claude darf nicht formulieren:

- „Das Problem ist behoben" — wenn nur die Arbeitskopie bereinigt wurde.
- „Der Key ist nicht mehr zugänglich" — wenn er in der History verbleibt.
- „Das war vermutlich kein echtes Risiko" — ohne konkrete Evidenz dafür.

Claude soll stattdessen formulieren:

- „Der Key ist in der Git-History exponiert und muss rotiert werden."
- „`.gitignore` verhindert künftige Commits, schützt aber nicht die bestehende History."
- „Solange der Key nicht rotiert ist, ist das Risiko nicht beseitigt."
