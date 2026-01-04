# Cursor Workspace auf Haupt-Repo umstellen - Schritt-für-Schritt-Anleitung

## Ziel
Cursor-Workspace soll auf das Haupt-Repo zeigen: `f:\Python_Projekt\Aktiengerüst`

---

## Methode 1: Workspace in Cursor öffnen (Empfohlen)

### Schritt 1: Cursor öffnen
1. Öffne **Cursor** (die Anwendung)

### Schritt 2: Workspace öffnen
1. Klicke auf **File** (oder `Strg+K Strg+O` drücken)
2. Wähle **Open Folder...** (oder `Strg+K Strg+O`)
3. Navigiere zu: `f:\Python_Projekt\Aktiengerüst`
4. Klicke auf **Select Folder**

### Schritt 3: Workspace prüfen
1. Unten links in der Statusleiste siehst du den aktuellen Workspace-Pfad
2. Er sollte zeigen: `f:\Python_Projekt\Aktiengerüst`
3. Falls ein anderer Pfad angezeigt wird (z.B. `c:\Users\hanso\.cursor\worktrees\...`), dann ist der falsche Workspace geöffnet

---

## Methode 2: Workspace-Pfad in Cursor prüfen

### Schritt 1: Statusleiste prüfen
1. Öffne Cursor
2. Schaue **unten links** in die Statusleiste
3. Dort steht der aktuelle Workspace-Pfad

### Schritt 2: Workspace wechseln (falls falsch)
1. Klicke auf den Workspace-Pfad in der Statusleiste
2. Oder: `File` → `Open Folder...`
3. Wähle: `f:\Python_Projekt\Aktiengerüst`

---

## Methode 3: Über Command Palette

### Schritt 1: Command Palette öffnen
1. Drücke `Strg+Shift+P` (oder `F1`)
2. Tippe: `File: Open Folder...`
3. Drücke Enter

### Schritt 2: Ordner auswählen
1. Navigiere zu: `f:\Python_Projekt\Aktiengerüst`
2. Klicke auf **Select Folder**

---

## Verifizierung: Workspace ist korrekt

### Prüfung 1: Statusleiste
- Unten links sollte stehen: `f:\Python_Projekt\Aktiengerüst`

### Prüfung 2: Explorer-Panel
- Im linken Explorer-Panel sollte der Ordner `Aktiengerüst` sichtbar sein
- Darin sollten die Ordner `src/`, `scripts/`, `docs/`, `data/`, `output/` sichtbar sein

### Prüfung 3: Terminal
1. Öffne ein Terminal in Cursor (`Strg+ö` oder `Terminal` → `New Terminal`)
2. Führe aus: `Get-Location`
3. Erwartete Ausgabe: `f:\Python_Projekt\Aktiengerüst`

---

## Falls Workspace auf Worktree zeigt

### Problem erkannt
Wenn der Workspace auf `c:\Users\hanso\.cursor\worktrees\Aktienger_st\feb` zeigt:

### Lösung
1. **Schließe** den aktuellen Workspace: `File` → `Close Folder`
2. **Öffne** das Haupt-Repo: `File` → `Open Folder...` → `f:\Python_Projekt\Aktiengerüst`

---

## Schnell-Check: Ist der Workspace korrekt?

Führe diesen Check in Cursor-Terminal aus:

```powershell
Get-Location
# Sollte ausgeben: f:\Python_Projekt\Aktiengerüst

Test-Path "src/assembled_core/config.py"
# Sollte ausgeben: True
```

---

## Troubleshooting

### Problem: "Workspace kann nicht geöffnet werden"
- Prüfe, ob der Pfad `f:\Python_Projekt\Aktiengerüst` existiert
- Prüfe Berechtigungen (sollte lesbar sein)

### Problem: "Workspace zeigt immer noch auf Worktree"
- Schließe Cursor komplett
- Öffne Cursor neu
- Öffne dann `f:\Python_Projekt\Aktiengerüst` als Workspace

### Problem: "Cursor öffnet automatisch den falschen Workspace"
- Cursor merkt sich den letzten Workspace
- Öffne manuell den korrekten Workspace (siehe Methode 1)

---

## Nach erfolgreicher Umstellung

Nachdem der Workspace korrekt ist, sollte:
- ✅ Der Fehler "Failed to read file" nicht mehr auftreten
- ✅ Alle Dateien in `src/assembled_core/` lesbar sein
- ✅ Die Scripts funktionieren


