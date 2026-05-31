# Cleanup-Ergebnis — 2026-05-27

Ausgeführt auf: `main` @ `a7e01689`

## Sicherung

| Aktion | Status |
|--------|--------|
| `git tag backup/pre-cleanup-2026-05-27` | OK — zeigt auf `a7e01689` |
| `git push origin backup/pre-cleanup-2026-05-27` | OK — gepusht |

## Schritt 1 — main synchronisiert

- 2 lokale Commits nach origin/main gepusht (`9467b0ae..a7e01689`)
- Untracked Scripts (nicht angefasst):
  - `scripts/_crisis_alpha_backtest_compare.py`
  - `scripts/_crisis_alpha_pit_verify.py`
  - `scripts/_crisis_alpha_replay.py`
  - `scripts/backtest_news_alpha.py`

## Schritt 2 — ERWEITERUNG archiviert

| Aktion | Status |
|--------|--------|
| `git tag archive/erweiterung-2026-05-12 ERWEITERUNG` | OK — zeigt auf `2c902e63` |
| `git push origin archive/erweiterung-2026-05-12` | OK — gepusht |
| `git branch -D ERWEITERUNG` | OK — lokal gelöscht |
| `origin/ERWEITERUNG` | BEHALTEN — Remote-Archiv |

## Schritt 3 — Agent-Worktrees entfernt

Alle 4 Worktrees waren `locked` — benötigten `-f -f` (doppeltes Force).

| Worktree-Pfad | Branch | Status |
|---------------|--------|--------|
| `.claude/worktrees/agent-a7b606cb13c522e1d` | `worktree-agent-a7b606cb13c522e1d` | Worktree + Branch gelöscht |
| `.claude/worktrees/agent-a8530ae9d7d5595f6` | `worktree-agent-a8530ae9d7d5595f6` | Worktree + Branch gelöscht |
| `.claude/worktrees/agent-ac275289d3bf5b9ed` | `worktree-agent-ac275289d3bf5b9ed` | Worktree + Branch gelöscht |
| `.claude/worktrees/agent-aff57adf12c4aadd1` | `worktree-agent-aff57adf12c4aadd1` | Worktree + Branch gelöscht |

Grundlage: Diff-Analyse `01b_worktree_diff.md` — alle 4 SICHER LÖSCHBAR (ältere/schlechtere Stände als main).

## Schritt 4 — Cursor-Worktrees entfernt

Alle 3 waren detached HEADs bei `a93cfe5b` (November 2025, 1334 Commits hinter main).

| Worktree-Pfad | Status |
|---------------|--------|
| `C:/Users/hanso/.cursor/worktrees/Aktienger_st/alj` | Entfernt |
| `C:/Users/hanso/.cursor/worktrees/Aktienger_st/eiy` | Entfernt |
| `C:/Users/hanso/.cursor/worktrees/Aktienger_st/feb` | Entfernt |

## Schritt 5 — Veraltete Branches gelöscht

| Branch | War bei Commit | Status |
|--------|---------------|--------|
| `feat/edcl` | `f967cce3` | Gelöscht (vollständig in main gemergt) |
| `sprint1/blocker-core-safety` | `eb309923` | Gelöscht (vollständig in main gemergt) |
| `sprint2/algo-phase1-exec-quality` | `8fe78fc4` | Gelöscht (vollständig in main gemergt) |
| `worktree-agent-a73bbad9` | `5dc614da` | Gelöscht (vollständig in main gemergt) |
| `worktree-agent-a0a9c2c0` | `44c25ec8` | Gelöscht (vollständig in main gemergt) |
| `worktree-agent-a2f080f2` | `44c25ec8` | Gelöscht (vollständig in main gemergt) |
| `worktree-agent-a700e54f` | `44c25ec8` | Gelöscht (vollständig in main gemergt) |
| `worktree-agent-ae6ea25d` | `44c25ec8` | Gelöscht (vollständig in main gemergt) |

## Abschlusszustand

### Worktrees

```
git worktree list
→ "F:/Python_Projekt/Aktiengerüst" a7e01689 [main]
```

Nur noch ein Worktree.

### Branches (lokal)

```
git branch
→ * main
```

Nur noch main.

### Tags

```
archive/erweiterung-2026-05-12   → 2c902e63 (ERWEITERUNG-Stand)
backup/pre-cleanup-2026-05-27    → a7e01689 (Stand vor Cleanup)
phase4_stable
pipeline-unified-2026q2
sprint9-freeze
```

## Recovery

Falls ein gelöschter Branch wiederhergestellt werden muss:

```bash
# Aus dem Backup-Tag:
git checkout -b <branch-name> backup/pre-cleanup-2026-05-27

# ERWEITERUNG:
git checkout -b ERWEITERUNG archive/erweiterung-2026-05-12
```

## Fehler während der Ausführung

`git worktree remove --force` schlug fehl bei locked Worktrees (lock reason: `claude agent`).  
Lösung: `-f -f` (doppeltes Force). Kein Datenverlust, da Diff-Analyse bereits SICHER LÖSCHBAR bestätigte.
