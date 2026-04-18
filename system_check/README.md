# System-Check Tournament

Adversariales Multi-Agent-Review für das Hauptprojekt **Assembled-Trading-AI**.

## Zweck

Ein wiederverwendbarer „Service-Check", der periodisch **systematische
Blindspots, versteckte Risiken, schwache Annahmen und Zukunfts-Chancen**
offenlegt. 5 Verteidiger-Agenten verteidigen das Projekt gegen 25
Kritiker-Agenten aus fünf Domänen, ein Judge-Agent fasst zusammen und
liefert einen zweiteiligen Report:

- **Sektion A — Current Weaknesses & Findings** (was ist heute falsch / schwach)
- **Sektion B — Strategic Improvement & Expansion Recommendations** (wo kann
  das System in 6–18 Monaten hin, priorisiert nach Impact × Effort)

Dies ist ein **Meta-Tool** — es liest das Projekt, modifiziert es nicht.

## Voraussetzungen

1. `ANTHROPIC_API_KEY` in `.env` (niemals committen).
2. Installation der SDK-Dependencies:
   ```bash
   pip install -e ".[system_check]"
   ```

## CLI

```bash
# Volles Turnier (5 Defender + 25 Critics + Judge)
python scripts/run_system_check.py

# Dry-Run: Brief wird gebaut, Personas werden geladen, KEIN API-Call
python scripts/run_system_check.py --dry-run

# Reduziertes Turnier für schnellen Test / geringere Kosten
python scripts/run_system_check.py --critics 10 --defenders 3

# Alternatives Output-Verzeichnis
python scripts/run_system_check.py --output system_check/runs/
```

## Output pro Lauf

Unter `system_check/runs/{YYYYMMDD_HHMMSS}_{short_sha}/`:

| Datei | Inhalt |
|---|---|
| `brief.md` | Der Projekt-Brief, den alle Agenten gesehen haben |
| `transcript.jsonl` | Eine Zeile pro Agent-Turn (round, agent_id, role, model, tokens, content) |
| `report.md` | Kondensierter Findings- und Empfehlungs-Report (zweiteilig) |
| `recommendations.json` | Strukturierte Expansion-Vorschläge (maschinen-lesbar) |
| `scoreboard.json` | Verteidiger-Scores + Critic-Attack-Gewichte |
| `config_snapshot.yaml` | Kopie der genutzten Config |
| `manifest.json` | Run-Metadaten: git_sha, Timestamps, Tokens, Kostenschätzung |

Einzelne Run-Ordner sind gitignored, nur `.gitkeep` bleibt versioniert.

## Struktur

```
system_check/
├── config/tournament_default.yaml   # Modell-Routing, Runden, Token-Caps
├── personas/defenders.yaml          # 5 Defender-Prompts
├── personas/critics.yaml            # 25 Critic-Prompts (5 Cluster × 5)
├── runner/
│   ├── brief_builder.py             # Komprimierter Projekt-Brief aus CLAUDE.md etc.
│   ├── claude_client.py             # async Anthropic-Wrapper mit Retry
│   ├── tournament.py                # 4-Runden-Orchestrator
│   ├── judge.py                     # Synthese + Scoreboard-Parser
│   └── report.py                    # Markdown + JSONL/JSON Writer
└── tests/                           # Unit-Tests (mit gemocktem Client)
```

## Sicherheits-Garantien

- Kein Schreibzugriff auf `src/assembled_core/`, `scripts/`, `tests/` außer auf Dateien,
  die zu diesem Tool gehören.
- Kein Secret wird geloggt oder in einen Run-Ordner kopiert.
- API-Fehler führen zu einem klaren Abbruch, nie zu silent skip.

## Tokenkosten (grobe Schätzung pro Vollturnier)

- Critics (25 × Haiku 4.5, je ~3k in/400 out): ~80k in / 10k out
- Defenders (5 × Sonnet 4.6, je ~15k in/1200 out): ~75k in / 6k out
- Counter-Rebuttals (10 × Haiku 4.5): ~20k in / 5k out
- Judge (1 × Sonnet 4.6, ~30k in / 4k out): ~30k in / 4k out

**Total**: ~205k Input + ~25k Output Tokens → ca. $0.60–$1.20 pro Lauf.
Das Manifest speichert die tatsächlich verbrauchten Tokens und eine
konservative Kostenschätzung.
