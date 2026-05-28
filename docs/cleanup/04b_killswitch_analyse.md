# Paket 4b — Kill-Switch-Zugriffskontrolle

**Datum:** 2026-05-28  
**Branch:** main  
**GO_LIVE_CHECKLIST:** C2

---

## Schritt 1 — Ist-Analyse

### kill_switch.py (499 Zeilen, rein funktionsbasiert)

| Funktion | Signatur | Auth |
|----------|----------|------|
| `activate_kill_switch` | `(*, throttle_pct, reason, actor)` | **Keine** — Low-Threshold, kein Barrier |
| `deactivate_kill_switch` | `(*, reason, actor)` | **Keine** — Problem |
| `guard_orders_with_kill_switch` | `(orders)` | — |
| `is_kill_switch_engaged` | `()` | — |
| `get_kill_switch_state` | `()` | — |
| `check_drawdown_kill_switch` | `(current, peak, ...)` | — |

**Audit-Infrastruktur (bereits vorhanden):**
- `_append_audit(event)` — SHA-256-Hash-Chain, JSONL-Append, fsync
- `verify_audit_chain()` — prüft Kettenintegrität
- Audit-Log: `output/ops/kill_switch_audit.jsonl`
- Feld `action` bereits vorhanden: `ACTIVATE`, `DEACTIVATE`, `GUARD`
- `actor`-Feld bereits in allen Einträgen

### auth.py — bestehendes API-Auth-Muster

- `require_api_key()` FastAPI-Dependency: `hmac.compare_digest(x_api_key, expected)`
- Env-Var: `ASSEMBLED_API_KEY`
- **Fail-OPEN** wenn env var nicht gesetzt (dev-freundlich für allgemeine API)
- **Nicht wiederverwendbar** für OPERATOR_KILL_TOKEN — anderes Sicherheitsniveau erforderlich

### app.py — bestehende Endpoints

```
POST /api/v1/kill-switch/activate    → protected via require_api_key (ASSEMBLED_API_KEY)
POST /api/v1/kill-switch/deactivate  → protected via require_api_key (ASSEMBLED_API_KEY)
```

**Problem:** `ASSEMBLED_API_KEY` ist ein allgemeiner API-Key. Der deactivate-Endpoint hat
keinen zweiten Schutzlayer für die spezifisch gefährliche Richtung.

### Callee-Stellen von `deactivate_kill_switch()`

**System-Stellen (müssen token-fähig gemacht werden):**

| Datei | Zeile | Kontext |
|-------|-------|---------|
| `src/assembled_core/pipeline/trading_cycle_v2.py` | 774 | Backtest-Guard: state restore nach Backtest |
| `scripts/drills/drill_kill_switch.py` | 56, 73 | Weekly drill cleanup + verification |
| `scripts/run_preflight_checks.py` | 137 | Preflight-Check vor Start |
| `scripts/smoke_test_paper.py` | 161, 180 | Pre-test cleanup + post-test restore |
| `src/assembled_core/api/app.py` | 143 | REST-API-Endpoint |

**Test-Stellen (werden mitaktualisiert, damit CI grün bleibt):**

| Datei | Zeilen | Methode |
|-------|--------|---------|
| `tests/test_audit_additions.py` | 298, 320, 648 | monkeypatch.setenv |
| `tests/test_property_fsm_pit.py` | 141 | via isolated_kill_switch-Fixture |
| `tests/chaos/test_broker_api_500.py` | 34, 39, 49 | autouse-Fixture + test body |

---

## Schritt 2 — Design

### Invarianten

1. `activate_kill_switch()` — **KEINE Änderung**: Low-Threshold, kein Barrier für Notfall-Stop.
2. `deactivate_kill_switch()` — Neue Pflicht: OPERATOR_KILL_TOKEN-Check vor Ausführung.
3. Fail-closed: Wenn `OPERATOR_KILL_TOKEN` nicht in ENV → Deaktivierung VERWEIGERT.
4. Timing-Schutz: `hmac.compare_digest` verhindert Timing-Side-Channel.
5. Audit: Alle Versuche (rejected + accepted) → Eintrag im Hash-Chain-Log.

### Neue Funktion: deactivate_kill_switch

```python
def deactivate_kill_switch(
    *,
    reason: str = "",
    actor: str = "system",
    operator_token: str | None = None,
) -> None:
```

**Auth-Logik (am Funktionsanfang, vor jeder Zustandsänderung):**

```
_expected = os.environ.get("OPERATOR_KILL_TOKEN", "")
if not _expected:
    → _append_audit({action: REJECT_DEACTIVATE, reason: "env not set", actor: actor})
    → logger.warning(...)
    → raise PermissionError("OPERATOR_KILL_TOKEN must be set")
if not hmac.compare_digest(operator_token or "", _expected):
    → _append_audit({action: REJECT_DEACTIVATE, reason: "invalid token", actor: actor})
    → logger.warning(...)
    → raise PermissionError("invalid operator token")
# → Ausführung wie bisher
```

### app.py — Deactivate-Endpoint

Zwei-Layer-Auth:
1. Layer 1: `Depends(require_api_key)` — Authentifizierung (wer darf die API nutzen)
2. Layer 2: `X-Operator-Token` Header — Autorisierung (wer darf deaktivieren)

```python
def deactivate_kill_switch_endpoint(
    reason: str = "",
    actor: str = "api",
    x_operator_token: str | None = Header(default=None, alias="X-Operator-Token"),
    _auth: None = Depends(require_api_key),
):
    deactivate_kill_switch(reason=reason, actor=actor, operator_token=x_operator_token)
```

### System-Call-Sites

Alle 4 System-Stellen: `operator_token=os.environ.get("OPERATOR_KILL_TOKEN")`

Für Skripte, die `os` noch nicht importieren (`drill_kill_switch.py`, `smoke_test_paper.py`):
`import os` am Anfang hinzufügen.

### Tests (5 Pflicht-Cases)

| # | Case | Erwartet |
|---|------|----------|
| 1 | deactivate ohne token (ENV gesetzt) | `PermissionError`, still engaged |
| 2 | deactivate mit falschem token | `PermissionError`, still engaged |
| 3 | deactivate mit korrektem token | Erfolg, not engaged |
| 4 | activate ohne token | Erfolg (keine Auth erforderlich) |
| 5 | ENV nicht gesetzt → fail-closed | `PermissionError` mit OPERATOR_KILL_TOKEN-Hinweis |
| 6 | rejected attempt → REJECT_DEACTIVATE im Audit-Log | Eintrag vorhanden, Chain ok |

---

## Schritt 3 — Implementierung (Ergebnis)

### Geänderte Dateien

| Datei | Änderung |
|-------|----------|
| `src/assembled_core/execution/kill_switch.py` | `import hmac`, neues `operator_token`-Param, Auth-Block |
| `src/assembled_core/api/app.py` | `Header` import, `X-Operator-Token`-Header-Param, token weitergeleitet |
| `src/assembled_core/pipeline/trading_cycle_v2.py` | `operator_token=os.environ.get(...)` |
| `scripts/drills/drill_kill_switch.py` | `import os`, token an beide Aufrufe |
| `scripts/run_preflight_checks.py` | token an Aufruf |
| `scripts/smoke_test_paper.py` | `import os`, token an beide Aufrufe |
| `tests/test_audit_additions.py` | 3x `monkeypatch.setenv("OPERATOR_KILL_TOKEN", ...)` + token |
| `tests/test_property_fsm_pit.py` | token in `isolated_kill_switch`-Fixture + test body |
| `tests/chaos/test_broker_api_500.py` | monkeypatch in autouse-Fixture + token |
| `tests/test_kill_switch_auth.py` | NEU — 6 Auth-Testfälle |

### Nicht geändert (außer Scope)

- `activate_kill_switch()` — keine Auth, absichtlich Low-Threshold
- `docs/specs/kill_switch.md` — Follow-up-Update empfohlen
- `ASSEMBLED_API_KEY`-Mechanismus in `auth.py` — bleibt für allgemeine API-Auth
- `kill_switch.md` API-Spec — Follow-up empfohlen

---

## Schritt 4 — Test-Ergebnis

**Neue Auth-Tests:** `tests/test_kill_switch_auth.py` — 6/6 passed (0.40s)

**Aktualisierte bestehende Tests:**
- `tests/test_audit_additions.py` — 49/49 passed
- `tests/test_property_fsm_pit.py` — 8/8 passed
- `tests/chaos/test_broker_api_500.py` — 4/4 passed

**Gesamt betroffene Test-Suite:** 61/61 passed (6.84s)

**Breite Regressionsprüfung:** Fast-Suite exit code 0 (keine neuen Fehler)

**Lint:** `ruff check` — All checks passed auf allen geänderten Dateien

---

_Dieses Dokument wurde während der Implementierung von Paket 4b erstellt._
