# 43 — Backtesting-Reproduzierbarkeit-Zertifikat

**Zweck:** Kannst du einen Backtest, den du heute laufen lässt, in 6 Monaten **bit-genau identisch** reproduzieren? Wenn nein, ist jede Performance-Zahl aus dem Backtest mit einem Fragezeichen versehen. Das Zertifikat ist ein systematischer Mechanismus, der dir für jeden Backtest-Lauf einen **Hash-basierten Fingerabdruck** liefert und Tage später die identische Reproduktion garantiert.

**Scope:** Rang 10 aus der Gap-Analyse. Letzte Datei. Verbindet `35_GOLDEN_EQUITY_SCENARIO_TESTS.md` (Characterization-Tests), `39_HYPERPARAMETER_GOVERNANCE.md` (MLflow/Config-Versionierung) und `42_EVENT_REPLAY_SYSTEM.md` (Event-Sourcing).

**Kern-Idee:** Jedes Backtest-Artefakt (Signal, Equity-Curve, Trade-Liste) bekommt einen SHA-256-Hash. Das "Zertifikat" ist ein JSON, das alle Inputs (Code, Daten, Config, Seeds, Umgebung) identifiziert und die erwarteten Output-Hashes dokumentiert. In N Tagen kannst du das Zertifikat laden, den Backtest erneut laufen lassen, und die Hash-Gleichheit verifizieren.

---

## 0. Warum das wichtig ist

### Die langsame Erosion deiner Backtest-Ergebnisse

Heute machst du einen Backtest. Sharpe 1.8, Max-Drawdown -12 %, 247 Trades. Du bist zufrieden, deployst.

Sechs Monate später willst du einen neuen Feature einbauen. Du willst den Backtest "wiederholen" — einmal ohne das neue Feature als Baseline, einmal mit Feature. Du lädst denselben Code-Branch, denselben Datensatz, dieselbe Config. Sharpe kommt: **1.92**. Max-Drawdown: **-11.2 %**. 

Was ist passiert?

Mögliche Ursachen:
1. **pandas-Version hat sich geändert** — eine obskure Timeseries-Operation produziert minimal andere Werte
2. **yfinance hat Daten revidiert** — 2 Dividenden-Adjustments waren falsch, sind jetzt korrigiert
3. **Python-Version-Update** — floating-point auf neuer CPU oder neuer Compiler-Version produziert 1e-15 Unterschiede, die über 1000 Trades kumulieren
4. **Deine eigene Code-Änderung** — du hast vor 3 Monaten einen "harmlos aussehenden" Refactor gemacht
5. **Seed ist weg** — du hast vergessen, den Random-State deterministisch zu initialisieren

**Ohne Reproducibility-Zertifikat kannst du die Frage nicht beantworten.** Der Sharpe-Unterschied kann signifikant sein (Feature-Wert) oder Rauschen (Umgebungs-Drift). Du weißt es nicht.

### Das Zertifikat als Safety-Net

Mit Zertifikat:
1. Heute: Backtest + Zertifikat generieren
2. In 6 Monaten: dieselbe Umgebung via Docker/uv-lock rekonstruieren → neuer Lauf → Hash-Vergleich
3. **Hash identisch** → du kannst das neue Feature eindeutig gegen den alten Baseline testen
4. **Hash unterschiedlich** → du weißt: Environment-Drift. Suche die Ursache, bevor du weiterarbeitest.

Das ist der Unterschied zwischen "ich glaube, mein Feature verbessert die Strategie" und "ich **weiß**, mein Feature bringt +0.12 Sharpe unter identischer Baseline".

### Die Forschungs-Integrität-Perspektive

Wenn du jemals deine Strategie einem Business-Partner, Investor oder späteren dir selbst zeigen willst, musst du beweisen können: **diese Zahlen sind kein Zufall**. Ohne Zertifikat hast du Screenshots. Mit Zertifikat hast du ein digital signiertes Dokument, das sagt: "Diese exakten Inputs produzieren diese exakten Outputs, verifiziert durch diesen Hash."

---

## 1. Was alles muss fixiert sein

Für bit-genaue Reproduzierbarkeit müssen **alle** diese Quellen von Non-Determinismus eliminiert sein:

### 1.1 Inputs

| Input | Fix-Mechanismus |
|---|---|
| Code-Stand | Git-SHA des Commits |
| Strategy-Config | Hash der Config-YAML (aus `39_HYPERPARAMETER_GOVERNANCE.md`) |
| ML-Modelle | MLflow-Model-Version + Hash der `.pkl`-Datei |
| Historische Daten | Hash des Parquet-Files |
| Event-Stream (bei Replay) | Session-ID + Sequence-Hash |

### 1.2 Umgebung

| Component | Fix-Mechanismus |
|---|---|
| Python-Version | pyproject.toml `requires-python = "==3.11.9"` |
| Pakete | uv.lock mit vollständigen Hashes |
| System-Libraries | Docker-Image-Hash |
| CPU-Architektur | Docker-Platform-Tag (z.B. `linux/amd64`) |
| OS-Zeit-Zone | ENV `TZ=UTC` |
| Locale | ENV `LC_ALL=C.UTF-8` |

### 1.3 Random-State

| Library | Seeding |
|---|---|
| Python `random` | `random.seed(42)` |
| NumPy | `np.random.default_rng(seed=42)` |
| scikit-learn | `random_state=42` in allen Splits + Models |
| pandas | kein globaler Seed, aber `DataFrame.sample(random_state=42)` |
| PyTorch (falls verwendet) | `torch.manual_seed(42)` + `torch.use_deterministic_algorithms(True)` |

### 1.4 Spezielle Fallen

**Floating-Point-Summations-Order:**
`sum([0.1, 0.2, 0.3])` auf Intel vs. AMD kann minimal andere Werte produzieren. Lösung: **sorted input** bei Summations, wo Reihenfolge egal sein sollte.

**Dict-Iteration (kein Problem seit 3.7, aber Sets!):**
```python
# PROBLEM:
for ticker in {"AAPL", "MSFT", "NVDA"}:  # Iteration-Order undefined
    ...

# LÖSUNG:
for ticker in sorted({"AAPL", "MSFT", "NVDA"}):
    ...
```

**Parallel-Processing:**
`multiprocessing.Pool.map` gibt Ergebnisse zurück in Aufruf-Reihenfolge, aber wenn die Worker in andere Reihenfolge fertig werden und der Haupt-Thread sie in Fertig-Reihenfolge verarbeitet → Non-Determinismus. Lösung: **synchrone Verarbeitung** oder **explizite Sortierung** am Ende.

**Thread-Local State:**
In Libraries wie `joblib` oder `numba` kann thread-local state andere Ergebnisse produzieren je nach thread-count. Lösung: `n_jobs=1` für reproducibility-kritische Pfade.

---

## 2. Die Zertifikats-Datenstruktur

### 2.1 Das Schema

```python
# src/assembled_core/certify/schema.py
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class EnvironmentFingerprint(BaseModel):
    """Umgebungs-Identitäts-Hash."""
    python_version: str                    # z.B. "3.11.9"
    platform: str                          # "Linux-5.15-x86_64"
    cpu_model: str                         # aus /proc/cpuinfo
    docker_image_sha: Optional[str] = None # falls in Docker gelaufen
    uv_lock_hash: str                      # SHA-256 der uv.lock
    system_libraries: dict[str, str]       # z.B. {"glibc": "2.35"}


class InputFingerprint(BaseModel):
    """Inputs-Identitäts-Hash."""
    git_sha: str                           # HEAD
    git_dirty: bool                        # ob uncommitted changes
    strategy_config_hash: str
    strategy_config_path: str
    data_file_hash: str
    data_file_path: str
    data_row_count: int                    # Sanity-Check
    ml_model_refs: dict[str, str]          # model_name → MLflow URI
    ml_model_hashes: dict[str, str]        # model_name → SHA-256 of .pkl
    seed: int                              # 42 (oder was immer verwendet wird)


class OutputFingerprint(BaseModel):
    """Outputs-Identitäts-Hash."""
    equity_curve_hash: str                 # SHA-256 der Equity-Curve-Series
    trade_list_hash: str                   # SHA-256 der Trade-Liste
    signal_history_hash: str
    summary_metrics: dict[str, float]      # Sharpe, MDD, etc. (als Check)


class ReproducibilityCertificate(BaseModel):
    """Das Zertifikat für einen Backtest-Lauf."""
    certificate_id: str                    # UUID
    certificate_version: str = "1.0"
    created_at: datetime
    
    # Was wurde gelauft
    backtest_name: str                     # z.B. "trend_news_v4_full_history"
    backtest_period_start: datetime
    backtest_period_end: datetime
    
    # Die drei Hashes
    environment: EnvironmentFingerprint
    inputs: InputFingerprint
    outputs: OutputFingerprint
    
    # Meta
    run_duration_seconds: float
    run_host: str                          # hostname
    notes: Optional[str] = None
```

### 2.2 Der Generator

```python
# src/assembled_core/certify/generator.py
import hashlib
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import (
    ReproducibilityCertificate,
    EnvironmentFingerprint,
    InputFingerprint,
    OutputFingerprint,
)


def file_sha256(path: Path) -> str:
    """SHA-256 hash eines Files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def object_sha256(obj: Any) -> str:
    """SHA-256 einer Python-Objekt-Serialisierung.
    
    Wichtig: sort_keys=True für deterministische Order.
    Floats werden auf 12 signifikante Stellen gerundet (um bit-level Unterschiede
    über Plattformen zu absorbieren). Wer bit-exact will, entfernt das Rounding.
    """
    if isinstance(obj, pd.Series):
        # Series als sortiertes dict mit float-normalize
        items = sorted(obj.items())
        normalized = [(str(k), round(float(v), 12)) for k, v in items]
        data = json.dumps(normalized, sort_keys=True)
    elif isinstance(obj, pd.DataFrame):
        data = obj.to_json(orient="records", date_format="iso")
    elif isinstance(obj, (dict, list)):
        data = json.dumps(obj, sort_keys=True, default=str)
    else:
        data = str(obj)
    
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def get_environment_fingerprint(uv_lock_path: Path = Path("uv.lock")) -> EnvironmentFingerprint:
    """Erfasst die aktuelle Umgebung."""
    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    
    # System libs (simple)
    sys_libs = {}
    try:
        result = subprocess.run(["ldd", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            first_line = result.stdout.split("\n")[0]
            sys_libs["glibc"] = first_line.split()[-1]
    except Exception:
        pass
    
    return EnvironmentFingerprint(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        cpu_model=cpu_model,
        docker_image_sha=_detect_docker_sha(),
        uv_lock_hash=file_sha256(uv_lock_path) if uv_lock_path.exists() else "no_lockfile",
        system_libraries=sys_libs,
    )


def _detect_docker_sha() -> str | None:
    """Detect if running in Docker, return image SHA if yes."""
    try:
        with open("/proc/1/cgroup") as f:
            content = f.read()
            if "docker" in content:
                # Try reading hostname (often is container-id)
                with open("/etc/hostname") as hf:
                    return hf.read().strip()
    except Exception:
        pass
    return None


def get_git_info() -> tuple[str, bool]:
    """Aktueller Git-SHA und ob Working-Tree dirty."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        return sha, dirty
    except Exception:
        return "unknown", True


def build_input_fingerprint(
    strategy_config_path: Path,
    data_file_path: Path,
    data_row_count: int,
    ml_model_refs: dict[str, str] = None,
    ml_model_paths: dict[str, Path] = None,
    seed: int = 42,
) -> InputFingerprint:
    """Erfasst alle Inputs."""
    git_sha, dirty = get_git_info()
    
    ml_model_hashes = {}
    if ml_model_paths:
        for name, path in ml_model_paths.items():
            ml_model_hashes[name] = file_sha256(path)
    
    return InputFingerprint(
        git_sha=git_sha,
        git_dirty=dirty,
        strategy_config_hash=file_sha256(strategy_config_path),
        strategy_config_path=str(strategy_config_path),
        data_file_hash=file_sha256(data_file_path),
        data_file_path=str(data_file_path),
        data_row_count=data_row_count,
        ml_model_refs=ml_model_refs or {},
        ml_model_hashes=ml_model_hashes,
        seed=seed,
    )


def build_output_fingerprint(
    equity_curve: pd.Series,
    trade_list: pd.DataFrame,
    signal_history: pd.DataFrame,
    summary_metrics: dict[str, float],
) -> OutputFingerprint:
    """Erfasst alle Outputs als Hashes."""
    return OutputFingerprint(
        equity_curve_hash=object_sha256(equity_curve),
        trade_list_hash=object_sha256(trade_list),
        signal_history_hash=object_sha256(signal_history),
        summary_metrics={k: round(v, 12) for k, v in summary_metrics.items()},
    )


def generate_certificate(
    backtest_name: str,
    period_start: datetime,
    period_end: datetime,
    strategy_config_path: Path,
    data_file_path: Path,
    data_row_count: int,
    equity_curve: pd.Series,
    trade_list: pd.DataFrame,
    signal_history: pd.DataFrame,
    summary_metrics: dict[str, float],
    run_duration: float,
    ml_model_refs: dict[str, str] = None,
    ml_model_paths: dict[str, Path] = None,
    seed: int = 42,
    notes: str = None,
) -> ReproducibilityCertificate:
    """Generiert ein vollständiges Reproducibility-Zertifikat."""
    import socket
    
    return ReproducibilityCertificate(
        certificate_id=str(uuid.uuid4()),
        created_at=datetime.utcnow(),
        backtest_name=backtest_name,
        backtest_period_start=period_start,
        backtest_period_end=period_end,
        environment=get_environment_fingerprint(),
        inputs=build_input_fingerprint(
            strategy_config_path=strategy_config_path,
            data_file_path=data_file_path,
            data_row_count=data_row_count,
            ml_model_refs=ml_model_refs,
            ml_model_paths=ml_model_paths,
            seed=seed,
        ),
        outputs=build_output_fingerprint(
            equity_curve=equity_curve,
            trade_list=trade_list,
            signal_history=signal_history,
            summary_metrics=summary_metrics,
        ),
        run_duration_seconds=run_duration,
        run_host=socket.gethostname(),
        notes=notes,
    )


def save_certificate(cert: ReproducibilityCertificate, path: Path):
    """Zertifikat als JSON speichern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cert.model_dump(), f, indent=2, default=str)
```

---

## 3. Der Integration-Point

### 3.1 Im Backtest-Runner

```python
# scripts/backtest/run_with_certificate.py
"""
Wrapper um den normalen Backtest-Runner, der ein Zertifikat generiert.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path
import random
import numpy as np

from assembled_core.backtest.engine import BacktestEngine
from assembled_core.certify.generator import generate_certificate, save_certificate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-config", required=True, type=Path)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()
    
    # --- Seeding ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    # falls PyTorch/Sklearn/... zusätzlich
    
    # --- Run ---
    start = time.time()
    
    engine = BacktestEngine(
        config_path=args.strategy_config,
        data_path=args.data_file,
        seed=args.seed,
    )
    
    results = engine.run(
        start=datetime.fromisoformat(args.start_date),
        end=datetime.fromisoformat(args.end_date),
    )
    
    duration = time.time() - start
    
    # --- Artefakte schreiben ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    equity_path = args.output_dir / "equity_curve.parquet"
    results.equity_curve.to_frame("equity").to_parquet(equity_path)
    
    trades_path = args.output_dir / "trades.parquet"
    results.trades.to_parquet(trades_path)
    
    signals_path = args.output_dir / "signals.parquet"
    results.signals.to_parquet(signals_path)
    
    # --- Zertifikat ---
    cert = generate_certificate(
        backtest_name=args.strategy_config.stem + "_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        period_start=datetime.fromisoformat(args.start_date),
        period_end=datetime.fromisoformat(args.end_date),
        strategy_config_path=args.strategy_config,
        data_file_path=args.data_file,
        data_row_count=len(results.signals),
        equity_curve=results.equity_curve,
        trade_list=results.trades,
        signal_history=results.signals,
        summary_metrics={
            "sharpe": results.sharpe_ratio,
            "max_drawdown": results.max_drawdown,
            "total_return": results.total_return,
            "n_trades": float(len(results.trades)),
        },
        run_duration=duration,
        ml_model_refs=engine.get_model_refs(),
        ml_model_paths=engine.get_model_paths(),
        seed=args.seed,
        notes=args.notes,
    )
    
    cert_path = args.output_dir / "certificate.json"
    save_certificate(cert, cert_path)
    
    print(f"Backtest complete. Certificate: {cert_path}")
    print(f"Equity-Curve-Hash: {cert.outputs.equity_curve_hash[:16]}...")
    print(f"Git-SHA: {cert.inputs.git_sha[:12]}{'[DIRTY]' if cert.inputs.git_dirty else ''}")
    print(f"Sharpe: {cert.outputs.summary_metrics['sharpe']:.4f}")


if __name__ == "__main__":
    main()
```

### 3.2 Der Verifier

```python
# scripts/backtest/verify_certificate.py
"""
Lädt ein altes Zertifikat, läuft den Backtest erneut, vergleicht.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from assembled_core.certify.schema import ReproducibilityCertificate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--certificate", required=True, type=Path,
                       help="Path to certificate.json to verify")
    parser.add_argument("--tolerance", type=float, default=0.0,
                       help="Tolerance on summary metrics (0.0 = exact)")
    args = parser.parse_args()
    
    cert = ReproducibilityCertificate(**json.load(open(args.certificate)))
    
    print(f"=== Verification of {cert.backtest_name} ===")
    print(f"Original run: {cert.created_at}")
    print(f"Git SHA: {cert.inputs.git_sha}")
    print()
    
    # --- Check environment matches ---
    from assembled_core.certify.generator import get_environment_fingerprint, get_git_info
    
    current_env = get_environment_fingerprint()
    env_warnings = []
    
    if cert.environment.python_version != current_env.python_version:
        env_warnings.append(
            f"Python version mismatch: {cert.environment.python_version} vs "
            f"{current_env.python_version}"
        )
    
    if cert.environment.uv_lock_hash != current_env.uv_lock_hash:
        env_warnings.append(
            f"uv.lock hash mismatch: {cert.environment.uv_lock_hash[:12]} vs "
            f"{current_env.uv_lock_hash[:12]}"
        )
    
    if cert.environment.cpu_model != current_env.cpu_model:
        env_warnings.append(
            f"CPU model mismatch: {cert.environment.cpu_model} vs {current_env.cpu_model} "
            f"(floating-point may differ)"
        )
    
    if env_warnings:
        print("⚠️  Environment divergence detected:")
        for w in env_warnings:
            print(f"    - {w}")
        print()
        print("    Reproducibility is NOT guaranteed in this environment.")
        print("    Recommendation: rebuild Docker image from pinned commit.")
        response = input("    Continue anyway? [y/N]: ")
        if response.lower() != "y":
            sys.exit(1)
    
    # --- Re-run backtest ---
    print("\nRe-running backtest with certified inputs...\n")
    
    output_dir = args.certificate.parent / f"verify_{datetime.utcnow():%Y%m%d_%H%M%S}"
    
    cmd = [
        sys.executable, "-m", "scripts.backtest.run_with_certificate",
        "--strategy-config", cert.inputs.strategy_config_path,
        "--data-file", cert.inputs.data_file_path,
        "--start-date", cert.backtest_period_start.isoformat(),
        "--end-date", cert.backtest_period_end.isoformat(),
        "--seed", str(cert.inputs.seed),
        "--output-dir", str(output_dir),
    ]
    
    subprocess.run(cmd, check=True)
    
    new_cert = ReproducibilityCertificate(**json.load(open(output_dir / "certificate.json")))
    
    # --- Compare ---
    print("\n=== Hash Comparison ===\n")
    
    checks = [
        ("Equity Curve", cert.outputs.equity_curve_hash, new_cert.outputs.equity_curve_hash),
        ("Trade List", cert.outputs.trade_list_hash, new_cert.outputs.trade_list_hash),
        ("Signal History", cert.outputs.signal_history_hash, new_cert.outputs.signal_history_hash),
    ]
    
    all_match = True
    for name, old, new in checks:
        if old == new:
            print(f"✓ {name}: {old[:16]}... = {new[:16]}...")
        else:
            print(f"✗ {name}: {old[:16]}... ≠ {new[:16]}...")
            all_match = False
    
    # --- Summary metrics with tolerance ---
    print("\n=== Summary Metrics ===\n")
    for metric_name in cert.outputs.summary_metrics:
        old_val = cert.outputs.summary_metrics[metric_name]
        new_val = new_cert.outputs.summary_metrics[metric_name]
        diff = abs(old_val - new_val)
        
        if diff <= args.tolerance:
            status = "✓"
        else:
            status = "✗"
            all_match = False
        
        print(f"{status} {metric_name}: {old_val:.6f} vs {new_val:.6f} (Δ={diff:.2e})")
    
    # --- Final verdict ---
    print()
    if all_match:
        print("✓ VERIFICATION SUCCESSFUL — Backtest is reproducible.")
        sys.exit(0)
    else:
        print("✗ VERIFICATION FAILED — Backtest is NOT reproducible.")
        print("  Investigate: environment drift, non-deterministic code, data mutations.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## 4. Die Docker-Container-Strategie

### 4.1 Warum Docker für Zertifikate

Der uv.lock-File fixiert Python-Packages. Aber **nicht** System-Packages, Glibc-Version, Timezone-Data, Compiler. Für bit-exact-Reproduzierbarkeit nach 6+ Monaten brauchst du einen **vollständig fixierten System-State**.

Docker-Image mit fixem Digest-Hash = das. Der Digest-Hash identifiziert Byte-identischen Image-Content.

### 4.2 Das Dockerfile

```dockerfile
# Dockerfile.certified
# Pinned base image by digest, not just tag
FROM python:3.11.9-slim-bookworm@sha256:<DIGEST_HERE>

# System deps (Stand der Dinge einfrieren)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv installieren
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv

WORKDIR /app

# Lock-File kopieren und Dependencies installieren
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# App-Code
COPY src/ src/
COPY scripts/ scripts/
RUN uv pip install --no-deps -e .

# Set determinism-critical envs
ENV TZ=UTC \
    LC_ALL=C.UTF-8 \
    PYTHONHASHSEED=42 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

CMD ["python", "-m", "scripts.backtest.run_with_certificate"]
```

**Wichtige Detail:**
- `python:3.11.9-slim-bookworm@sha256:<DIGEST>` — nicht nur der Tag, der kann sich ändern. Der Digest ist ein content-hash.
- `--frozen` bei `uv sync` verhindert Lock-File-Updates.
- `PYTHONHASHSEED=42` macht Python's Hash-Randomization deterministisch (Dict-Order).
- `OMP_NUM_THREADS=1` und `MKL_NUM_THREADS=1` verhindern Floating-Point-Non-Determinismus durch Threading.

### 4.3 Build und Run

```bash
# Build, tag mit Zertifikats-Info
docker build -f Dockerfile.certified -t ata-backtest:cert-2026-04-24 .

# Get Image-SHA für Zertifikat
docker inspect --format='{{.Id}}' ata-backtest:cert-2026-04-24

# Run
docker run --rm \
    -v $(pwd)/data:/app/data:ro \
    -v $(pwd)/configs:/app/configs:ro \
    -v $(pwd)/results:/app/results \
    ata-backtest:cert-2026-04-24 \
    --strategy-config configs/strategies/trend_news_v4.yaml \
    --data-file data/sp500_2020_2026.parquet \
    --start-date 2023-01-01 \
    --end-date 2026-04-01 \
    --seed 42 \
    --output-dir results/cert_run_2026-04-24
```

### 4.4 Die Registry als Langzeit-Archiv

```bash
# Push to your own registry
docker tag ata-backtest:cert-2026-04-24 registry.hans-oertel.de/ata-backtest:cert-2026-04-24
docker push registry.hans-oertel.de/ata-backtest:cert-2026-04-24

# In 6 Monaten: pull und verify
docker pull registry.hans-oertel.de/ata-backtest:cert-2026-04-24@sha256:<DIGEST_FROM_CERT>
```

**Oder:** Docker-Save als tar für lokales Archiv:

```bash
docker save ata-backtest:cert-2026-04-24 | gzip > archive/images/ata-backtest_cert-2026-04-24.tar.gz
```

---

## 5. Wann ein Zertifikat generieren

### 5.1 Empfehlungen

**Immer:**
- Vor jedem "Feature vergleichen"-Experiment: Baseline-Zertifikat erstellen
- Bei Release eines neuen Strategy-Versions (vor Go-Live in Staging)
- Jeden Monat als "Sanity-Check" mit identischen Inputs (Drift-Detection)

**Nicht nötig:**
- Bei jedem Dev-Iteration-Run (zu aufwendig)
- Bei Exploratory-Analysen (Zweck ist Insights, nicht Reproduzierbarkeit)

### 5.2 Der Monatliche Drift-Check

```bash
# scripts/maintenance/monthly_drift_check.sh
#!/bin/bash
# Läuft am 1. jedes Monats via cron
set -euo pipefail

CERT_DIR="archive/certificates"
BASELINE_CERT="$CERT_DIR/baseline_trend_news_v4.json"

echo "=== Monthly Reproducibility Drift Check ==="
echo "Baseline cert: $BASELINE_CERT"

python -m scripts.backtest.verify_certificate \
    --certificate "$BASELINE_CERT" \
    --tolerance 1e-10

if [ $? -ne 0 ]; then
    # Send alert
    echo "DRIFT DETECTED" | mail -s "Reproducibility Drift Alert" hans@example.com
fi
```

**Was du trackst:** Sollte die Hash-Gleichheit über die Zeit brechen, ohne dass du Code/Configs geändert hast, ist etwas in der Umgebung gedriftet. Meist eine dependency, die trotz Lockfile anders auflöst.

---

## 6. Realistische Grenzen

### 6.1 Was Bit-Exaktheit NICHT kann

**CPU-Architektur-Wechsel:** Intel ↔ AMD, x86 ↔ ARM. Floating-Point-Einheiten haben minimal unterschiedliche Rounding-Behaviors. Auch innerhalb der x86-Familie (Haswell vs. Skylake vs. Zen) können Werte um 1-2 ULP abweichen.

**Lösung:** Docker-Platform-Tag explizit pinnen, nur auf derselben Architektur verifizieren.

**GPU-basierte Inferenz:** Wenn du ein News-Klassifikations-Modell auf GPU laufen lässt, ist Bit-Exaktheit **nicht garantiert**, selbst mit `torch.use_deterministic_algorithms(True)` bei manchen Operations.

**Lösung:** ML-Inference auf CPU in Backtest (meist akzeptabel bei Score-Batches), GPU nur für Training.

**Pandas-Update:** Eine neue pandas-Version kann Timeseries-Operationen ändern. Auch innerhalb einer Minor-Version gibt es gelegentlich "Bug-Fixes", die die Outputs minimal ändern.

**Lösung:** uv.lock pinnt die exakte Version. Bei Update: erst Drift-Check laufen, dann Update akzeptieren.

### 6.2 Was "genau genug" ist

Für einen Hobby-Quant reicht typischerweise:
- **Summary-Metrics (Sharpe, MDD) auf 4 Dezimalstellen genau**
- **Trade-Count exakt identisch**
- **Equity-Curve-Hash identisch innerhalb derselben CPU-Familie**

Das entspricht "praktischer Reproduzierbarkeit" — genug für Science, nicht bit-exact im Sinne von Zero-Knowledge-Proofs.

### 6.3 Was nicht reproduzierbar ist (und ok ist)

- **Wall-Clock-Zeit der Runs:** läuft länger oder kürzer je nach Load
- **Log-Einträge mit Timestamps:** immer anders
- **Randomisierte Rechenreihenfolge in `joblib`-Parallel:** kann ok sein wenn final sorted

---

## 7. Umsetzungs-Checkliste

**Phase 1 — Datenstruktur und Hash-Utils (Tag 1-2):**
- [ ] Pydantic-Schemas für Certificate
- [ ] `file_sha256`, `object_sha256` mit Float-Rounding
- [ ] Unit-Tests für Determinismus der Hashes

**Phase 2 — Backtest-Integration (Tag 3-5):**
- [ ] `run_with_certificate.py` Script
- [ ] Alle Seed-Sources konfigurieren (random, numpy, sklearn)
- [ ] Output-Artefakte standardisiert (equity.parquet, trades.parquet, signals.parquet)

**Phase 3 — Verifier (Tag 6-7):**
- [ ] `verify_certificate.py` Script
- [ ] Environment-Warning-Logic
- [ ] Tolerance-Parameter

**Phase 4 — Docker-Container (Tag 8-10):**
- [ ] Dockerfile mit Digest-Pinning
- [ ] uv.lock-Integration
- [ ] Build + Run-Scripts
- [ ] Registry-Setup (optional)

**Phase 5 — Monatliches Monitoring (Tag 11):**
- [ ] Baseline-Zertifikat erzeugen
- [ ] Cron-Job für Drift-Check
- [ ] Alert-Pipeline

**Phase 6 — Non-Determinismus-Cleanup (Tag 12-14):**
- [ ] Grep nach `datetime.now`, `random.random`, `set()`-Iterationen
- [ ] Systematischer Refactor problematischer Stellen
- [ ] Re-Verification auf demselben Input

**Gesamt-Aufwand:** 2-3 Wochen für initiale Implementation. Danach 1-2 h/Monat für Drift-Checks.

---

## 8. Quellen

**Reproducibility Best Practices:**
- GeeksforGeeks (2025): [Reproducibility in Machine Learning](https://www.geeksforgeeks.org/machine-learning/reproducibility-in-machine-learning/)
- Ingonyama (September 2024): [Solving Reproducibility Challenges in Deep Learning and LLMs](https://www.ingonyama.com/post/solving-reproducibility-challenges-in-deep-learning-and-llms-our-journey) — detaillierte Analyse von Non-Determinismus-Quellen
- Ogochukwu Ikegbo (Dezember 2024): [Ensuring Consistent Random Outputs for Reproducibility](https://medium.com/@stacymacbrains/ensuring-consistent-random-outputs-for-reproducibility-in-machine-learning-9bb23165f5c1)

**PyTorch Determinism:**
- [PyTorch Documentation: Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) — offizielle Guide
- Hey Amit (Februar 2025): [PyTorch Reproducibility: A Practical Guide](https://medium.com/@heyamit10/pytorch-reproducibility-a-practical-guide-d6f573cba679)
- Darina Bal Roitshtain: [Reproducible Deep Learning Using PyTorch](https://darinabal.medium.com/deep-learning-reproducible-results-using-pytorch-42034da5ad7) — cuDNN-Details

**Lockfiles und Environment-Pinning:**
- Andrew Nesbitt (Dezember 2025): [Docker is the Lockfile for System Packages](https://nesbitt.io/2025/12/18/docker-is-the-lockfile-for-system-packages.html) — System-Package-Reproduzierbarkeit
- Prefix.dev: [Pixi - reproducible, scientific software workflows](https://prefix.dev/blog/pixi_for_scientists) — Alternative zu uv+Docker
- Thomas Bury (April 2025): [Python Project Management with uv: MLops](https://dev.to/thomas_bury_b1a50c1156cbf/mastering-python-project-management-with-uv-part-3-mlops-38e2) — uv-Workflow

**Hash-based Validation:**
- arxiv 2602.23193 (Feb 2026): [ESAA: Event Sourcing für Agents](https://arxiv.org/html/2602.23193) — SHA-256-canonicalization für Projection-Hash-Verification

---

## 9. Ehrliche Einschätzung

**Was dieses Playbook dir gibt:**
- Hash-basiertes "Ja oder Nein" zur Frage: "Ist dieser Backtest reproduzierbar?"
- Früherkennung von Environment-Drift (monatliche Drift-Checks)
- Grundlage für belastbare "Feature X bringt +Y Sharpe"-Aussagen
- Audit-Trail für "diese Zahl kam aus genau diesem Lauf, mit diesem Input, zu dieser Zeit"

**Was es dir nicht gibt:**
- **Cross-Architecture-Reproduzierbarkeit.** Intel vs. AMD kann 1-2 ULP divergieren. Wenn du den Backtest auf deinem Laptop laufen lässt und dein Hetzner-Server hat andere CPU, sind kleine Abweichungen erwartbar.
- **GPU-Bit-Exaktheit.** Deep-Learning auf GPU ist fundamental non-deterministisch.
- **Schutz vor Data-Mutations.** Wenn yfinance dein Parquet-File **neu lädt** und Rows revidiert, ändert sich der Data-Hash. Das ist Feature (Drift-Detection), nicht Bug.

**Die drei Sachen, die du nicht auslassen darfst:**
1. **uv.lock committen und Docker mit Digest-Pinning.** Ohne das ist alles andere Theater. Der Moment, in dem dein pandas ohne deine Kenntnis von 2.1.4 auf 2.2.0 springt, ist der Moment, in dem dein ganzes Zertifikats-System kaputt ist.
2. **Seed-Management an genau einer Stelle.** Nicht in 17 verschiedenen `random.seed(42)`-Aufrufen über das Codebase verteilt. Eine Funktion `initialize_seeds(seed)` am Anfang jedes Runs, die alle Libraries seedet. Grep sollte nur diese eine Stelle finden.
3. **Monatlicher Drift-Check.** Das ist wie ein Batterie-Test im Rauchmelder — wenn du nicht checkst, weißt du nicht, ob er noch funktioniert. Einmal eingerichtet ist es in 10 Minuten pro Monat erledigt. Ausfall kann dich ganze Strategie-Investigationen kosten.

**Der ultimative Punkt:** Reproduzierbarkeit ist kein Luxus für Akademiker. Für einen Einzel-Quant ist sie **die Versicherung gegen dich selbst in 6 Monaten**. In 6 Monaten erinnerst du dich nicht mehr, welche pandas-Version, welcher Random-Seed, welche Datei-Version beim "Das war der Backtest, der +2.3 Sharpe zeigte" im Einsatz war. Das Zertifikat ist die Zeitkapsel, die es dir sagt — mit mathematischer Gewissheit, nicht mit Erinnerung.

**Damit ist das Plan-Paket komplett.** Ränge 1-10 aus der Gap-Analyse sind dokumentiert. Die 10 neuen Dateien zusammen mit den bestehenden 22 geben dir einen kompletten Playbook-Stack für den Aufbau eines produktionsreifen Einzel-Quant-Systems.
