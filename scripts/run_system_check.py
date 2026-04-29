"""CLI entry point for the System-Check Tournament.

Usage:
    python scripts/run_system_check.py                    # full tournament
    python scripts/run_system_check.py --dry-run          # no API calls
    python scripts/run_system_check.py --critics 10       # reduced run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Ensure repo root importability when run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from system_check.runner.claude_client import (  # noqa: E402
    ClaudeClient,
    ClaudeClientConfig,
    RetryConfig,
)
from system_check.runner.judge import parse_judge_output  # noqa: E402
from system_check.runner.report import write_run_artifacts  # noqa: E402
from system_check.runner.tournament import run_tournament  # noqa: E402

SYSTEM_CHECK_DIR = REPO_ROOT / "system_check"
DEFAULT_CONFIG = SYSTEM_CHECK_DIR / "config" / "tournament_default.yaml"
DEFAULT_DEFENDERS = SYSTEM_CHECK_DIR / "personas" / "defenders.yaml"
DEFAULT_CRITICS = SYSTEM_CHECK_DIR / "personas" / "critics.yaml"
DEFAULT_RUNS_DIR = SYSTEM_CHECK_DIR / "runs"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _try_load_dotenv(repo_root: Path) -> None:
    """Read a `.env` file into os.environ (minimal parser, never logs values)."""
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        logging.getLogger(__name__).warning(".env unreadable: %s", exc)


def _build_run_dir(override: Path | None, git_sha: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = git_sha[:7] if git_sha and git_sha != "unknown" else "nogit"
    base = override or DEFAULT_RUNS_DIR
    return base / f"{ts}_{short}"


def _git_sha() -> str:
    import subprocess
    try:
        res = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an adversarial system-check tournament.",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="Tournament config YAML.")
    p.add_argument("--defenders-yaml", type=Path, default=DEFAULT_DEFENDERS,
                   help="Defender persona YAML.")
    p.add_argument("--critics-yaml", type=Path, default=DEFAULT_CRITICS,
                   help="Critic persona YAML.")
    p.add_argument("--critics", type=int, default=None,
                   help="Cap number of critics used (applied in order).")
    p.add_argument("--defenders", type=int, default=None,
                   help="Cap number of defenders used (applied in order).")
    p.add_argument("--output", type=Path, default=None,
                   help="Alternative parent dir for the run (default: system_check/runs/).")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip all API calls; produce placeholder transcript.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


async def _run_async(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    log = logging.getLogger("system_check.cli")

    if not args.config.exists():
        log.error("config not found: %s", args.config)
        return 2
    if not args.defenders_yaml.exists() or not args.critics_yaml.exists():
        log.error("persona YAMLs missing (defenders=%s critics=%s)",
                  args.defenders_yaml, args.critics_yaml)
        return 2

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    # Safety gate: real API calls require ANTHROPIC_API_KEY.
    _try_load_dotenv(REPO_ROOT)
    if not args.dry_run and cfg.get("safety", {}).get("require_api_key", True):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log.error(
                "ANTHROPIC_API_KEY missing. Add it to .env (never commit) or "
                "run with --dry-run."
            )
            return 3

    full_scale = args.critics is None and args.defenders is None

    # Pre-compute run dir so the client + tournament share the same path.
    git_sha = _git_sha()
    run_dir = _build_run_dir(args.output, git_sha)
    log.info("run_dir=%s", run_dir)

    client = ClaudeClient(ClaudeClientConfig(
        dry_run=args.dry_run,
        retry=RetryConfig(
            max_attempts=cfg["retry"]["max_attempts"],
            initial_backoff_seconds=cfg["retry"]["initial_backoff_seconds"],
            backoff_multiplier=cfg["retry"]["backoff_multiplier"],
            retry_on_status=tuple(cfg["retry"]["retry_on_status"]),
            per_call_timeout_seconds=cfg["rounds"]["per_call_timeout_seconds"],
        ),
    ))

    result = await run_tournament(
        project_root=REPO_ROOT,
        run_dir=run_dir,
        config=cfg,
        defenders_path=args.defenders_yaml,
        critics_path=args.critics_yaml,
        client=client,
        max_defenders=args.defenders,
        max_critics=args.critics,
        full_scale=full_scale,
    )

    judge_output = parse_judge_output(result.judge_content)
    written = write_run_artifacts(result, judge_output=judge_output)

    log.info(
        "[done] turns=%s tokens_in=%s tokens_out=%s cost_est_usd=$%.4f",
        len(result.turns),
        result.total_input_tokens,
        result.total_output_tokens,
        result.cost_estimate_usd,
    )
    log.info("report: %s", written["report"])

    if result.errors:
        log.warning("%s errors recorded — see report/manifest", len(result.errors))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(_run_async(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[system_check] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
