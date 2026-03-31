"""E2E: Ops golden path for Evidence Pack.

Flow (lightweight, tmp_path only):
1) CLI import snapshot
2) build_ledger_from_trades(... broker_snapshot_policy="require", broker_snapshot_run_id="ops_ns", write_evidence_pack=True)
3) Verify evidence pack ZIP offline (verify_evidence_pack_zip -> ok True)
4) Optional: ensure pack_manifest_*.json exists in ZIP and schema_version==1
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_pack import verify_evidence_pack_zip
from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades


def test_ops_golden_path_evidence_pack_e2e(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    as_of_date = "2025-01-15"
    run_id = "ops_chain_run"
    snapshot_run_id = "ops_ns"

    # Create external snapshot input for CLI import (matches expected ledger result).
    # start_cash=10000, BUY 1 @ 100 -> cash=9900, position AAPL qty=1
    snapshot_input = tmp_path / "external_snapshot.json"
    snapshot_payload = {
        "as_of": as_of_date,
        "cash": 9900.0,
        "positions": [{"symbol": "AAPL", "qty": 1.0}],
    }
    snapshot_input.write_text(
        json.dumps(snapshot_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    # Step 1: CLI import snapshot
    import_script = ROOT / "scripts" / "import_broker_snapshot.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(import_script),
            "--input",
            str(snapshot_input),
            "--run-id",
            snapshot_run_id,
            "--as-of-date",
            as_of_date,
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert (
        proc.returncode == 0
    ), f"import failed: stdout={proc.stdout} stderr={proc.stderr}"

    # Step 2: Minimal ledger build that requires the imported snapshot + writes evidence pack
    ts = pd.Timestamp(f"{as_of_date}T00:00:00Z")
    orders_df = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 1.0,
                "price": 100.0,
            }
        ]
    )
    trades_df = pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 1.0,
                "price": 100.0,
            }
        ]
    )

    ledger_result = build_ledger_from_trades(
        orders_df=orders_df,
        trades_df=trades_df,
        run_id=run_id,
        output_dir=output_dir,
        as_of_date=as_of_date,
        start_cash=10000.0,
        broker_snapshot_policy="require",
        broker_snapshot_run_id=snapshot_run_id,
        write_evidence_pack=True,
    )

    # Step 3: Assert ZIP exists + offline verify ok
    pack_rel = ledger_result.get("evidence_pack_path")
    assert (
        isinstance(pack_rel, str) and pack_rel
    ), f"missing evidence_pack_path: {ledger_result}"
    zip_path = output_dir / pack_rel
    assert zip_path.exists(), f"expected evidence pack zip: {zip_path}"

    verify = verify_evidence_pack_zip(zip_path)
    assert verify["ok"] is True, f"expected ok=True, got: {verify}"

    # Step 4: Optional - pack manifest exists in ZIP and schema_version==1
    with zipfile.ZipFile(zip_path, "r") as zf:
        manifest_names = [
            n
            for n in zf.namelist()
            if n.startswith("pack_manifest_")
            and n.endswith(".json")
            and "/" not in n.strip("/")
        ]
        assert manifest_names, "expected pack_manifest_*.json in ZIP root"
        manifest_name = sorted(manifest_names)[0]
        with zf.open(manifest_name) as f:
            manifest = json.loads(f.read().decode("utf-8"))
        assert manifest.get("schema_version") == 1
