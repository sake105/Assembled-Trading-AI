#!/usr/bin/env python
"""
Batch Backtest Entry Point (P4) - Official wrapper for batch_runner.

This is the "blessed" entry point for batch backtests. It wraps
scripts/batch_runner.py and provides the same functionality without
duplicating logic.

Usage:
    python scripts/batch_backtest.py --config-file configs/batch_example.yaml

See docs/BATCH_RUNNER_P4.md for detailed documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import and delegate to batch_runner.main()
from scripts.batch_runner import main

if __name__ == "__main__":
    sys.exit(main())
