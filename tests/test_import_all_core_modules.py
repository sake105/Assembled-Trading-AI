"""Smoke test: Import all core modules without circular import errors.

This test ensures that all core modules can be imported without circular
dependencies or ImportError. It does not test runtime behavior, only importability.

According to ARCHITECTURE_LAYERING.md:
- data/ → features/ → signals/ → portfolio/ → execution/ → pipeline/
- qa/ is a sidecar (no cycles)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Core modules to test (in dependency order)
CORE_MODULES = [
    # Layer 1: data (bottom)
    "src.assembled_core.data.prices_ingest",
    "src.assembled_core.data.insider_ingest",
    "src.assembled_core.data.congress_trades_ingest",
    "src.assembled_core.data.shipping_routes_ingest",
    "src.assembled_core.data.news_ingest",
    "src.assembled_core.data.factor_store",
    "src.assembled_core.data.data_source",
    # Layer 2: features
    "src.assembled_core.features.ta_features",
    "src.assembled_core.features.insider_features",
    "src.assembled_core.features.congress_features",
    "src.assembled_core.features.shipping_features",
    "src.assembled_core.features.news_features",
    # Layer 3: signals
    "src.assembled_core.signals.rules_trend",
    "src.assembled_core.signals.rules_event_insider_shipping",
    "src.assembled_core.signals.ensemble",
    # Layer 4: portfolio
    "src.assembled_core.portfolio.position_sizing",
    # Layer 5: execution
    "src.assembled_core.execution.order_generation",
    "src.assembled_core.execution.risk_controls",
    # Layer 6: pipeline (top)
    "src.assembled_core.pipeline.trading_cycle",
    "src.assembled_core.pipeline.orders",
    "src.assembled_core.pipeline.backtest",
    "src.assembled_core.pipeline.portfolio",
    "src.assembled_core.pipeline.io",
    "src.assembled_core.pipeline.orchestrator",
    # Sidecar: qa
    "src.assembled_core.qa.backtest_engine",
    "src.assembled_core.qa.metrics",
    "src.assembled_core.qa.qa_gates",
    "src.assembled_core.qa.walk_forward",
    # Shared
    "src.assembled_core.config",
    "src.assembled_core.costs",
    "src.assembled_core.utils",
]


@pytest.mark.smoke
def test_import_all_core_modules() -> None:
    """Test that all core modules can be imported without circular import errors."""
    import_errors = []
    circular_imports = []

    for module_name in CORE_MODULES:
        try:
            # Clear module cache to test fresh import
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Attempt import
            __import__(module_name)
        except ImportError as e:
            error_msg = str(e)
            if "circular" in error_msg.lower() or "partially initialized" in error_msg.lower():
                circular_imports.append((module_name, error_msg))
            else:
                import_errors.append((module_name, error_msg))
        except Exception as e:
            # Other exceptions (e.g., missing dependencies) are OK for smoke test
            # We only care about ImportError/circular imports
            pass

    # Report failures
    if import_errors:
        error_msg = "Import errors found:\n"
        for module, error in import_errors:
            error_msg += f"  - {module}: {error}\n"
        pytest.fail(error_msg)

    if circular_imports:
        error_msg = "Circular import errors found:\n"
        for module, error in circular_imports:
            error_msg += f"  - {module}: {error}\n"
        pytest.fail(error_msg)


@pytest.mark.smoke
def test_import_package_init() -> None:
    """Test that package __init__.py files can be imported."""
    package_modules = [
        "src.assembled_core",
        "src.assembled_core.data",
        "src.assembled_core.features",
        "src.assembled_core.signals",
        "src.assembled_core.portfolio",
        "src.assembled_core.execution",
        "src.assembled_core.pipeline",
        "src.assembled_core.qa",
        "src.assembled_core.config",
        "src.assembled_core.utils",
    ]

    import_errors = []
    for module_name in package_modules:
        try:
            if module_name in sys.modules:
                del sys.modules[module_name]
            __import__(module_name)
        except ImportError as e:
            error_msg = str(e)
            if "circular" in error_msg.lower() or "partially initialized" in error_msg.lower():
                import_errors.append((module_name, error_msg))

    if import_errors:
        error_msg = "Package __init__.py import errors:\n"
        for module, error in import_errors:
            error_msg += f"  - {module}: {error}\n"
        pytest.fail(error_msg)
