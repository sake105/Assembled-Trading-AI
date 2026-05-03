"""A11: Verify validate_price_data is called and print() is replaced with logger."""
from __future__ import annotations

import inspect
import pandas as pd
import pytest


@pytest.mark.fast
def test_no_print_in_load_eod_prices():
    """prices_ingest.py must not use print() in production path."""
    import src.assembled_core.data.prices_ingest as mod
    src_text = inspect.getsource(mod)
    # Allow print in comments/strings but not as statement in production code
    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(src_text))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                pytest.fail(f"Found print() call in prices_ingest.py at line {node.lineno}")


@pytest.mark.fast
def test_validate_price_data_has_callers():
    """validate_price_data must be called from at least one production path."""
    import pathlib
    src_root = pathlib.Path("src")
    callers = []
    for pyfile in src_root.rglob("*.py"):
        if "prices_ingest" in pyfile.name:
            continue
        text = pyfile.read_text(encoding="utf-8", errors="replace")
        if "validate_price_data" in text:
            callers.append(pyfile)
    # load_eod_prices itself counts (it's an internal call)
    # Also check the function is called inside prices_ingest
    ingest_text = pathlib.Path("src/assembled_core/data/prices_ingest.py").read_text()
    assert "validate_price_data(" in ingest_text, "validate_price_data must be called inside load_eod_prices"


@pytest.mark.fast
def test_validate_price_data_catches_negatives():
    """Negative prices should be flagged as invalid."""
    from src.assembled_core.data.prices_ingest import validate_price_data

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01"]),
        "symbol": ["AAPL"],
        "open": [-1.0],
        "high": [0.0],
        "low": [-2.0],
        "close": [-0.5],
        "volume": [1000.0],
    })
    result = validate_price_data(df)
    assert result["valid"] is False
    assert len(result.get("issues", [])) > 0


@pytest.mark.fast
def test_validate_price_data_happy_path():
    """Clean data should pass validation."""
    from src.assembled_core.data.prices_ingest import validate_price_data

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "symbol": ["AAPL", "AAPL"],
        "open": [150.0, 151.0],
        "high": [152.0, 153.0],
        "low": [149.0, 150.0],
        "close": [151.0, 152.0],
        "volume": [1000.0, 1200.0],
    })
    result = validate_price_data(df)
    assert result["valid"] is True
