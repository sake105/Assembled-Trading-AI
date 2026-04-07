"""Signal Plugin System (Plan 11.8).

Load signal functions from external Python files via auto-discovery.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def load_signal_plugin(filepath: str | Path) -> Callable | None:
    """Load a signal function from an external Python file.

    The file must define a `signal_fn` callable.

    Args:
        filepath: Path to Python file.

    Returns:
        The signal_fn callable, or None on failure.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning("[Plugin] File not found: %s", filepath)
        return None

    try:
        spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "signal_fn"):
            logger.info("[Plugin] Loaded signal_fn from %s", filepath)
            return module.signal_fn
        else:
            logger.warning("[Plugin] No signal_fn found in %s", filepath)
            return None
    except Exception as exc:
        logger.error("[Plugin] Failed to load %s: %s", filepath, exc)
        return None


def discover_signal_plugins(plugin_dir: str = "plugins/signals") -> dict[str, Callable]:
    """Auto-discover signal plugins from a directory.

    Args:
        plugin_dir: Directory containing plugin .py files.

    Returns:
        Dict of plugin_name → signal_fn callable.
    """
    plugin_path = Path(plugin_dir)
    if not plugin_path.exists():
        return {}

    plugins: dict[str, Callable] = {}
    for py_file in sorted(plugin_path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        fn = load_signal_plugin(py_file)
        if fn is not None:
            plugins[py_file.stem] = fn

    logger.info("[Plugin] Discovered %d signal plugins from %s", len(plugins), plugin_dir)
    return plugins


__all__ = ["load_signal_plugin", "discover_signal_plugins"]
