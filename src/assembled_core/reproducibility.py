# src/assembled_core/reproducibility.py
"""Determinism-hardening helpers for research and CI replay (audit C2-049).

A single call to :func:`set_deterministic` makes the next pipeline run as
bit-reproducible as Python and the installed numerical libraries allow. The
function sets environment variables (must be done before the affected
library imports), seeds Python / numpy / torch RNGs, and toggles framework
deterministic-algorithm flags.

Usage at the head of a script:

    from src.assembled_core.reproducibility import set_deterministic
    set_deterministic(seed=42)

Some flags must be set *before* the corresponding library is imported. The
function is intentionally tolerant of missing optional deps — when torch is
absent, the torch branch is silently skipped.

The function returns a dict describing what was applied, suitable for
logging into the run manifest.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

logger = logging.getLogger(__name__)


def set_deterministic(seed: int = 42) -> dict[str, Any]:
    """Set environment + RNG flags for bit-deterministic-ish replay.

    Sets:
    - ``PYTHONHASHSEED`` (no effect after interpreter start, but logged so
      the caller knows to wrap their entrypoint in a shell prefix).
    - ``OMP_NUM_THREADS=1`` and ``MKL_NUM_THREADS=1`` (BLAS reductions order).
    - ``MKL_CBWR=COMPATIBLE`` (MKL conditional bitwise reproducibility).
    - ``CUBLAS_WORKSPACE_CONFIG=":4096:8"`` (cuBLAS determinism — only takes
      effect if torch/CUDA loaded AFTER this call).
    - ``random.seed(seed)``.
    - ``numpy.random.seed(seed)`` (legacy global; new code should use
      :class:`numpy.random.Generator` with ``np.random.default_rng(seed)``).
    - ``torch.manual_seed`` / ``torch.cuda.manual_seed_all`` /
      ``torch.use_deterministic_algorithms(True)`` if torch is importable.
    - ``torch.backends.cudnn.deterministic=True``, ``cudnn.benchmark=False``.

    Returns:
        Dict with keys: ``seed``, ``env`` (the env vars set), ``rngs``
        (which RNGs were seeded), ``warnings`` (e.g. PYTHONHASHSEED already
        set differently → process-restart needed for full effect).
    """
    applied_env: dict[str, str] = {}
    warnings: list[str] = []

    desired_env = {
        "PYTHONHASHSEED": str(seed),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MKL_CBWR": "COMPATIBLE",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for k, v in desired_env.items():
        current = os.environ.get(k)
        if current and current != v:
            warnings.append(
                f"{k} already set to {current!r} (wanted {v!r}); "
                "restart the process with the desired value for full effect."
            )
        os.environ[k] = v
        applied_env[k] = v

    rngs_set: list[str] = []
    random.seed(seed)
    rngs_set.append("python_random")

    try:
        import numpy as np

        np.random.seed(seed)
        rngs_set.append("numpy_legacy")
    except ImportError:
        warnings.append("numpy not importable — skipped numpy seed")

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            rngs_set.append("torch_cuda")
        rngs_set.append("torch_cpu")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"torch.use_deterministic_algorithms failed: {exc}")
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"torch.backends.cudnn flags failed: {exc}")
    except ImportError:
        pass  # torch is optional in this repo

    summary: dict[str, Any] = {
        "seed": seed,
        "env": applied_env,
        "rngs": rngs_set,
        "warnings": warnings,
    }
    logger.info(
        "[reproducibility] seeded RNGs=%s, env_keys=%s, warnings=%d",
        rngs_set,
        list(applied_env.keys()),
        len(warnings),
    )
    return summary
