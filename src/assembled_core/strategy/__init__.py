"""Strategy configuration and hyperparameter governance.

From 39_HYPERPARAMETER_GOVERNANCE.md.

RESEARCH-ONLY / NOT ON ANY LIVE-EXECUTION PATH.
This package (singular ``strategy``) is covered by tests but is not wired into the
live or paper execution path. The canonical strategy/config sources used at
runtime are ``strategies`` (plural) together with ``config/models.py``. Treat the
weights and config schema here as research scaffolding, not the live source of
truth, and do not introduce live-path dependencies on this package.
"""
