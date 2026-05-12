"""Bootstrap — composition root (DI container).

The single module in the codebase allowed to import from BOTH the
ports layer and the adapters layer. Everything else gets its
dependencies passed in via constructor (Dependency Injection).
"""

from src.assembled_core.bootstrap.container import (
    Container,
    build_production_container,
    build_test_container,
)

__all__ = [
    "Container",
    "build_production_container",
    "build_test_container",
]
