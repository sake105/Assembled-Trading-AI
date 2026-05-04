"""Reproducibility certificate data models.

From 43_BACKTEST_REPRODUCIBILITY_CERTIFICATE.md.

A ReproducibilityCertificate captures all inputs and outputs of a
backtest run as SHA-256 hashes.  Re-running the same code on the same
environment should produce bit-identical outputs and thus identical hashes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class EnvironmentFingerprint:
    """Captures the computational environment."""

    python_version: str = ""
    platform: str = ""
    git_sha: str = ""
    git_dirty: bool = False
    package_hashes: dict[str, str] = field(default_factory=dict)  # pkg → version
    random_seeds: dict[str, int] = field(default_factory=dict)  # numpy, random, torch


@dataclass
class InputFingerprint:
    """Captures input data and configuration."""

    data_file_hashes: dict[str, str] = field(default_factory=dict)  # path → sha256
    config_hash: str = ""
    config_path: str = ""
    model_hashes: dict[str, str] = field(default_factory=dict)  # name → sha256


@dataclass
class OutputFingerprint:
    """Captures output artifacts."""

    equity_curve_hash: str = ""
    trades_hash: str = ""
    signals_hash: str = ""
    summary_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ReproducibilityCertificate:
    """Full reproducibility certificate for a single backtest run."""

    certificate_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    environment: EnvironmentFingerprint = field(default_factory=EnvironmentFingerprint)
    inputs: InputFingerprint = field(default_factory=InputFingerprint)
    outputs: OutputFingerprint = field(default_factory=OutputFingerprint)
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "ReproducibilityCertificate":
        d = dict(d)
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        d["environment"] = EnvironmentFingerprint(**d.get("environment", {}))
        d["inputs"] = InputFingerprint(**d.get("inputs", {}))
        d["outputs"] = OutputFingerprint(**d.get("outputs", {}))
        return cls(**d)


__all__ = [
    "EnvironmentFingerprint",
    "InputFingerprint",
    "OutputFingerprint",
    "ReproducibilityCertificate",
]
