"""GDPR compliance utilities.

From 50_COMPLIANCE_RECHT.md §50.2.
Personal-use only: aggregation-first approach to avoid storing PII.
"""

from __future__ import annotations

import hashlib
import os


def pseudonymize_user(user_id: str, salt_env_var: str = "ATA_PSEUDO_SALT") -> str:
    """Irreversible pseudonym for a user identifier.

    Uses SHA-256 with a secret salt so the same user_id always maps to the
    same pseudonym, but cannot be reversed without the salt.

    Args:
        user_id: Raw user identifier (username, Reddit user, etc.).
        salt_env_var: Environment variable holding the salt.  Falls back to a
            deterministic but weaker constant when the env var is absent (dev
            / test environments without real PII).

    Returns:
        16-character hex string safe to store in the DB.
    """
    salt = os.environ.get(salt_env_var, "dev_pseudo_salt_not_for_prod")
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
    return digest[:16]


def should_retain(created_at_iso: str, retention_days: int = 365) -> bool:
    """Return True if the record is still within the retention window.

    Args:
        created_at_iso: ISO-8601 timestamp string of when the record was created.
        retention_days: Maximum age in days (default 365).
    """
    from datetime import datetime, timezone

    created = datetime.fromisoformat(created_at_iso)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - created).days
    return age <= retention_days


def anonymize_news_headline(headline: str) -> str:
    """Return a SHA-256 hash of the headline to avoid storing raw author-
    linkable text in persistent storage.

    Use this for headlines that may contain identifiable information
    (e.g. author names embedded in meta-data).
    """
    return hashlib.sha256(headline.encode()).hexdigest()
