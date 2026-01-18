"""Leakage tests for PIT-safe feature validation.

This module provides helpers and tests to detect look-ahead bias in alt-data features.
Leakage tests verify that features do not use events that have not yet been disclosed.

Key Functions:
    - assert_feature_zero_before_disclosure(): Validates feature is zero before disclosure_date
"""

from src.assembled_core.qa.leakage_tests.altdata_leakage import (
    assert_feature_zero_before_disclosure,
)

__all__ = [
    "assert_feature_zero_before_disclosure",
]
