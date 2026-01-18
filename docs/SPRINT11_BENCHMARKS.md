# Sprint 11 Benchmarks: Event Features Vectorization

## Purpose

This document describes how to run benchmarks comparing the legacy and vectorized
implementations of event features (Sprint 11.E1).

## Equivalence Tests

The equivalence tests verify that both implementations produce identical outputs
for the same inputs.

### Run Equivalence Tests

```bash
pytest tests/test_event_features_vectorized_equivalence.py -v
```

**What it tests:**
- `build_event_feature_panel`: legacy vs vectorized
- `add_disclosure_count_feature`: legacy vs vectorized
- Edge cases: empty events, symbols without events

**Assertions:**
- Same columns
- Same row count
- Same sorting (symbol, timestamp)
- Same feature values (exact match for counts, tolerance for floats)

## Benchmark Script

The benchmark script generates deterministic synthetic data and measures
performance of both implementations.

### Run Benchmark

```bash
python scripts/dev/bench_event_features_vectorized.py
```

**Output:**
- JSON log file: `output/event_study_bench_<timestamp>.json`
- Console summary with timing and speedup

**Synthetic Data:**
- 1000 symbols
- 5 years of daily data (1825 days)
- ~50,000 events total
- Deterministic (seed=42)

**What it measures:**
- Execution time for `build_event_feature_panel` (legacy vs vectorized)
- Execution time for `add_disclosure_count_feature` (legacy vs vectorized)
- Speedup ratio (legacy_time / vectorized_time)

**Sanity Checks:**
- Both methods produce same number of rows
- Both methods produce same columns
- Feature columns are present
- No hard timing assertions (only logging)

### Benchmark Output Format

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "data": {
    "n_symbols": 1000,
    "n_days": 1825,
    "n_price_rows": 1825000,
    "n_events": 50000
  },
  "benchmarks": {
    "build_event_feature_panel": {
      "legacy": {
        "time_seconds": 123.45,
        "n_rows": 1825000,
        "n_columns": 6,
        "has_features": true
      },
      "vectorized": {
        "time_seconds": 12.34,
        "n_rows": 1825000,
        "n_columns": 6,
        "has_features": true
      },
      "speedup": 10.0
    },
    "add_disclosure_count_feature": {
      "legacy": {
        "time_seconds": 98.76,
        "n_rows": 1825000,
        "has_feature": true
      },
      "vectorized": {
        "time_seconds": 9.87,
        "n_rows": 1825000,
        "has_feature": true
      },
      "speedup": 10.0
    }
  }
}
```

## Interpretation

- **Speedup > 1.0**: Vectorized is faster
- **Speedup < 1.0**: Legacy is faster (unexpected, investigate)
- **Speedup ≈ 1.0**: Similar performance (vectorized may still be preferred for scalability)

## Notes

- Benchmarks use deterministic synthetic data (seed=42) for reproducibility
- No hard timing assertions to avoid flakiness in CI
- Results are logged for analysis, not used for pass/fail decisions
- Equivalence tests ensure correctness regardless of performance
