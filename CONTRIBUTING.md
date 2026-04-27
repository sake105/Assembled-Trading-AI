# Contributing

This is a personal research project. External contributions are not expected.

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -q --ignore=tests/regression/test_htb_rate_table.py
```

## Code style

```bash
ruff check src/ scripts/ tests/
black --check src/ scripts/ tests/
```

## Commit conventions

Use conventional commits: `feat:`, `fix:`, `refactor:`, `perf:`, `test:`, `docs:`, `chore:`.
