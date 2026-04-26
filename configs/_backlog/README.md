# configs/_backlog

Config files moved here because they have no active callers in `src/`, `scripts/`, or `tests/`.
They are kept for reference but not loaded at runtime.

Archived 2026-04-26:
- `batch_backtest_example_doc_schema.yaml` — documentation example schema, no runtime caller
- `news_sources.yaml` — superseded by `configs/news/news.yaml` + `configs/intel/rss_feeds.yaml`
- `nation_profiles.yaml` — only caller is `intel/nation_profiles.py` which itself has no pipeline callers (see B6)
