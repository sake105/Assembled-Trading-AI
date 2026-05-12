# Determinism & Reproducibility Playbook

> Audit C2-049 + C2-047 + C2-048 — how the three reproducibility tools
> introduced through the audit sweep fit together. The goal is not
> "byte-identical CI replays for ever". The goal is:
> *six months from now, someone (including future you) can read this
> file, this commit, this seed, this data hash, and reproduce the same
> Sharpe within machine epsilon.*

---

## The three tools

| Tool | Lives in | Purpose |
|---|---|---|
| `set_deterministic(seed)` | `src/assembled_core/reproducibility.py` | Pin RNGs + numerical-library flags before any backtest run. |
| `docs/RESEARCH_REGISTER_TEMPLATE.md` | docs/ | Index every backtest that ends up on the roadmap. |
| `research/dead_ends/README.md` + entries | research/dead_ends/ | Record every experiment that didn't survive validation. |

## When you start a new experiment

```python
from src.assembled_core.reproducibility import set_deterministic

summary = set_deterministic(seed=42)  # write the seed into the register

# ... your backtest code ...

print(summary)  # paste into the research-register entry
```

Notes:
- Call `set_deterministic` **at the top** of the entry-script. Some env
  vars (e.g. `CUBLAS_WORKSPACE_CONFIG`) only take effect before the
  affected library imports.
- Pass the **same seed** as the value you write into the register.
  Mismatch is the most common reproducibility bug.

## When you publish a result

The result MUST be promotable only after these are all done:

1. Backtest ran with `set_deterministic`.
2. An entry is appended to `docs/research_register.md` (copy from
   `docs/RESEARCH_REGISTER_TEMPLATE.md`) with all fields filled —
   especially `Data hash / DVC tag`.
3. The four quant gates have been computed and pass per
   `configs/policy.yaml > quant_gates`:
   - `deflated_sharpe.block_threshold` (DSR > 0)
   - `probabilistic_sharpe.block_threshold` (PSR ≥ 0.95)
   - `pbo.block_threshold` (PBO < 0.5)
   - `min_trl.block_when_below_observed_track`
4. The permutation test (`qa/metrics.permutation_p_value`) returns
   p < 0.01.
5. If the result is a candidate for the ensemble, the
   `correlation_promotion_gate` admits it.

If any of those fails, the experiment goes to `research/dead_ends/`,
not to a Slack screenshot.

## When you abandon an experiment

Don't just delete the notebook. Write a dead-end:

```bash
cp research/dead_ends/README.md research/dead_ends/$(date +%Y-%m-%d)-<slug>.md
# edit with: hypothesis, what we tried, why it failed,
# "when NOT to retry", "when TO retry (legitimate triggers)"
git add research/dead_ends/
git commit -m "research(dead-end): <slug> — <one-line reason>"
```

This stops the same experiment being rerun six months later.

## What you should NOT expect

- **Bit-for-bit reproducibility across hardware**: BLAS / threading
  layouts differ. Two different machines can produce different
  Sharpe numbers in the 6th decimal even with the same seed.
- **Reproducibility across yfinance revisions**: vendor data changes.
  The `Data hash` field exists exactly to flag this.
- **Reproducibility on the prefix of a long run** in HAR-RV /
  walk-forward style — the OLS fit on a prefix is structurally
  different from the fit on the whole series. The PIT-safety
  contract is at the regressor level, not at the beta level
  (see `tests/test_audit_additions.py::test_har_rv_forecast_prefix_correlation_high`).

## Where to look when reproduction fails

1. **Different seed**: did the runtime read the env var
   `PYTHONHASHSEED` you intended? `set_deterministic` returns the
   final env state; print it.
2. **Different numpy version**: random-number streams between major
   versions of numpy can differ. `numpy.__version__` is logged into
   the register entry for exactly this reason.
3. **Different data**: `Data hash / DVC tag` mismatch.
4. **Different config**: did `policy.yaml` change between the runs?
   The `config_hash` written into `output/factors/.../manifest.json`
   by FactorStore is the right place to inspect.

## How the snapshot tests fit in

`tests/test_snapshot_metrics.py` pins the numerical output of
metric functions on a **fixed seeded input**. When you intentionally
change the math, you regenerate the snapshots with
`pytest --snapshot-update`. Unintentional drift is caught
immediately. This is the lowest-friction guard against silent
numerical bugs.
