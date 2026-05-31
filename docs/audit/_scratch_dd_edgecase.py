"""THROWAWAY — probe production max_drawdown_pct denominator (global peak vs
peak-to-date). Safe to delete."""

from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
import pandas as pd

from src.assembled_core.qa.metrics import compute_drawdown

# Equity: rise to 100, drop to 80 (a -20% DD from peak-to-date 100),
# then RECOVER and make a NEW HIGH at 200 later. Textbook MDD% = -20%.
# Production divides max_dd_abs by GLOBAL peak (200) -> understates the %.
eq = pd.Series([100.0, 90.0, 80.0, 120.0, 200.0])
peak_to_date = eq.cummax()
dd_abs = eq - peak_to_date
mdd_abs = float(dd_abs.min())  # = -20 (at idx2: 80-100)
textbook_pct = float(((eq / peak_to_date - 1.0) * 100.0).min())  # -20.0%
global_peak = float(peak_to_date.max())  # 200
prodstyle_pct = mdd_abs / global_peak * 100.0  # -20/200 = -10.0%

_, prod_mdd_abs, prod_mdd_pct, _ = compute_drawdown(eq)

print(f"mdd_abs           : {mdd_abs:.4f}   (prod {prod_mdd_abs:.4f})")
print(f"textbook  mdd_pct : {textbook_pct:.4f} %")
print(f"prodstyle mdd_pct : {prodstyle_pct:.4f} %  (prod {prod_mdd_pct:.4f} %)")
print(f"-> production matches PRODSTYLE: {abs(prodstyle_pct - prod_mdd_pct) < 1e-9}")
print(
    f"-> divergence from textbook    : {prod_mdd_pct - textbook_pct:+.4f} pct-points "
    f"(production reports a SMALLER/BETTER drawdown when a higher peak follows the trough)"
)
