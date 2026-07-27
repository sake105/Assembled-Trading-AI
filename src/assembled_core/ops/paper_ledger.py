"""OPS-4: Paper execution ledger — load/save state, simulate fills, apply to ledger, mark-to-market equity.

Safety features:
- File locking via filelock (prevents concurrent writes)
- Backup rotation (3 generations: .1, .2, .3)
- JSON validation on load (falls back to backup if corrupted)
- Equity curve deduplication (prevents duplicate entries on re-run)
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "paper.ledger_state.v1"
_BACKUP_GENERATIONS = 3


class LedgerCorruptionError(RuntimeError):
    """Raised when ledger files exist but no generation can be parsed.

    Distinguishes catastrophic corruption (main + at least one backup existed,
    so prior persisted state is lost) from a legitimately missing file (cold
    start) or a single interrupted first write with no backups yet. The first
    two stay silent / recoverable; only catastrophic corruption fails loud so it
    cannot be masked by a silent reset to ``start_capital`` (audit R2-5, E-025).
    """


def load_ledger_state(
    path: str | Path,
    start_capital: float = 10000.0,
) -> dict[str, Any]:
    """Load ledger state from JSON. If missing or corrupted, try backups, then return fresh state."""
    p = Path(path)

    # Try main file first, then backups (.1, .2, .3)
    candidates = [p] + [
        p.with_suffix(p.suffix + f".{i}") for i in range(1, _BACKUP_GENERATIONS + 1)
    ]
    data = None
    n_existing = 0
    backup_existed = False
    for idx, candidate in enumerate(candidates):
        if not candidate.exists():
            continue
        n_existing += 1
        if idx > 0:  # candidates[0] is the main file; the rest are backups
            backup_existed = True
        try:
            raw = candidate.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                data = parsed
                if candidate != p:
                    logger.warning(
                        "[paper_ledger] main file corrupted, loaded from backup: %s",
                        candidate,
                    )
                break
            logger.warning(
                "[paper_ledger] %s parsed but is not a dict (got %s) — skipping",
                candidate,
                type(parsed).__name__,
            )
        except Exception as exc:
            logger.warning("[paper_ledger] failed to load %s: %s", candidate, exc)
            continue

    if data is None:
        if n_existing == 0:
            # Genuine cold start — no generation ever written. Silent fresh state.
            return _fresh_state(start_capital)
        if backup_existed:
            # Main AND at least one backup existed but none parsed → prior
            # persisted state is unrecoverable. Fail loud instead of silently
            # resetting to start_capital, which would mask the loss (R2-5/E-025).
            raise LedgerCorruptionError(
                f"[paper_ledger] all {n_existing} existing ledger generation(s) "
                f"for {p} are unreadable/corrupt — refusing to silently reset to "
                f"start_capital={start_capital}. Restore a known-good backup or "
                f"remove the corrupt files to force a deliberate cold start."
            )
        # Only the main file existed and was corrupt, with no backups yet — most
        # likely an interrupted first write. Preserve the chaos-recovery contract
        # (return fresh state, no crash), but log it loudly (E-025 detectability).
        logger.warning(
            "[paper_ledger] main file %s corrupt and no backups exist — "
            "starting fresh (start_capital=%s)",
            p,
            start_capital,
        )
        return _fresh_state(start_capital)
    # Normalize schema
    cash = data.get("cash")
    try:
        cash = float(cash) if cash is not None else start_capital
    except (TypeError, ValueError):
        cash = start_capital
    positions = data.get("positions")
    if not isinstance(positions, dict):
        positions = {}
    equity_curve = data.get("equity_curve")
    if not isinstance(equity_curve, list):
        equity_curve = []
    return {
        "schema_version": data.get("schema_version") or SCHEMA_VERSION,
        "updated_utc": data.get("updated_utc"),
        "cash": cash,
        "positions": {str(k): _norm_position(v) for k, v in positions.items()},
        "equity_curve": list(equity_curve),
    }


def _norm_position(v: Any) -> dict[str, Any]:
    if isinstance(v, dict):
        qty = v.get("qty")
        avg_price = v.get("avg_price")
        hwm = v.get("hwm")
        try:
            qty_f = float(qty) if qty is not None else 0.0
        except (TypeError, ValueError):
            qty_f = 0.0
        try:
            avg_f = float(avg_price) if avg_price is not None else 0.0
        except (TypeError, ValueError):
            avg_f = 0.0
        try:
            hwm_f = float(hwm) if hwm is not None else avg_f
        except (TypeError, ValueError):
            hwm_f = avg_f
        out: dict[str, Any] = {"qty": qty_f, "avg_price": avg_f, "hwm": hwm_f}
        # entry_ts (ISO UTC string, position-open time) feeds the zombie-killer
        # hold-time check; preserved when present, never invented (legacy
        # positions without it are skipped LOUDLY by risk/zombie_killer).
        entry_ts = v.get("entry_ts")
        if entry_ts:
            out["entry_ts"] = str(entry_ts)
        return out
    return {"qty": 0.0, "avg_price": 0.0, "hwm": 0.0}


def _fresh_state(start_capital: float) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_utc": None,
        "cash": float(start_capital),
        "positions": {},
        "equity_curve": [],
    }


def _rotate_backups(p: Path) -> None:
    """Rotate backup files: .3 deleted, .2 → .3, .1 → .2, current → .1."""
    for i in range(_BACKUP_GENERATIONS, 1, -1):
        src = p.with_suffix(p.suffix + f".{i - 1}")
        dst = p.with_suffix(p.suffix + f".{i}")
        if src.exists():
            try:
                shutil.copy2(str(src), str(dst))
            except OSError as exc:
                logger.warning(
                    "[PaperLedger] backup rotation copy failed %s -> %s: %s",
                    src,
                    dst,
                    exc,
                )
    # Current → .1
    if p.exists():
        try:
            shutil.copy2(str(p), str(p.with_suffix(p.suffix + ".1")))
        except OSError as exc:
            logger.warning(
                "[PaperLedger] backup rotation copy failed for current -> .1: %s", exc
            )


def save_ledger_state(state: dict[str, Any], path: str | Path) -> Path:
    """Persist ledger state atomically with file locking and backup rotation.

    Safety measures:
    - Rotates previous state into .1, .2, .3 backup generations
    - Uses file lock to prevent concurrent writes
    - Writes to temp file then renames (atomic on most filesystems)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    state["schema_version"] = state.get("schema_version") or SCHEMA_VERSION

    lock_path = p.with_suffix(p.suffix + ".lock")
    try:
        from filelock import FileLock

        lock = FileLock(str(lock_path), timeout=10)
    except ImportError:
        lock = None
        logger.warning("[paper_ledger] filelock not installed, skipping file lock")

    def _do_save() -> Path:
        _rotate_backups(p)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
        tmp.replace(p)
        return p

    if lock is not None:
        with lock:
            return _do_save()
    else:
        return _do_save()


def simulate_fills(
    orders: pd.DataFrame,
    prices_latest: pd.DataFrame,
    cost_model_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Simulate fills: fill at close, qty as requested; optional commission/slippage bps.

    If ``slippage_bps`` is absent from *cost_model_cfg*, falls back to ``spread_w + impact_w``
    (Almgren-Chriss params, both in bps units). Explicit ``slippage_bps=0.0`` is respected as-is.
    """
    fills: list[dict[str, Any]] = []
    if orders.empty:
        return fills
    cost = cost_model_cfg or {}
    commission_bps = float(cost.get("commission_bps", 0) or 0)
    slippage_bps = float(cost.get("slippage_bps", 0) or 0)
    if "slippage_bps" not in cost:
        # Almgren-Chriss params from policy.yaml: use spread_w + impact_w as slippage fallback.
        # Only applies when caller did NOT set slippage_bps explicitly (even slippage_bps=0 is respected).
        slippage_bps = float(cost.get("spread_w", 0) or 0) + float(
            cost.get("impact_w", 0) or 0
        )

    price_col = "close" if "close" in prices_latest.columns else "price"
    if "symbol" not in prices_latest.columns or price_col not in prices_latest.columns:
        return fills
    price_map = prices_latest.set_index("symbol")[price_col].to_dict()

    df = orders.copy()
    df["_sym"] = df["symbol"].astype(str).str.strip()
    df["_qty"] = pd.to_numeric(df["qty"], errors="coerce")
    valid = (
        df["_sym"].notna()
        & ~df["_sym"].isin(("None", "nan", ""))
        & df["_qty"].notna()
        & (df["_qty"] > 0)
    )
    df = df[valid].copy()
    if df.empty:
        return fills

    df["_base"] = df["_sym"].map(price_map).astype(float).fillna(0.0)
    df = df[df["_base"] > 0]
    if df.empty:
        return fills

    slippage_mult = 1.0 + (slippage_bps / 10000.0)
    is_sell = df["side"].astype(str).str.upper() == "SELL"
    raw_price = np.where(
        is_sell, df["_base"] / slippage_mult, df["_base"] * slippage_mult
    )
    commission = df["_base"].values * (commission_bps / 10000.0)
    fill_price = np.where(is_sell, raw_price - commission, raw_price + commission)

    fills = [
        {"symbol": sym, "side": side, "qty": float(qty), "price": float(price)}
        for sym, side, qty, price in zip(
            df["_sym"],
            df["side"].astype(str).str.upper().fillna("BUY"),
            df["_qty"],
            fill_price,
        )
    ]
    return fills


def _with_entry_ts(pos_dict: dict[str, Any], entry_ts: Any) -> dict[str, Any]:
    """Attach entry_ts (position-open time) when known; never invent one."""
    if entry_ts:
        pos_dict["entry_ts"] = str(entry_ts)
    return pos_dict


def apply_fills_to_ledger(
    state: dict[str, Any], fills: list[dict[str, Any]]
) -> dict[str, Any]:
    """Update cash and positions from fills. Positions: qty and avg_price (weighted). Returns new state (copy).

    Item 41: Cash accumulation uses Decimal arithmetic to avoid float drift across many fills.
    Cash is converted back to float on output so the rest of the system is unaffected.
    """
    # Use Decimal for cash to prevent accumulated float rounding error (Item 41)
    _cash_d = Decimal(str(state.get("cash") or 0))
    # dict[str, Any]: values are heterogeneous JSON shapes (None/str/dict/list);
    # without the annotation mypy infers a join-union that breaks every
    # indexed access below. Annotation only — no runtime change.
    out: dict[str, Any] = {
        "schema_version": state.get("schema_version") or SCHEMA_VERSION,
        "updated_utc": state.get("updated_utc"),
        "cash": None,  # set from _cash_d at the end
        "positions": {k: dict(v) for k, v in (state.get("positions") or {}).items()},
        "equity_curve": list(state.get("equity_curve") or []),
    }
    # entry_ts for NEW position opens (zombie-killer hold-time base): prefer a
    # fill-provided timestamp; fall back to apply time. EOD paper fills carry
    # no timestamp today, so the fallback (== booking time) is the honest base.
    _apply_ts_iso = datetime.now(tz=timezone.utc).isoformat()
    for f in fills:
        symbol = f.get("symbol")
        if not symbol:
            continue
        symbol = str(symbol).strip()
        side = str(f.get("side", "BUY")).upper()
        qty = float(f.get("qty", 0))
        price = float(f.get("price", 0))
        if qty <= 0 or price <= 0:
            continue
        _fill_ts = str(f.get("timestamp") or "") or _apply_ts_iso
        pos = out["positions"].setdefault(
            symbol, {"qty": 0.0, "avg_price": 0.0, "hwm": 0.0}
        )
        pos_qty = pos["qty"]
        pos_avg = pos["avg_price"]
        pos_hwm = pos.get("hwm", pos_avg)
        # Preserved on same-side adds/partials; replaced on open-from-zero and
        # side flips (a flip IS a new position); dropped with the position.
        pos_entry_ts = pos.get("entry_ts")
        _notional = Decimal(str(qty)) * Decimal(str(price))
        # F-A-1 fix: explicit cases distinguishing long/short and cover/flip.
        # Cash flow rule: BUY always debits qty*price, SELL always credits qty*price.
        # Position rule: track signed qty, with avg_price as weighted avg of opens
        # on the same side. When covering or flipping, the OPPOSITE-side leg's avg
        # is replaced (not blended) at the fill price.
        if side == "BUY":
            _cash_d -= _notional
            if pos_qty >= 0:
                # Long addition (or opening from zero)
                new_qty = pos_qty + qty
                new_avg = (
                    (pos_avg * pos_qty + price * qty) / new_qty if new_qty else 0.0
                )
                new_hwm = max(pos_hwm, price) if pos_hwm > 0 else price
                out["positions"][symbol] = _with_entry_ts(
                    {"qty": new_qty, "avg_price": new_avg, "hwm": new_hwm},
                    _fill_ts if pos_qty == 0 else pos_entry_ts,
                )
            else:
                # Covering short. cover_qty bounded by short size; remainder flips long.
                short_open = -pos_qty  # positive
                cover_qty = min(qty, short_open)
                remaining_buy = qty - cover_qty
                new_short = pos_qty + cover_qty  # less negative or 0
                if new_short < 0:
                    # Still short, qty reduced; short avg preserved
                    out["positions"][symbol] = _with_entry_ts(
                        {"qty": new_short, "avg_price": pos_avg, "hwm": pos_hwm},
                        pos_entry_ts,
                    )
                elif new_short == 0 and remaining_buy == 0:
                    # Short fully covered, no overflow
                    out["positions"].pop(symbol, None)
                else:
                    # Short fully covered, overflow opens new long at fill price
                    out["positions"][symbol] = _with_entry_ts(
                        {"qty": remaining_buy, "avg_price": price, "hwm": price},
                        _fill_ts,
                    )
        else:  # SELL
            _cash_d += _notional
            if pos_qty > 0:
                new_qty = pos_qty - qty
                if new_qty > 0:
                    # Partial sell of long; avg/hwm preserved
                    out["positions"][symbol] = _with_entry_ts(
                        {"qty": new_qty, "avg_price": pos_avg, "hwm": pos_hwm},
                        pos_entry_ts,
                    )
                elif new_qty == 0:
                    out["positions"].pop(symbol, None)
                else:
                    # Oversell: close long + open short for the overflow
                    short_qty = qty - pos_qty  # positive overflow
                    out["positions"][symbol] = _with_entry_ts(
                        {"qty": -short_qty, "avg_price": price, "hwm": price},
                        _fill_ts,
                    )
            else:
                # Opening or adding to short (pos_qty <= 0)
                new_qty = pos_qty - qty  # more negative
                if pos_qty == 0:
                    new_avg = price
                    new_hwm = price
                else:
                    # Weighted avg of short opens: prior short_qty * prior_avg + new qty * price
                    short_open_prior = -pos_qty
                    short_open_new = -new_qty
                    new_avg = (
                        pos_avg * short_open_prior + price * qty
                    ) / short_open_new
                    new_hwm = max(pos_hwm, price) if pos_hwm > 0 else price
                out["positions"][symbol] = _with_entry_ts(
                    {"qty": new_qty, "avg_price": new_avg, "hwm": new_hwm},
                    _fill_ts if pos_qty == 0 else pos_entry_ts,
                )
    # W18 (2026-07-21, GESAMTBEWERTUNG): dust sweep. Position qty is float
    # arithmetic; partial closes of fractional positions leave residues like
    # 7.1e-15 (live-verified: CVX/KO/WMT in the pilot ledger) because a
    # position is only popped on an EXACT zero. Sweep sub-epsilon positions
    # after all fills are applied. Epsilon 1e-9 is far below the live sync
    # qty_tol (0.001 in position_sync.sync_positions_from_broker), so
    # dropping dust can never create a reportable ledger-vs-broker mismatch.
    _DUST_EPS = 1e-9
    for _sym in [
        s for s, p in out["positions"].items() if abs(p.get("qty", 0.0)) < _DUST_EPS
    ]:
        out["positions"].pop(_sym, None)
    out["cash"] = float(_cash_d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    return out


def append_equity_curve_deduped(
    state: dict[str, Any], utc_iso: str, equity: float
) -> None:
    """Append equity curve entry, deduplicating by date (not full timestamp).

    Prevents duplicate entries if the same day is run twice.
    """
    curve = state.setdefault("equity_curve", [])
    date_str = utc_iso[:10]  # Extract YYYY-MM-DD
    # Check if this date already exists
    for i, entry in enumerate(curve):
        existing_date = str(entry.get("utc", ""))[:10]
        if existing_date == date_str:
            # Replace existing entry for this date
            curve[i] = {"utc": utc_iso, "equity": equity}
            return
    # New date — append
    curve.append({"utc": utc_iso, "equity": equity})


def mark_to_market_equity(state: dict[str, Any], prices_latest: pd.DataFrame) -> float:
    """Compute equity = cash + sum(position qty * latest price).

    Also updates HWM (high-water mark) for each position in-place.

    Day-5-bug fix (2026-05-15): when a held position is NOT in prices_latest
    (e.g. universe shrank but position still held), previously fell back to
    px=0.0 silently → equity_curve underestimated by full position notional.
    Now falls back to position avg_price as last-resort MTM and logs a WARNING
    so the operator notices the data gap. Better than zero but operator must
    re-add the symbol to the universe to restore accurate MTM.
    """
    cash = float(state.get("cash", 0))
    positions = state.get("positions") or {}
    if not positions:
        return cash
    price_col = "close" if "close" in prices_latest.columns else "price"
    if "symbol" not in prices_latest.columns or price_col not in prices_latest.columns:
        return cash
    price_map = prices_latest.set_index("symbol")[price_col].to_dict()
    mtm = cash
    missing_price_syms: list[tuple[str, float, float]] = []
    for sym, pos in positions.items():
        qty = float(pos.get("qty", 0))
        if qty == 0:
            continue
        p = price_map.get(sym)
        try:
            px = float(p) if p is not None else None
        except (TypeError, ValueError):
            px = None
        if px is None or px <= 0:
            # Fallback to avg_price (cost basis) so equity doesn't silently
            # drop by the missing position's notional. Operator-visible via WARN.
            fallback_px = float(pos.get("avg_price", 0))
            missing_price_syms.append((sym, qty, fallback_px))
            px = fallback_px
        mtm += qty * px
        # Update HWM for trailing stop support (only on real market price)
        if p is not None and px > 0:
            current_hwm = float(pos.get("hwm", 0))
            if px > current_hwm:
                pos["hwm"] = px

    if missing_price_syms:
        details = ", ".join(
            f"{s}(qty={q:.2f}, used_avg=${p:.2f})" for s, q, p in missing_price_syms
        )
        logger.warning(
            "[mark_to_market] %d held position(s) missing from prices_latest — "
            "falling back to avg_price for MTM (equity may drift from broker truth). "
            "Re-add to universe/watchlist to fix: %s",
            len(missing_price_syms),
            details,
        )
    return mtm


def write_ledger_snapshot(
    output_dir: str | Path,
    state: dict[str, Any],
    equity: float,
) -> Path:
    """Write output_dir/ledger_snapshot.json (schema paper.ledger_snapshot.v1)."""
    out_dir = Path(output_dir)
    path = out_dir / "ledger_snapshot.json"
    payload: dict[str, Any] = {
        "schema_version": "paper.ledger_snapshot.v1",
        "cash": state.get("cash"),
        "positions": state.get("positions"),
        "equity": equity,
        "updated_utc": state.get("updated_utc")
        or datetime.now(timezone.utc).isoformat(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)
    return path
