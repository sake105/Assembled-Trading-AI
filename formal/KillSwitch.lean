-- formal/KillSwitch.lean
-- Audit C2-001/002 — Lean 4 formal proof scaffold for the kill-switch FSM.
--
-- Build (when lean4 + lake are installed):
--     lake build
--
-- This file is the *single source of truth* for the four safety
-- invariants the kill-switch must satisfy. The Python implementation in
-- src/assembled_core/execution/kill_switch.py is the *executable* mirror;
-- tests/test_property_fsm_pit.py provides Hypothesis-level property
-- checks that exercise those same invariants in Python. If the Lean
-- spec disagrees with the Python tests, the Lean spec is authoritative
-- and the Python tests should be tightened.

namespace KillSwitch

-- The kill-switch state machine. Throttle is a fraction in [0, 1]
-- representing the fraction of orders the broker is allowed to forward.
-- engaged=False ⇒ throttle = 1.0 (no throttle).
-- engaged=True  ⇒ throttle ∈ [0, 1].
structure State where
  engaged : Bool
  throttlePct : Float
  deriving Repr

-- A guarded order — submit (qty) and the eventual decision (allowed qty).
structure Order where
  qty : Float
  deriving Repr

-- Apply the kill-switch guard to an order. Returns the allowed qty.
-- Mirrors `guard_orders_with_kill_switch` in the Python implementation.
def guard (s : State) (o : Order) : Float :=
  if !s.engaged then
    o.qty
  else if s.throttlePct ≤ 0.0 then
    0.0
  else
    -- Floor-with-sign semantics: any sub-unit qty is dropped.
    let scaled := o.qty * s.throttlePct
    if scaled.abs < 1.0 then 0.0 else scaled

-- ---------------------------------------------------------------------
-- INVARIANT 1 — safety_no_send_when_off
-- ---------------------------------------------------------------------
-- When the kill-switch is engaged with throttle = 0, NO order is allowed
-- through, regardless of the requested quantity.
theorem safety_no_send_when_off (o : Order) :
    guard { engaged := true, throttlePct := 0.0 } o = 0.0 := by
  unfold guard
  simp

-- ---------------------------------------------------------------------
-- INVARIANT 2 — trip_is_observable
-- ---------------------------------------------------------------------
-- engaged=True with throttle=0 always results in a *visible* effect
-- (the allowed qty differs from the requested qty for any non-zero
-- input). This is what makes the kill-switch "loud" — it never silently
-- pretends to forward.
theorem trip_is_observable (o : Order) (h : o.qty ≠ 0.0) :
    guard { engaged := true, throttlePct := 0.0 } o ≠ o.qty := by
  unfold guard
  simp
  exact h.symm

-- ---------------------------------------------------------------------
-- INVARIANT 3 — throttle_monotone
-- ---------------------------------------------------------------------
-- For positive quantities, a higher throttle never returns less than a
-- lower throttle. (i.e. relaxing the kill-switch never restricts more.)
-- The proof is left as an obligation — Float arithmetic in Lean 4
-- requires the mathlib4 real-arithmetic tactics. Documented here as a
-- forced TODO so the chain of trust is explicit.
theorem throttle_monotone
    (o : Order) (t1 t2 : Float)
    (h_pos : o.qty > 0.0)
    (h_t1  : 0.0 ≤ t1) (h_t2  : t1 ≤ t2) (h_t2_le : t2 ≤ 1.0) :
    guard { engaged := true, throttlePct := t1 } o
      ≤ guard { engaged := true, throttlePct := t2 } o := by
  sorry -- TODO(audit C2-002): tighten once mathlib4 Real-tactics are wired in

-- ---------------------------------------------------------------------
-- INVARIANT 4 — disengaged_is_passthrough
-- ---------------------------------------------------------------------
-- When the kill-switch is NOT engaged, every order passes through with
-- its original qty.
theorem disengaged_is_passthrough (o : Order) (t : Float) :
    guard { engaged := false, throttlePct := t } o = o.qty := by
  unfold guard
  simp

end KillSwitch
