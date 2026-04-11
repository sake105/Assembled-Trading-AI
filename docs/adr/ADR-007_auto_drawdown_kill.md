# ADR-007: Automatic Drawdown Kill Switch

**Status:** proposed
**Date:** 2026-04-11
**Deciders:** trading-ops, risk

## Context

Manual drawdown response is unreliable. By the time an operator
notices a deteriorating account and opens a chart, the drawdown has
often already passed a level the risk policy would have halted at.
The existing `pre_trade_checks` layer scales sizing by a drawdown
factor but does not engage the kill switch — so new orders get
smaller, existing positions keep riding the drawdown, and there is
no hard stop.

We also need the halt to be proportional. A single hard halt at
18% is a cliff: below that threshold the system acts as if nothing
is wrong, above it all trading stops. A graduated response is
closer to how a human operator would actually intervene.

## Decision

A new phase in `pipeline/trading_cycle.py` (phase 18.5) evaluates
the current drawdown against three levels and escalates the
three-tier kill switch (ADR-003) accordingly:

- `-8% drawdown` → tier 1 soft throttle (50%)
- `-12% drawdown` → tier 2 hard throttle (80%)
- `-18% drawdown` → tier 3 full kill (100%)

Thresholds are configurable per policy. The drawdown is computed
from the ledger equity vs its running high-water-mark. Escalation
is one-way within a session: once tier 3 is hit, the switch stays
at tier 3 until a human resets it. Recovery to a lower tier is
**not** automatic; the safety invariant in `CLAUDE.md` forbids
automatic recovery after a kill.

## Consequences

### Positive

- Drawdown response is proportional and automatic; no need for an
  operator to be watching a chart in real time.
- The three thresholds map directly onto the three throttle tiers
  from ADR-003, so there is no new primitive to learn.
- Post-mortem is simple: the audit log shows the ledger equity at
  each transition and the exact time.
- Compatible with runbook 04 (drawdown limit hit) which assumes the
  system has already acted by the time an operator arrives.

### Negative

- A false drawdown reading — e.g. a bad price in the mark-to-market
  leg — can trigger a false kill. Upstream freshness and PIT gates
  need to be trusted for this phase to be trusted.
- The thresholds are configured in one place but interact with vol
  targeting, correlation guard and group exposure caps. A very
  tight portfolio can hit a tier 1 threshold in a single bad day,
  which may or may not be the intended behaviour.
- Recovery requires human action. A weekend drawdown event can
  therefore keep the system halted until Monday, which is the
  intended trade-off.

## Alternatives Considered

- **Manual-only**: rejected. See context.
- **Time-decay recovery** (e.g. auto-unhalt after 24h of no new
  drawdown): rejected for now. Safety invariant wins.
- **Single hard halt at one threshold**: rejected. See context
  ("cliff").
- **Per-strategy drawdown**: deferred. Currently only one strategy
  is live at a time; when v1 and v2 run in parallel this ADR will
  need an addendum.
