"""Crisis-Alpha v1 subsystem — M5.

Separate crisis trading layer that activates on geo-risk escalation,
confirmed by multi-source news evidence and market stress.

Package layout:
    context.py      — CrisisAlphaContext input contract
    state_machine.py — persistent WATCH/ACTIVE/COOLDOWN/PAUSE state machine
    gates.py        — activation and deactivation gate checks
    baskets.py      — ETF basket definitions (crisis instruments)
    entry.py        — simple rule-based entry signals
    risk_budget.py  — daily loss guard and position limits
    exit_rules.py   — exits (time stop, break-even, trail) and deactivation triggers
    pipeline.py     — orchestrates the above into a single run function
"""
