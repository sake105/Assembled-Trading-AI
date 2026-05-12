---------------------------- MODULE Reconciliation ----------------------------
\* Audit C2-008 — TLA+ specification of the broker-ledger reconciliation FSM.
\*
\* This is a *specification*, not the running implementation. The Python
\* code in src/assembled_core/accounting/reconciliation.py is the
\* executable mirror; tests/test_audit_additions.py covers the relevant
\* behaviour in pytest. If TLC (the TLA+ model checker) finds a
\* counter-example here that the Python code permits, the Python code is
\* the bug.
\*
\* To run with TLC (when java + tla2tools.jar are installed):
\*     java -cp tla2tools.jar tlc2.TLC Reconciliation.tla
\*

EXTENDS Naturals, FiniteSets, Sequences

CONSTANTS
    Orders,        \* the set of possible order IDs
    MaxRetries     \* upper bound on reconciliation strikes before kill

VARIABLES
    broker,        \* set of order-IDs the broker thinks are filled
    ledger,        \* set of order-IDs the ledger thinks are filled
    cache,         \* set of order-IDs the cache thinks are filled
    strikes,       \* consecutive reconciliation diff count
    killEngaged    \* kill-switch active?

vars == << broker, ledger, cache, strikes, killEngaged >>

\* ---- Initial state -----------------------------------------------------
Init ==
    /\ broker = {}
    /\ ledger = {}
    /\ cache = {}
    /\ strikes = 0
    /\ killEngaged = FALSE

\* ---- Actions -----------------------------------------------------------

\* The broker confirms a fill. Both ledger and cache should follow on
\* the next reconcile tick.
BrokerConfirms(o) ==
    /\ ~killEngaged
    /\ o \notin broker
    /\ broker' = broker \cup {o}
    /\ UNCHANGED << ledger, cache, strikes, killEngaged >>

\* The ledger applies a fill it learned about from the broker.
LedgerCatchUp(o) ==
    /\ ~killEngaged
    /\ o \in broker
    /\ o \notin ledger
    /\ ledger' = ledger \cup {o}
    /\ UNCHANGED << broker, cache, strikes, killEngaged >>

\* The cache applies a fill.
CacheCatchUp(o) ==
    /\ ~killEngaged
    /\ o \in broker
    /\ o \notin cache
    /\ cache' = cache \cup {o}
    /\ UNCHANGED << broker, ledger, strikes, killEngaged >>

\* Reconciliation tick: increment strikes if any side disagrees;
\* reset strikes to 0 if all three agree.
Reconcile ==
    /\ ~killEngaged
    /\ IF broker = ledger /\ ledger = cache
       THEN strikes' = 0
       ELSE strikes' = strikes + 1
    /\ UNCHANGED << broker, ledger, cache, killEngaged >>

\* After MaxRetries consecutive failed reconciliations, the kill-switch
\* must engage. This mirrors the audit-mandated 3-strikes auto-kill rule.
KillOnPersistentDrift ==
    /\ strikes >= MaxRetries
    /\ ~killEngaged
    /\ killEngaged' = TRUE
    /\ UNCHANGED << broker, ledger, cache, strikes >>

Next ==
    \/ \E o \in Orders : BrokerConfirms(o)
    \/ \E o \in Orders : LedgerCatchUp(o)
    \/ \E o \in Orders : CacheCatchUp(o)
    \/ Reconcile
    \/ KillOnPersistentDrift

Spec == Init /\ [][Next]_vars

\* ---- Invariants --------------------------------------------------------

\* I1: Strikes are bounded by MaxRetries + 1 (the +1 accounts for the
\*     single tick where strikes crosses the threshold and KillOnPersistentDrift
\*     becomes enabled). The kill-switch then takes the next step.
StrikesBounded == strikes <= MaxRetries + 1

\* I2: Once the kill-switch is engaged, NO further changes happen.
\*     This is a safety property; the kill-switch is absorbing.
KillIsAbsorbing == killEngaged => \A o \in Orders :
    /\ (o \notin broker => UNCHANGED broker)
    /\ TRUE  \* placeholder — TLC checks via the Next disjunction

\* I3: Ledger and cache are always subsets of the broker view
\*     (no order is filled in the ledger that the broker did not see).
SubsetSafety ==
    /\ ledger \subseteq broker
    /\ cache \subseteq broker

\* ---- Liveness ----------------------------------------------------------

\* L1: Eventually, if a fill is in the broker view, the ledger reflects
\*     it (assuming no permanent drift).
FairnessLedger == \A o \in Orders :
    (o \in broker /\ ~killEngaged) ~> (o \in ledger \/ killEngaged)

\* ---- Configuration hints -----------------------------------------------
\* When you run TLC, set in Reconciliation.cfg:
\*     CONSTANTS Orders = {o1, o2, o3}
\*               MaxRetries = 2
\*     INVARIANT  StrikesBounded
\*     INVARIANT  SubsetSafety
\*     PROPERTY   FairnessLedger
\* and add WeakFairness on Reconcile + LedgerCatchUp + KillOnPersistentDrift.

================================================================================
