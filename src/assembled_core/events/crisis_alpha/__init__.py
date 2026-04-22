"""Crisis-Alpha v1 subsystem — M5.

Separate crisis trading layer that activates on geo-risk escalation,
confirmed by multi-source news evidence and market stress.
"""

from __future__ import annotations

from .baskets import get_basket_by_name, get_basket_symbols, get_baskets
from .context import CrisisAlphaContext
from .entry import generate_crisis_entry
from .exit_rules import (
    check_break_even,
    check_deactivation_triggers,
    check_no_overnight,
    check_time_stop,
    get_positions_to_exit,
)
from .gates import (
    check_daily_loss_gate,
    check_evidence_gate,
    check_evidence_grade_gate_from_ctx,
    check_health_gate,
    check_market_stress_gate,
    check_social_only_guard,
    check_source_gate,
    run_all_activation_gates,
)
from .pipeline import run_crisis_alpha_pipeline
from .risk_budget import (
    apply_risk_budget,
    apply_weight_caps,
    check_daily_loss,
    check_gross_exposure,
    scale_to_gross_cap,
)
from .state_machine import (
    CrisisStateRecord,
    compute_next_crisis_state,
    load_crisis_state,
    save_crisis_state,
)

__all__ = [
    "CrisisAlphaContext",
    "CrisisStateRecord",
    "apply_risk_budget",
    "apply_weight_caps",
    "check_break_even",
    "check_daily_loss",
    "check_daily_loss_gate",
    "check_deactivation_triggers",
    "check_evidence_gate",
    "check_evidence_grade_gate_from_ctx",
    "check_gross_exposure",
    "check_health_gate",
    "check_market_stress_gate",
    "check_no_overnight",
    "check_social_only_guard",
    "check_source_gate",
    "check_time_stop",
    "compute_next_crisis_state",
    "generate_crisis_entry",
    "get_basket_by_name",
    "get_basket_symbols",
    "get_baskets",
    "get_positions_to_exit",
    "load_crisis_state",
    "run_all_activation_gates",
    "run_crisis_alpha_pipeline",
    "save_crisis_state",
    "scale_to_gross_cap",
]
