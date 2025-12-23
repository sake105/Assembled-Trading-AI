"""Paper Trading Track Module.

This module provides the paper track runner for simulating trading strategies
in a stateful, daily-execution manner without real orders.
"""

from src.assembled_core.paper.paper_track import (
    PaperTrackConfig,
    PaperTrackDayResult,
    PaperTrackState,
    load_paper_state,
    run_paper_day,
    save_paper_state,
    write_paper_day_outputs,
)

__all__ = [
    "PaperTrackConfig",
    "PaperTrackState",
    "PaperTrackDayResult",
    "load_paper_state",
    "save_paper_state",
    "run_paper_day",
    "write_paper_day_outputs",
]
