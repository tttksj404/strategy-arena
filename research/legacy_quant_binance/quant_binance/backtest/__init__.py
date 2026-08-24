"""Replay and oracle utilities."""

from .comparison import (
    StrategyComparisonReport,
    StrategyComparisonResult,
    compare_strategies,
    render_compact_report,
    write_comparison_report,
)
from .metrics import VirtualPerformance, virtual_performance_from_decisions
from .oracle import OracleReport, OracleSegment, compare_decisions_to_oracle, load_oracle
from .replay import ReplayResult, run_replay

__all__ = [
    "OracleReport",
    "OracleSegment",
    "ReplayResult",
    "StrategyComparisonReport",
    "StrategyComparisonResult",
    "VirtualPerformance",
    "compare_strategies",
    "compare_decisions_to_oracle",
    "load_oracle",
    "render_compact_report",
    "run_replay",
    "virtual_performance_from_decisions",
    "write_comparison_report",
]
