"""Tunable parameter definitions with bounds and safety constraints."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TunableParam:
    path: tuple[str, ...]       # Nested config key path
    min_bound: float
    max_bound: float
    step_pct: float             # Max change per cycle (e.g. 0.05 = +-5%)
    min_trades: int             # Minimum trades before tuning
    risk_tier: str = "medium"   # "low" | "medium" | "high"
    description: str = ""

    def clamp(self, value: float) -> float:
        return max(self.min_bound, min(self.max_bound, value))

    def max_delta(self, current: float) -> float:
        return abs(current * self.step_pct)


# Tier 1: Entry thresholds (highest impact)
TIER1_ENTRY = [
    TunableParam(
        path=("mode_thresholds", "futures_score_min"),
        min_bound=60.0, max_bound=85.0, step_pct=0.05, min_trades=50,
        risk_tier="medium", description="futures entry score threshold",
    ),
    TunableParam(
        path=("mode_thresholds", "spot_score_min"),
        min_bound=40.0, max_bound=70.0, step_pct=0.05, min_trades=50,
        risk_tier="medium", description="spot entry score threshold",
    ),
    TunableParam(
        path=("futures_exposure", "min_entry_net_edge_bps"),
        min_bound=-15.0, max_bound=15.0, step_pct=0.05, min_trades=50,
        risk_tier="medium", description="minimum net edge for futures entry",
    ),
]

# Tier 2: Stop/Exit parameters
TIER2_EXIT = [
    TunableParam(
        path=("sizing", "atr_multiple_for_stop"),
        min_bound=1.2, max_bound=3.5, step_pct=0.05, min_trades=30,
        risk_tier="medium", description="ATR multiple for stop distance",
    ),
    TunableParam(
        path=("exit_rules", "futures_max_holding_minutes"),
        min_bound=30.0, max_bound=480.0, step_pct=0.05, min_trades=30,
        risk_tier="medium", description="max futures holding time",
    ),
    TunableParam(
        path=("exit_rules", "score_drop_exit_buffer"),
        min_bound=0.5, max_bound=15.0, step_pct=0.05, min_trades=30,
        risk_tier="medium", description="score drop buffer for exit trigger",
    ),
]

# Tier 3: Position sizing (high risk)
TIER3_SIZING = [
    TunableParam(
        path=("risk", "per_trade_equity_risk"),
        min_bound=0.002, max_bound=0.01, step_pct=0.03, min_trades=50,
        risk_tier="high", description="equity % risked per trade",
    ),
]

# Tier 4: Signal weights (needs most data)
TIER4_WEIGHTS = [
    TunableParam(
        path=("weights", "trend_strength"),
        min_bound=0.15, max_bound=0.50, step_pct=0.03, min_trades=80,
        risk_tier="high", description="trend strength weight",
    ),
    TunableParam(
        path=("weights", "volume_confirmation"),
        min_bound=0.10, max_bound=0.35, step_pct=0.03, min_trades=80,
        risk_tier="high", description="volume confirmation weight",
    ),
    TunableParam(
        path=("weights", "liquidity_score"),
        min_bound=0.10, max_bound=0.35, step_pct=0.03, min_trades=80,
        risk_tier="high", description="liquidity score weight",
    ),
]

ALL_TUNABLE_PARAMS = TIER1_ENTRY + TIER2_EXIT + TIER3_SIZING + TIER4_WEIGHTS
