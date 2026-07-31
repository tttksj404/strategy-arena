# Wave-35 genome: identical to genome30 except that the symbols gene becomes a BREADTH tier.
#
# genome30's symbols gene enumerates five hardcoded subsets of {BTC, ETH, SOL}, which cannot express
# a 20-symbol universe. Here the gene selects how MANY of the volume-ranked symbols to trade
# (1/3/5/10/20) -- the same formulation wave13 used to find the carry breadth curve (BTC+ETH 11% ->
# top30 14.7% -> top100 20.3% -> top200 22.0% peak). Breadth becomes something the search discovers
# instead of something hardcoded.
#
# Every leverage/stop constraint is inherited from genome30 UNCHANGED and re-validated here:
#   lev = risk_frac / stop_pct, no clipping, 1x <= lev <= 20x
#   stop_pct <= 0.9 * (1/lev - 0.005)      (stop strictly interior to the liquidation band)
# Validation is duplicated rather than subclassed because genome30.Genome is a frozen dataclass whose
# validate() hardcodes its own SYMBOL_SETS membership check; duck typing keeps engine30 working
# without touching a module four waves of results depend on.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from research.wave30_qd.genome30 import (
    CONCURRENCY_CHOICES,
    COOLDOWN_CHOICES,
    ENTRY_THRESHOLD_RANGE,
    LEV_CAP,
    LOOKBACK_CHOICES,
    MAINT_MARGIN,
    MAX_HOLD_CHOICES,
    RISK_FRAC_RANGE,
    SIGNAL_FAMILIES,
    SLEEVE_FRACTION_CHOICES,
    STOP_BAND_MARGIN,
    STOP_PCT_RANGE,
    TARGET_R_RANGE,
    InvalidGenomeError,
)

BREADTH_CHOICES: Final = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class Genome35:
    signal_family: str
    lookback_bars: int
    entry_threshold: float
    stop_pct: float
    target_r: float
    trail_enabled: bool
    risk_frac: float
    max_hold_bars: int
    allow_short: bool
    symbols: tuple[str, ...]
    max_concurrent: int
    cooldown_bars_after_loss: int
    sleeve_fraction: float

    @property
    def leverage(self) -> float:
        return float(self.risk_frac / self.stop_pct)

    @property
    def liquidation_band(self) -> float:
        return max(0.0, 1.0 / self.leverage - MAINT_MARGIN)

    def validate(self) -> "Genome35":
        if self.signal_family not in SIGNAL_FAMILIES:
            raise InvalidGenomeError(f"unknown signal_family {self.signal_family!r}")
        if self.lookback_bars not in LOOKBACK_CHOICES:
            raise InvalidGenomeError(f"lookback_bars {self.lookback_bars} outside frozen set")
        if self.max_hold_bars not in MAX_HOLD_CHOICES:
            raise InvalidGenomeError(f"max_hold_bars {self.max_hold_bars} outside frozen set")
        if not self.symbols or len(self.symbols) not in BREADTH_CHOICES:
            raise InvalidGenomeError(f"universe breadth {len(self.symbols)} outside frozen tiers")
        if len(set(self.symbols)) != len(self.symbols):
            raise InvalidGenomeError("duplicate symbols in universe")
        if self.max_concurrent not in CONCURRENCY_CHOICES:
            raise InvalidGenomeError(f"max_concurrent {self.max_concurrent} outside frozen set")
        if self.cooldown_bars_after_loss not in COOLDOWN_CHOICES:
            raise InvalidGenomeError("cooldown outside frozen set")
        if self.sleeve_fraction not in SLEEVE_FRACTION_CHOICES:
            raise InvalidGenomeError("sleeve_fraction outside frozen set")
        if not STOP_PCT_RANGE[0] <= self.stop_pct <= STOP_PCT_RANGE[1]:
            raise InvalidGenomeError(f"stop_pct {self.stop_pct} outside range")
        if not TARGET_R_RANGE[0] <= self.target_r <= TARGET_R_RANGE[1]:
            raise InvalidGenomeError(f"target_r {self.target_r} outside range")
        if not RISK_FRAC_RANGE[0] <= self.risk_frac <= RISK_FRAC_RANGE[1]:
            raise InvalidGenomeError(f"risk_frac {self.risk_frac} outside range")
        if not ENTRY_THRESHOLD_RANGE[0] <= self.entry_threshold <= ENTRY_THRESHOLD_RANGE[1]:
            raise InvalidGenomeError("entry_threshold outside range")
        if self.leverage > LEV_CAP + 1e-9:
            raise InvalidGenomeError(f"leverage {self.leverage:.4f} exceeds cap {LEV_CAP}")
        if self.leverage < 1.0 - 1e-9:
            raise InvalidGenomeError(f"leverage {self.leverage:.4f} below 1x")
        if self.stop_pct > STOP_BAND_MARGIN * self.liquidation_band + 1e-12:
            raise InvalidGenomeError(
                f"stop_pct {self.stop_pct:.5f} not interior to liquidation band {self.liquidation_band:.5f}"
            )
        return self

    @property
    def is_feasible(self) -> bool:
        try:
            self.validate()
        except InvalidGenomeError:
            return False
        return True

    def key(self) -> tuple:
        return (
            self.signal_family,
            self.lookback_bars,
            round(self.entry_threshold, 4),
            round(self.stop_pct, 6),
            round(self.target_r, 4),
            self.trail_enabled,
            round(self.risk_frac, 6),
            self.max_hold_bars,
            self.allow_short,
            self.symbols,
            self.max_concurrent,
            self.cooldown_bars_after_loss,
            self.sleeve_fraction,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_family": self.signal_family,
            "lookback_bars": self.lookback_bars,
            "entry_threshold": round(self.entry_threshold, 6),
            "stop_pct": round(self.stop_pct, 8),
            "target_r": round(self.target_r, 6),
            "trail_enabled": self.trail_enabled,
            "risk_frac": round(self.risk_frac, 8),
            "max_hold_bars": self.max_hold_bars,
            "allow_short": self.allow_short,
            "symbols": list(self.symbols),
            "universe_breadth": len(self.symbols),
            "max_concurrent": self.max_concurrent,
            "cooldown_bars_after_loss": self.cooldown_bars_after_loss,
            "sleeve_fraction": self.sleeve_fraction,
            "derived_leverage": round(self.leverage, 6),
            "derived_liquidation_band": round(self.liquidation_band, 6),
        }
