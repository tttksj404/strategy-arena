# Wave-30 genome. The single structural difference from every earlier wave's genome
# (wave21/23/27/28) is that LEVERAGE IS NOT A GENE -- it is derived:
#
#     lev      = clip(risk_frac / stop_pct, 1.0, LEV_CAP)
#     liq_band = 1/lev - MAINT
#     hard constraint: stop_pct <= STOP_BAND_MARGIN * liq_band
#
# wave4 and wave29 both multiplied a leverage factor onto a strategy whose stop had already
# been fixed, which is why wave29's V1 was liquidated on its first trade at 5x: its measured
# MAE routinely exceeded the band. Deriving leverage from the stop makes the stop provably
# interior to the liquidation band, so liquidation can only come from a GAP THROUGH the stop
# -- which engine30 then measures from real 1h bars instead of assuming it away.
#
# Genomes violating the constraint are rejected before evaluation (fail-closed): they are
# never silently repaired, because repairing would let the search drift into the infeasible
# region and report a leverage it could not actually run.

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np

LEV_CAP: Final = 20.0  # user-specified hard ceiling (SPEC.md 자본 규약)
MAINT_MARGIN: Final = 0.005  # Bitget USDT-M maintenance margin, same constant as wave29_lev10
STOP_BAND_MARGIN: Final = 0.9  # stop must sit at most 90% of the way to the liquidation band

SIGNAL_FAMILIES: Final = ("breakout", "momentum", "reversion", "funding_breakout")
LOOKBACK_CHOICES: Final = (6, 12, 24, 48, 72, 120, 168, 336)
MAX_HOLD_CHOICES: Final = (6, 12, 24, 48, 96, 168, 336, 720)
SYMBOL_SETS: Final = (
    ("BTCUSDT",),
    ("ETHUSDT",),
    ("SOLUSDT",),
    ("BTCUSDT", "ETHUSDT"),
    ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
)
CONCURRENCY_CHOICES: Final = (1, 2, 3)
COOLDOWN_CHOICES: Final = (0, 6, 24, 72)
SLEEVE_FRACTION_CHOICES: Final = (0.10, 0.25, 0.50, 0.75, 1.00)

STOP_PCT_RANGE: Final = (0.003, 0.060)
TARGET_R_RANGE: Final = (1.0, 8.0)
RISK_FRAC_RANGE: Final = (0.01, 0.40)
ENTRY_THRESHOLD_RANGE: Final = (0.0, 3.0)


class InvalidGenomeError(ValueError):
    """Genome violates a frozen structural constraint. Fail-closed: never auto-repaired."""


@dataclass(frozen=True)
class Genome:
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
        """Exactly risk_frac / stop_pct, with NO clipping.

        Clipping was the original implementation and it was wrong: risk_frac=0.40 with a 1%
        stop implies 40x, and silently clamping that to 20x would mean the genome reported
        "I risk 40% at my stop" while actually risking 20%. Every gate and every archive cell
        is indexed by leverage, so a quietly false leverage would corrupt the whole map. The
        ratio is therefore left exact and validate() REJECTS anything outside [1x, LEV_CAP]
        -- fail-closed, consistent with the stop/band constraint. As a result risk_frac is
        always a true statement: a stop-out costs exactly risk_frac of the position's base.
        """
        return float(self.risk_frac / self.stop_pct)

    @property
    def liquidation_band(self) -> float:
        return max(0.0, 1.0 / self.leverage - MAINT_MARGIN)

    def validate(self) -> "Genome":
        if self.signal_family not in SIGNAL_FAMILIES:
            raise InvalidGenomeError(f"unknown signal_family {self.signal_family!r}")
        if self.lookback_bars not in LOOKBACK_CHOICES:
            raise InvalidGenomeError(f"lookback_bars {self.lookback_bars} outside frozen set")
        if self.max_hold_bars not in MAX_HOLD_CHOICES:
            raise InvalidGenomeError(f"max_hold_bars {self.max_hold_bars} outside frozen set")
        if self.symbols not in SYMBOL_SETS:
            raise InvalidGenomeError(f"symbols {self.symbols} outside frozen set")
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
            raise InvalidGenomeError(f"leverage {self.leverage:.4f} below 1x (stop wider than risk budget)")
        if self.stop_pct > STOP_BAND_MARGIN * self.liquidation_band + 1e-12:
            raise InvalidGenomeError(
                f"stop_pct {self.stop_pct:.5f} not interior to liquidation band "
                f"{self.liquidation_band:.5f} (limit {STOP_BAND_MARGIN * self.liquidation_band:.5f})"
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
        """Cache key with continuous genes rounded, so numerically identical genomes reached
        by different mutation paths share one evaluation (same trick as wave21's genome_key)."""
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
            "max_concurrent": self.max_concurrent,
            "cooldown_bars_after_loss": self.cooldown_bars_after_loss,
            "sleeve_fraction": self.sleeve_fraction,
            "derived_leverage": round(self.leverage, 6),
            "derived_liquidation_band": round(self.liquidation_band, 6),
        }


def _max_stop_for_risk(risk_frac: float) -> float:
    """Largest feasible stop_pct for a given risk_frac.

    A wider stop only ever lowers derived leverage, which widens the liquidation band faster
    than it widens the stop, so the band constraint never binds at the top. What DOES bind is
    lev >= 1: since lev = risk/stop, a stop wider than risk_frac would imply sub-1x leverage,
    which the exact (unclipped) leverage definition forbids.
    """
    return float(min(STOP_PCT_RANGE[1], risk_frac))


def _min_stop_for_risk(risk_frac: float) -> float:
    """Smallest feasible stop_pct for a given risk_frac.

    Feasibility couples the two continuous genes, so the sampler SOLVES the coupling rather
    than rejection-sampling it (rejection would over-sample whichever corner happens to be
    easy to hit and silently bias the leverage distribution). With lev = risk/stop the
    constraint stop <= 0.9*(1/lev - MAINT) expands to stop*(1 - 0.9/risk) <= -0.9*MAINT;
    since risk <= 0.40 < 0.9 the bracket is negative, so dividing flips the inequality into a
    LOWER bound stop >= 0.9*MAINT/(0.9/risk - 1). The separate cap lev <= LEV_CAP contributes
    a second lower bound stop >= risk/LEV_CAP, and the frozen range edge a third.
    """
    lower_from_cap = risk_frac / LEV_CAP
    bracket = STOP_BAND_MARGIN / risk_frac - 1.0
    lower_from_band = (STOP_BAND_MARGIN * MAINT_MARGIN / bracket) if bracket > 0 else np.inf
    return float(max(STOP_PCT_RANGE[0], lower_from_cap, lower_from_band))


def random_genome(rng: np.random.Generator) -> Genome:
    """Uniform over the frozen space, with stop_pct sampled from its feasible sub-interval so
    that every returned genome already satisfies validate() (no silent repair, no bias from
    rejection loops that would over-sample easy corners)."""
    for _ in range(200):
        risk_frac = float(rng.uniform(*RISK_FRAC_RANGE))
        low = _min_stop_for_risk(risk_frac)
        high = _max_stop_for_risk(risk_frac)
        if low > high:
            continue
        genome = Genome(
            signal_family=str(rng.choice(SIGNAL_FAMILIES)),
            lookback_bars=int(rng.choice(LOOKBACK_CHOICES)),
            entry_threshold=float(rng.uniform(*ENTRY_THRESHOLD_RANGE)),
            stop_pct=float(rng.uniform(low, high)),
            target_r=float(rng.uniform(*TARGET_R_RANGE)),
            trail_enabled=bool(rng.random() < 0.5),
            risk_frac=risk_frac,
            max_hold_bars=int(rng.choice(MAX_HOLD_CHOICES)),
            allow_short=bool(rng.random() < 0.5),
            symbols=SYMBOL_SETS[int(rng.integers(len(SYMBOL_SETS)))],
            max_concurrent=int(rng.choice(CONCURRENCY_CHOICES)),
            cooldown_bars_after_loss=int(rng.choice(COOLDOWN_CHOICES)),
            sleeve_fraction=float(rng.choice(SLEEVE_FRACTION_CHOICES)),
        )
        if genome.is_feasible:
            return genome
    raise InvalidGenomeError("failed to sample a feasible genome in 200 attempts")


def _jitter(value: float, low: float, high: float, rng: np.random.Generator, scale: float = 0.15) -> float:
    return float(np.clip(value + rng.normal(0.0, scale * (high - low)), low, high))


def mutate(genome: Genome, rng: np.random.Generator, rate: float = 0.25) -> Genome:
    """Gaussian jitter on continuous genes, uniform resample on categorical ones. Returns a
    FEASIBLE genome or raises; callers treat a raise as "mutation produced nothing usable"
    and retry, which keeps the infeasible region genuinely unexplored."""
    fields: dict[str, Any] = {}
    if rng.random() < rate:
        fields["signal_family"] = str(rng.choice(SIGNAL_FAMILIES))
    if rng.random() < rate:
        fields["lookback_bars"] = int(rng.choice(LOOKBACK_CHOICES))
    if rng.random() < rate:
        fields["entry_threshold"] = _jitter(genome.entry_threshold, *ENTRY_THRESHOLD_RANGE, rng=rng)
    if rng.random() < rate:
        fields["target_r"] = _jitter(genome.target_r, *TARGET_R_RANGE, rng=rng)
    if rng.random() < rate:
        fields["trail_enabled"] = not genome.trail_enabled
    if rng.random() < rate:
        fields["max_hold_bars"] = int(rng.choice(MAX_HOLD_CHOICES))
    if rng.random() < rate:
        fields["allow_short"] = not genome.allow_short
    if rng.random() < rate:
        fields["symbols"] = SYMBOL_SETS[int(rng.integers(len(SYMBOL_SETS)))]
    if rng.random() < rate:
        fields["max_concurrent"] = int(rng.choice(CONCURRENCY_CHOICES))
    if rng.random() < rate:
        fields["cooldown_bars_after_loss"] = int(rng.choice(COOLDOWN_CHOICES))
    if rng.random() < rate:
        fields["sleeve_fraction"] = float(rng.choice(SLEEVE_FRACTION_CHOICES))

    risk_frac = genome.risk_frac
    stop_pct = genome.stop_pct
    if rng.random() < rate:
        risk_frac = _jitter(risk_frac, *RISK_FRAC_RANGE, rng=rng)
    if rng.random() < rate:
        stop_pct = _jitter(stop_pct, *STOP_PCT_RANGE, rng=rng)
    stop_pct = float(np.clip(stop_pct, _min_stop_for_risk(risk_frac), _max_stop_for_risk(risk_frac)))
    fields["risk_frac"] = risk_frac
    fields["stop_pct"] = stop_pct
    return replace(genome, **fields).validate()


def crossover(left: Genome, right: Genome, rng: np.random.Generator) -> Genome:
    """Uniform crossover. The (risk_frac, stop_pct) pair is inherited as a UNIT, because the
    feasibility constraint couples them -- mixing one parent's risk with the other's stop
    would mostly produce infeasible children and bias the search toward whichever parent had
    the looser stop."""
    fields: dict[str, Any] = {}
    for name in (
        "signal_family",
        "lookback_bars",
        "entry_threshold",
        "target_r",
        "trail_enabled",
        "max_hold_bars",
        "allow_short",
        "symbols",
        "max_concurrent",
        "cooldown_bars_after_loss",
        "sleeve_fraction",
    ):
        fields[name] = getattr(left if rng.random() < 0.5 else right, name)
    donor = left if rng.random() < 0.5 else right
    fields["risk_frac"] = donor.risk_frac
    fields["stop_pct"] = donor.stop_pct
    return Genome(**fields).validate()
