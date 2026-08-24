"""OI-VWAP-SMC integrated strategy: entry gating and signal boosting.

Combines three alpha sources into the existing regime-switching pipeline:
  1. OI-Price Divergence: hard gate against fake breakouts
  2. VWAP Deviation: entry timing optimizer (pullback in trend, mean-reversion in range)
  3. SMC (FVG + Order Block + BOS/CHoCH): structural confirmation

Integration points:
  - Hard gate: blocks entry when OI divergence signals fake breakout
  - Soft gate: reduces position size when SMC structure is weak
  - Size boost: increases size when all three signals align
  - VWAP timing: blocks ranged entries far from VWAP
"""

from __future__ import annotations

from dataclasses import dataclass

from quant_binance.models import FeatureVector
from quant_binance.settings import Settings


@dataclass(frozen=True)
class OIVWAPSMCResult:
    """Result of the OI-VWAP-SMC strategy evaluation."""
    entry_ok: bool
    reasons: tuple[str, ...]
    size_multiplier: float
    boost_reasons: tuple[str, ...]
    signal_quality: str  # "strong", "moderate", "weak", "reject"


# ---------------------------------------------------------------------------
# Thresholds (conservative defaults, tunable per backtest)
# ---------------------------------------------------------------------------

OI_HARD_REJECT: float = -0.5       # below this = definite fake breakout
OI_SOFT_WARN: float = -0.3         # below this = suspicious, reduce size
OI_CONFIRM_BOOST: float = 0.4      # above this = OI confirms direction

SMC_STRONG: float = 0.5            # above this = strong structural setup
SMC_MODERATE: float = 0.25         # above this = acceptable structure
SMC_WEAK_SIZE_MULT: float = 0.6    # size reduction when SMC < MODERATE

VWAP_EXTREME_Z: float = 2.5        # far from VWAP in ranging market
VWAP_PULLBACK_ZONE: tuple[float, float] = (-1.8, -0.4)  # trend pullback zone
ADX_RANGING_THRESHOLD: float = 18.0  # below this = ranging market

ALL_ALIGNED_BOOST: float = 1.3     # size boost when OI + SMC + VWAP all confirm


def evaluate_entry(
    features: FeatureVector,
    symbol: str,
) -> OIVWAPSMCResult:
    """Evaluate entry using OI-VWAP-SMC signals.

    Returns OIVWAPSMCResult with:
      - entry_ok: whether the trade is allowed
      - reasons: list of rejection/warning reasons
      - size_multiplier: 1.0 = normal, <1.0 = reduce, >1.0 = boost
      - boost_reasons: reasons for size increase
      - signal_quality: "strong", "moderate", "weak", "reject"
    """
    reasons: list[str] = []
    boost_reasons: list[str] = []
    size_multiplier = 1.0

    # ── 1. OI Divergence Gate ──────────────────────
    oi = features.oi_divergence_score

    if oi < OI_HARD_REJECT:
        return OIVWAPSMCResult(
            entry_ok=False,
            reasons=("OI_FAKE_BREAKOUT",),
            size_multiplier=0.0,
            boost_reasons=(),
            signal_quality="reject",
        )

    if oi < OI_SOFT_WARN:
        reasons.append("OI_DIVERGENCE_WARN")
        size_multiplier *= 0.7

    oi_confirms = oi >= OI_CONFIRM_BOOST

    # ── 2. SMC Structure Gate ──────────────────────
    smc = features.smc_composite_score

    if smc >= SMC_STRONG:
        boost_reasons.append("SMC_STRONG_STRUCTURE")
    elif smc < SMC_MODERATE:
        reasons.append("SMC_WEAK_STRUCTURE")
        size_multiplier *= SMC_WEAK_SIZE_MULT

    smc_confirms = smc >= SMC_MODERATE

    # ── 3. VWAP Timing Gate ────────────────────────
    vwap_z = features.vwap_deviation_z
    adx = features.adx_1h
    is_ranging = adx < ADX_RANGING_THRESHOLD
    trend_dir = features.trend_direction

    if is_ranging:
        # In ranging market: block entries far from VWAP
        if abs(vwap_z) > VWAP_EXTREME_Z:
            reasons.append("VWAP_EXTREME_RANGING")
            size_multiplier *= 0.5
    else:
        # In trending market: prefer pullback-to-VWAP entries
        if trend_dir > 0 and VWAP_PULLBACK_ZONE[0] <= vwap_z <= VWAP_PULLBACK_ZONE[1]:
            boost_reasons.append("VWAP_TREND_PULLBACK")
        elif trend_dir < 0 and -VWAP_PULLBACK_ZONE[1] <= vwap_z <= -VWAP_PULLBACK_ZONE[0]:
            boost_reasons.append("VWAP_TREND_PULLBACK")

    vwap_favorable = "VWAP_TREND_PULLBACK" in boost_reasons or (not is_ranging and abs(vwap_z) < 1.5)

    # ── 4. Combined Assessment ─────────────────────
    aligned_count = sum([oi_confirms, smc_confirms, vwap_favorable])

    if aligned_count >= 3:
        signal_quality = "strong"
        size_multiplier = max(size_multiplier, ALL_ALIGNED_BOOST)
        boost_reasons.append("ALL_SIGNALS_ALIGNED")
    elif aligned_count >= 2:
        signal_quality = "moderate"
    elif aligned_count >= 1:
        signal_quality = "weak"
    else:
        signal_quality = "weak"

    return OIVWAPSMCResult(
        entry_ok=True,
        reasons=tuple(reasons),
        size_multiplier=round(size_multiplier, 6),
        boost_reasons=tuple(boost_reasons),
        signal_quality=signal_quality,
    )


def should_block_entry(features: FeatureVector) -> tuple[bool, str]:
    """Quick check: should this trade be blocked outright?

    Returns (block, reason). Use this for fast rejection before full evaluation.
    """
    if features.oi_divergence_score < OI_HARD_REJECT:
        return True, "OI_FAKE_BREAKOUT"
    return False, ""
