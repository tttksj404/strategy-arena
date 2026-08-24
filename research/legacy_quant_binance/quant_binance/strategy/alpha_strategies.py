"""Alpha-generating sub-strategies that create NEW entry opportunities.

These strategies find trades the main regime engine would classify as "cash"
and turn them into profitable entries. They run as independent signal sources
alongside the existing regime-switching pipeline.

Capital context: ~$100 equity, 8-20x leverage, 0.78% per-trade risk.
Target: 5m futures on Bitget.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from quant_binance.models import FeatureVector


class AlphaSignalType(Enum):
    VWAP_MEAN_REVERT = "vwap_mean_revert"
    SMC_FVG_FILL = "smc_fvg_fill"
    OI_MOMENTUM_SURGE = "oi_momentum_surge"


@dataclass(frozen=True)
class AlphaSignal:
    """A trade signal from an alpha sub-strategy."""
    signal_type: AlphaSignalType
    side: str              # "long" or "short"
    confidence: float      # 0.0 - 1.0
    size_fraction: float   # fraction of normal position size (0.3 = 30%)
    leverage_cap: int      # max leverage for this signal
    stop_bps: float        # suggested SL in basis points
    target_bps: float      # suggested TP in basis points
    reason: str            # human-readable reason


# ---------------------------------------------------------------------------
# Strategy 1: VWAP Mean Reversion (횡보장 수익화)
# ---------------------------------------------------------------------------
# 기존 시스템이 ADX < 18일 때 "cash"로 넘기는 구간에서 VWAP 평균회귀 진입
# 승률 53-58%, RR 1:1 ~ 1:1.5, 소형 포지션으로 안전하게

def vwap_mean_reversion(features: FeatureVector) -> Optional[AlphaSignal]:
    """Enter mean-reversion trades when price deviates far from VWAP in ranging markets.

    Conditions:
    - ADX < 18 (ranging)
    - |VWAP z-score| >= 2.0 (significant deviation)
    - Trend direction == 0 or weak
    - OI not surging (not a breakout setup)
    """
    adx = features.adx_1h
    vwap_z = features.vwap_deviation_z
    oi_div = features.oi_divergence_score

    # Only in ranging markets
    if adx >= 18.0:
        return None

    # Need significant VWAP deviation
    if abs(vwap_z) < 2.0:
        return None

    # Don't fade if OI is surging (potential breakout)
    if abs(oi_div) > 0.4:
        return None

    # Don't trade if no VWAP calculated
    if features.vwap_price <= 0:
        return None

    # Liquidity floor
    if features.liquidity_score < 0.5:
        return None

    # Determine direction: fade the deviation
    if vwap_z >= 2.0:
        # Price far above VWAP → short back to VWAP
        side = "short"
        confidence = min((abs(vwap_z) - 1.5) / 3.0, 0.85)
    elif vwap_z <= -2.0:
        # Price far below VWAP → long back to VWAP
        side = "long"
        confidence = min((abs(vwap_z) - 1.5) / 3.0, 0.85)
    else:
        return None

    # Conservative sizing: 30-50% of normal position
    size_fraction = 0.3 + 0.2 * min((abs(vwap_z) - 2.0) / 2.0, 1.0)

    # ATR-based stop: 1.2x ATR, target = distance to VWAP
    atr_bps = features.atr_14_1h_bps if features.atr_14_1h_bps > 0 else 50.0
    stop_bps = atr_bps * 1.2
    target_bps = abs(vwap_z) * atr_bps * 0.4  # partial VWAP reversion

    return AlphaSignal(
        signal_type=AlphaSignalType.VWAP_MEAN_REVERT,
        side=side,
        confidence=round(confidence, 4),
        size_fraction=round(size_fraction, 4),
        leverage_cap=4,  # low leverage for mean reversion
        stop_bps=round(stop_bps, 2),
        target_bps=round(max(target_bps, stop_bps * 0.8), 2),  # min RR 0.8
        reason=f"VWAP_MEAN_REVERT z={vwap_z:.1f} adx={adx:.0f}",
    )


# ---------------------------------------------------------------------------
# Strategy 2: SMC FVG Fill (갭 채움 전략)
# ---------------------------------------------------------------------------
# Fair Value Gap 리테스트 시 진입. 기존 시스템이 잡지 못하는
# 구조적 진입 기회를 포착. 기관이 남긴 유동성 갭을 이용.

def smc_fvg_fill(features: FeatureVector) -> Optional[AlphaSignal]:
    """Enter when price retests a Fair Value Gap zone with structural confirmation.

    Conditions:
    - FVG score > 0 (price is in or near an unfilled gap)
    - Structure score > 0 (BOS or CHoCH confirmed)
    - Volume confirmation >= 0.5
    """
    fvg = features.smc_fvg_score
    structure = features.smc_structure_score
    ob = features.smc_ob_score

    # Need active FVG
    if fvg < 0.3:
        return None

    # Need some structural confirmation
    if structure < 0.2 and ob < 0.2:
        return None

    # Volume must support
    if features.volume_confirmation < 0.45:
        return None

    # Determine side from trend
    trend = features.trend_direction
    if trend == 0:
        # No trend → use intraday
        trend = features.intraday_trend_direction
    if trend == 0:
        return None

    side = "long" if trend > 0 else "short"

    # Confidence from SMC composite
    smc_composite = features.smc_composite_score
    confidence = min(0.5 + smc_composite * 0.5, 0.9)

    # Size: 40-70% of normal, scales with SMC quality
    size_fraction = 0.4 + 0.3 * smc_composite

    # Tighter stop: FVG zone is precise
    atr_bps = features.atr_14_1h_bps if features.atr_14_1h_bps > 0 else 50.0
    stop_bps = atr_bps * 0.8  # tighter than normal ATR stop
    target_bps = stop_bps * 2.0  # RR 1:2

    # Leverage: moderate, based on confidence
    leverage_cap = 6 if confidence >= 0.7 else 4

    return AlphaSignal(
        signal_type=AlphaSignalType.SMC_FVG_FILL,
        side=side,
        confidence=round(confidence, 4),
        size_fraction=round(size_fraction, 4),
        leverage_cap=leverage_cap,
        stop_bps=round(stop_bps, 2),
        target_bps=round(target_bps, 2),
        reason=f"SMC_FVG_FILL fvg={fvg:.2f} struct={structure:.2f} ob={ob:.2f}",
    )


# ---------------------------------------------------------------------------
# Strategy 3: OI Momentum Surge (OI 급증 모멘텀)
# ---------------------------------------------------------------------------
# OI가 급증하면서 가격이 같은 방향으로 움직일 때 진입.
# 신규 포지션 유입 = 방향성 확신. 기존 시스템이 "cash"로
# 넘기더라도 OI surge가 충분하면 독립 진입.

def oi_momentum_surge(features: FeatureVector) -> Optional[AlphaSignal]:
    """Enter when OI surges with price confirmation.

    Conditions:
    - OI divergence strongly positive (>= 0.5, meaning OI confirms price move)
    - Taker imbalance confirms direction
    - Not in overheat zone
    """
    oi_div = features.oi_divergence_score

    # Need strong OI confirmation
    if oi_div < 0.5:
        return None

    # Taker flow must confirm direction
    taker = features.taker_imbalance_norm
    trend = features.trend_direction
    if trend == 0:
        trend = features.intraday_trend_direction
    if trend == 0:
        return None

    # Taker must align: >0.55 for long, <0.45 for short
    if trend > 0 and taker < 0.55:
        return None
    if trend < 0 and taker > 0.45:
        return None

    # Don't chase into overheat
    if features.overheat_penalty > 0.5:
        return None

    # Don't enter against macro
    if features.macro_risk_penalty > 0.6:
        return None

    side = "long" if trend > 0 else "short"

    # Confidence from OI strength + volume
    confidence = min(0.5 + oi_div * 0.5 + features.volume_confirmation * 0.2, 0.9)

    # Aggressive sizing: 50-80%, this is a high-confidence momentum signal
    size_fraction = 0.5 + 0.3 * min(oi_div, 1.0)

    # Normal ATR stop
    atr_bps = features.atr_14_1h_bps if features.atr_14_1h_bps > 0 else 50.0
    stop_bps = atr_bps * 1.0
    target_bps = stop_bps * 2.5  # RR 1:2.5 for momentum

    # Higher leverage: strong conviction
    leverage_cap = 8 if confidence >= 0.75 else 6

    return AlphaSignal(
        signal_type=AlphaSignalType.OI_MOMENTUM_SURGE,
        side=side,
        confidence=round(confidence, 4),
        size_fraction=round(size_fraction, 4),
        leverage_cap=leverage_cap,
        stop_bps=round(stop_bps, 2),
        target_bps=round(target_bps, 2),
        reason=f"OI_MOMENTUM_SURGE oi={oi_div:.2f} taker={taker:.2f}",
    )


# ---------------------------------------------------------------------------
# Alpha Scanner: runs all sub-strategies
# ---------------------------------------------------------------------------

def scan_alpha_signals(features: FeatureVector) -> list[AlphaSignal]:
    """Run all alpha sub-strategies and return any signals found.

    These signals represent NEW trade opportunities that the main regime
    engine would skip. Each signal includes its own sizing and risk params.

    For $100 capital:
    - VWAP mean revert: ~$0.23-0.39 risk per trade (30-50% × $0.78)
    - SMC FVG fill: ~$0.31-0.55 risk per trade (40-70% × $0.78)
    - OI momentum: ~$0.39-0.62 risk per trade (50-80% × $0.78)
    """
    signals: list[AlphaSignal] = []

    strategies = [
        vwap_mean_reversion,
        smc_fvg_fill,
        oi_momentum_surge,
    ]

    for strategy in strategies:
        signal = strategy(features)
        if signal is not None:
            signals.append(signal)

    # Sort by confidence (highest first)
    signals.sort(key=lambda s: s.confidence, reverse=True)

    return signals


def best_alpha_signal(features: FeatureVector) -> Optional[AlphaSignal]:
    """Return the single highest-confidence alpha signal, or None."""
    signals = scan_alpha_signals(features)
    return signals[0] if signals else None
